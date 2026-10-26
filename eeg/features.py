"""Epoch-local spectral features and strictly trailing context. No fitted normalizer."""
import hashlib
import json
from pathlib import Path
import mne
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from data import digest

CLASSES = ['Wake', 'N1', 'N2', 'N3', 'REM']
MAP = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
       'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
BANDS = [('delta', .5, 4), ('theta', 4, 8), ('alpha', 8, 12), ('sigma', 12, 16), ('beta', 16, 30)]


def spectrum(x, fs, extended=True):
    hz, psd = welch(x, fs=fs, nperseg=min(len(x), round(4*fs)))
    select = (hz >= .5) & (hz < 30)
    total = max(trapezoid(psd[select], hz[select]), 1e-20)
    values, names, powers = [], [], []
    for name, lo, hi in BANDS:
        mask = (hz >= lo) & (hz < hi)
        power = max(trapezoid(psd[mask], hz[mask]), 1e-20)
        powers.append(power)
        values += [np.log10(power), power/total]
        names += [f'{name}_logpower', f'{name}_relative']
    values += [np.log10(max(np.std(x), 1e-20))]; names += ['logstd']
    if extended:
        v = max(np.var(x), 1e-20)
        d = np.diff(x); vd = max(np.var(d), 1e-20)
        mobility = np.sqrt(vd/v)
        p = psd[select] / max(psd[select].sum(), 1e-20)
        entropy = -(p * np.log(p + 1e-20)).sum()/np.log(len(p))
        values += [np.ptp(x), np.percentile(x,75)-np.percentile(x,25),
                   skew(x) if np.std(x)>1e-12 else 0,
                   kurtosis(x) if np.std(x)>1e-12 else 0,
                   mobility, np.sqrt(np.var(np.diff(d))/vd)/max(mobility,1e-20),
                   entropy, np.log10(powers[1]/powers[2]), np.log10(powers[3]/powers[4])]
        names += ['ptp','iqr','skew','kurtosis','hjorth_mobility','hjorth_complexity',
                  'spectral_entropy','theta_alpha','sigma_beta']
    return values, names


def trailing_context(x, times, valid, window=4):
    """Current epoch plus <=3 previous epochs; reset at gaps and bad signal."""
    out = []; start = 0
    for i in range(len(x)):
        if i == 0 or times[i]-times[i-1] != 30 or not valid[i] or not valid[i-1]:
            start = i
        history = x[max(start, i-window+1):i+1]
        previous = x[i-1] if i>start else x[i]
        out.append(np.r_[x[i], history.mean(axis=0), x[i]-previous])
    return np.asarray(out)


def quality_flags(signal, muscle):
    reasons = []
    for j, channel in enumerate(CHANNELS[:3]):
        if not np.isfinite(signal[j]).all():
            reasons.append(channel + ':nonfinite')
        else:
            if np.std(signal[j]) <= 1e-8:
                reasons.append(channel + ':flat')
            if np.ptp(signal[j]) >= 1e-3:
                reasons.append(channel + ':high_amplitude')
    if not np.isfinite(muscle).all():
        reasons.append('EMG:nonfinite')
    return ';'.join(reasons)


def load_record(psg, hyp, cache_dir):
    config = {'version': 3, 'psg': digest(psg), 'hyp': digest(hyp), 'mne': mne.__version__,
              'source': digest(__file__)}
    key = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir)/f'{key}.npz'; cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return {k:z[k] for k in z.files}
    raw = mne.io.read_raw_edf(psg, preload=False, verbose='ERROR')
    annotations = mne.read_annotations(hyp)
    raw.set_annotations(annotations, emit_warning=False)
    missing = set(CHANNELS)-set(raw.ch_names)
    if missing: raise ValueError(f'필수 SC 채널 없음: {missing}')
    # Read EMG separately to avoid MNE upsampling a 1 Hz envelope to EEG rate.
    emg = mne.io.read_raw_edf(psg, include=['EMG submental'], preload=False, verbose='ERROR')
    if emg.info['sfreq'] != 1:
        raise ValueError('이 버전은 SC의 1 Hz EMG 포락선 전용입니다.')
    events, _ = mne.events_from_annotations(raw,event_id={k:v+1 for k,v in MAP.items()},chunk_duration=30,verbose='ERROR')
    sf = raw.info['sfreq']; n = round(30*sf)
    baseline, enhanced, labels, times, quality, reasons = [], [], [], [], [], []
    for event in events:
        start = int(event[0]-raw.first_samp)
        if start < 0 or start+n > raw.n_times: continue
        sec = start/sf
        signal = raw.get_data(picks=CHANNELS[:3], start=start, stop=start+n)
        muscle = emg.get_data(start=round(sec),stop=round(sec+30))[0]
        if len(muscle)!=30: continue
        finite = bool(np.isfinite(signal).all() and np.isfinite(muscle).all())
        # Exploratory QC, not clinically validated; flag and abstain, never relabel.
        reason = quality_flags(signal, muscle)
        reasons.append(reason)
        good = not reason
        clean = np.nan_to_num(signal); muscle=np.nan_to_num(muscle)
        b, e, names = [], [], []
        for j,ch in enumerate(CHANNELS[:3]):
            vals, nm = spectrum(clean[j],sf)
            e.extend(vals); names.extend([ch+':'+s for s in nm])
            if j<2: b.extend(spectrum(clean[j],sf,False)[0])
        e.extend([np.mean(muscle),np.std(muscle),np.percentile(muscle,90),np.mean(np.abs(np.diff(muscle)))])
        names.extend(['EMG:mean','EMG:std','EMG:p90','EMG:absdiff'])
        baseline.append(b); enhanced.append(e); labels.append(event[2]-1); times.append(sec); quality.append(good)
    raw.close(); emg.close()
    if not labels: raise ValueError('유효 주석 구간 없음')
    data = dict(baseline=np.asarray(baseline),enhanced=np.asarray(enhanced),y=np.asarray(labels,dtype=int),
                times=np.asarray(times),valid=np.asarray(quality),names=np.asarray(names),qc_reasons=np.asarray(reasons))
    if not np.isfinite(data['enhanced']).all():raise ValueError('비유한 특징')
    tmp=cache.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**data)
    tmp.replace(cache)
    return data
