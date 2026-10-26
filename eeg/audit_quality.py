"""Sleep-EDF SC first-night signal audit. Diagnostic only; no training or downloads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

VERSION = '0.5.0-audit'
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_header(path):
    with Path(path).open('rb') as f:
        h = f.read(256)
        if len(h) != 256 or h[:8].strip() != b'0':
            raise ValueError('16-bit EDF 파일이 아닙니다.')
        if b'EDF+D' in h[192:236]:
            raise ValueError('시간이 불연속적인 EDF+D는 지원하지 않습니다.')
        size, records, duration, n = int(h[184:192]), int(h[236:244]), float(h[244:252]), int(h[252:256])
        if not (0 < n <= 256 and size == 256 * (n + 1) and records > 0 and duration == 30):
            raise ValueError('30초 레코드와 확정된 레코드 수를 가진 SC EDF가 필요합니다.')
        body = f.read(size - 256)
    if len(body) != size - 256:
        raise ValueError('EDF 헤더가 잘렸습니다.')
    pos = 0
    fields = {}
    for key, width in [('label',16),('transducer',80),('unit',8),('pmin',8),('pmax',8),('dmin',8),('dmax',8),('prefilter',80),('samples',8),('reserved',32)]:
        fields[key] = [body[pos+i*width:pos+(i+1)*width].decode('latin1').strip() for i in range(n)]
        pos += n * width
    channels, offset = [], 0
    for i in range(n):
        c = {key: value[i] for key, value in fields.items()}
        for key in ['pmin','pmax']:
            c[key] = float(c[key])
        for key in ['dmin','dmax','samples']:
            c[key] = int(c[key])
        if not (np.isfinite([c['pmin'], c['pmax']]).all() and c['pmax'] > c['pmin'] and -32768 <= c['dmin'] < c['dmax'] <= 32767 and c['samples'] > 0):
            raise ValueError('잘못된 채널 범위/표본 수: ' + c['label'])
        c['offset'] = offset
        c['hz'] = c['samples'] / duration
        offset += c['samples']
        channels.append(c)
    if Path(path).stat().st_size != size + records * offset * 2:
        raise ValueError('EDF 파일 크기가 헤더와 일치하지 않습니다. 잘림/추가 데이터 확인이 필요합니다.')
    for label in CHANNELS:
        matches = [c for c in channels if c['label'] == label]
        if len(matches) != 1:
            raise ValueError('필수 채널이 없거나 중복됨: ' + label)
        c = matches[0]
        if c['unit'] not in ('uV', 'µV', 'mV', 'V'):
            raise ValueError('지원하지 않는 전압 단위: ' + repr(c['unit']))
        if c['hz'] != (1 if label == CHANNELS[3] else 100):
            raise ValueError('SC 전용: EEG/EOG 100Hz, EMG 1Hz가 필요합니다.')
    return {'header_bytes': size, 'records': records, 'record_seconds': duration, 'samples_per_record': offset, 'channels': channels}


def longest(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if len(starts) else 0


def metrics(d, c):
    scale = {'uV':1., 'µV':1., 'mV':1000., 'V':1e6}[c['unit']]
    x = ((d.astype(float) - c['dmin']) * (c['pmax']-c['pmin']) / (c['dmax']-c['dmin']) + c['pmin']) * scale
    low, high = d == c['dmin'], d == c['dmax']
    std, ptp = float(np.std(x)), float(np.ptp(x))
    return {'ptp_uv': ptp, 'p99_p01_uv': float(np.percentile(x,99)-np.percentile(x,1)),
            'std_uv': std, 'rail_samples': int((low | high).sum()),
            'rail_fraction': float((low | high).mean()),
            'longest_same_rail_seconds': max(longest(low), longest(high)) / c['hz'],
            'out_of_header_samples': int(((d < c['dmin']) | (d > c['dmax'])).sum()),
            'constant': bool(np.all(d == d[0])),
            'legacy_high_amplitude': bool(ptp >= 1000) if c['label'] != CHANNELS[3] else False,
            'legacy_flat': bool(std <= .01) if c['label'] != CHANNELS[3] else False}


def audit(path, subject, writer):
    h = read_header(path)
    raw = np.memmap(path, dtype='<i2', mode='r', offset=h['header_bytes'], shape=(h['records'],h['samples_per_record']))
    selected = [next(c for c in h['channels'] if c['label'] == label) for label in CHANNELS]
    aggregates = {c['label']: [] for c in selected}
    legacy_bad, rail_pass = [], {label: 0 for label in CHANNELS}
    for epoch in range(h['records']):
        rows = []
        for c in selected:
            d = raw[epoch, c['offset']:c['offset']+c['samples']]
            m = metrics(d, c)
            rows.append({'subject':subject, 'file':Path(path).name, 'epoch':epoch, 'start_seconds':epoch*30, 'channel':c['label'], **m})
        bad = any(r['legacy_high_amplitude'] or r['legacy_flat'] for r in rows)
        legacy_bad.append(bad)
        for r in rows:
            r['legacy_epoch_pass'] = not bad
            writer.writerow(r)
            aggregates[r['channel']].append(r)
            if not bad and r['rail_samples']:
                rail_pass[r['channel']] += 1
    del raw
    summaries = []
    for c in selected:
        rows = aggregates[c['label']]
        summaries.append({'subject': subject, 'file':Path(path).name, 'channel':c['label'], 'native_hz':c['hz'],
            'header_unit':c['unit'], 'header_physical_min':c['pmin'], 'header_physical_max':c['pmax'],
            'header_digital_min':c['dmin'], 'header_digital_max':c['dmax'], 'epochs':len(rows),
            'ptp_uv_median':float(np.median([r['ptp_uv'] for r in rows])), 'ptp_uv_max':max(r['ptp_uv'] for r in rows),
            'legacy_high_amplitude_epochs':sum(r['legacy_high_amplitude'] for r in rows),
            'legacy_flat_epochs':sum(r['legacy_flat'] for r in rows),
            'constant_epochs':sum(r['constant'] for r in rows),
            'rail_contact_epochs':sum(r['rail_samples']>0 for r in rows),
            'rail_samples':sum(r['rail_samples'] for r in rows),
            'rail_fraction_all_samples':sum(r['rail_samples'] for r in rows)/(len(rows)*c['samples']),
            'rail_contact_legacy_pass_epochs':rail_pass[c['label']],
            'max_same_rail_seconds_within_epoch':max(r['longest_same_rail_seconds'] for r in rows),
            'out_of_header_samples':sum(r['out_of_header_samples'] for r in rows)})
    return {'subject':subject, 'file':Path(path).name, 'sha256':sha256(path), 'epochs':h['records'],
            'legacy_equivalent_excluded_epochs':sum(legacy_bad), 'longest_legacy_exclusion_seconds':longest(legacy_bad)*30,
            'channels':summaries}


def discover(root, subjects):
    files = []
    for subject in subjects:
        if not 0 <= subject <= 9:
            raise ValueError('이번 도구의 참가자 번호는 0~9입니다.')
        found = sorted(Path(root).rglob(f'SC4{subject:02d}1*-PSG.edf'))
        if len(found) != 1:
            raise ValueError(f'참가자 {subject}: 첫날 PSG 파일이 정확히 1개 필요합니다. 발견 {len(found)}개 ({root})')
        files.append((subject, found[0]))
    return files


def main():
    p = argparse.ArgumentParser(description='SC 참가자 0~9 첫날 채널 품질 진단. 다운로드/재학습 없음.')
    p.add_argument('--data-dir', default='eeg_data')
    p.add_argument('--subjects', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--out', default='output/quality_audit_01')
    a = p.parse_args()
    out = Path(a.out)
    try:
        if len(set(a.subjects)) != len(a.subjects):
            raise ValueError('참가자 번호가 중복됐습니다.')
        files = discover(a.data_dir, a.subjects)
        for _, path in files:
            read_header(path)
        out.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as e:
        p.exit(2, f'품질 점검 시작 불가: {e}\n새 출력 폴더와 원본 경로를 확인하세요.\n')
    summary = {'version': VERSION, 'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
        'requested_subjects':a.subjects, 'processed_subjects':[], 'source_sha256':sha256(__file__),
        'numpy_version':np.__version__, 'python_version':sys.version.split()[0],
        'scope':'all 30-second PSG records; no hypnogram filtering; native rates; no resampling',
        'legacy_comparison':'EEG/EOG std <= 0.01 uV or peak-to-peak >= 1000 uV; EMG finite only. All int16 samples are finite.',
        'limitations':['Diagnostic only: rail contact is not an automatic exclusion.', 'Longest rail run is within each 30-second epoch, not across boundaries.', 'No sleep-stage labels or accuracy evaluation. Counts may differ from annotation-filtered training epochs.', 'SHA256 identifies the input; it is not a comparison with an official checksum.'], 'records':[]}
    target = out/'quality_audit_summary.json'
    def save():
        target.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    save()
    fields = ['subject','file','epoch','start_seconds','channel','ptp_uv','p99_p01_uv','std_uv','rail_samples','rail_fraction','longest_same_rail_seconds','out_of_header_samples','constant','legacy_high_amplitude','legacy_flat','legacy_epoch_pass']
    try:
        with (out/'quality_audit_epochs.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for subject,path in files:
                print(f'[점검 중] 참가자 {subject}: {path.name}',flush=True)
                r = audit(path,subject,writer)
                summary['records'].append(r); summary['processed_subjects'].append(subject); save()
                print(f"[점검 완료] {r['epochs']}구간, 기존 기준 제외 {r['legacy_equivalent_excluded_epochs']}구간",flush=True)
        rows = [c for r in summary['records'] for c in r['channels']]
        with (out/'quality_audit_channels.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summary['status']='completed'
        summary['finished_utc']=datetime.now(timezone.utc).isoformat(); save()
        print(f'완료: {out.resolve()}\n공유할 파일: quality_audit_summary.json, quality_audit_channels.csv',flush=True)
    except Exception as e:
        summary['status']='failed'; summary['error']=str(e); save()
        raise


if __name__ == '__main__':
    main()
