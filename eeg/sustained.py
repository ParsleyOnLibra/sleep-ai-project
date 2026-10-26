"""Exploratory continuous-sleep bouts. Decisions occur at epoch ends.

All sleep stages count. Wake, missing epochs and QC abstentions reset a bout.
No lights-off marker is available, so these are not clinical sleep latencies.
"""
import numpy as np


def bouts(stages, times, valid, minutes):
    required = int(minutes * 2)
    if required < 1 or required != minutes * 2:
        raise ValueError('Duration must be a positive multiple of 0.5 minutes')
    result = []
    start = None
    count = 0
    for i, stage in enumerate(stages):
        if i and times[i] - times[i-1] != 30:
            start = None
            count = 0
        if not valid[i] or stage not in (1, 2, 3, 4):
            start = None
            count = 0
            continue
        if start is None:
            start = i
        count += 1
        if count == required:
            result.append(dict(start_s=float(times[start]),
                               confirmation_s=float(times[i]+30),
                               end_s=float(times[i]+30)))
        elif count > required:
            result[-1]['end_s'] = float(times[i]+30)
    return result


def evaluate_sustained(y, p, times, valid, minutes, onset_tolerance=120):
    refs = bouts(y, times, valid, minutes)
    predictions = bouts(np.argmax(p, axis=1), times, valid, minutes)
    remaining = set(range(len(refs)))
    matches = []
    false = []
    for pred in predictions:
        candidates = [j for j in remaining
                      if abs(pred['start_s']-refs[j]['start_s']) <= onset_tolerance
                      and refs[j]['confirmation_s'] <= pred['confirmation_s'] <= refs[j]['end_s']]
        if not candidates:
            false.append(pred)
            continue
        j = min(candidates, key=lambda j: abs(pred['start_s']-refs[j]['start_s']))
        remaining.remove(j)
        ref = refs[j]
        matches.append(dict(reference_start_s=ref['start_s'],
                            estimated_start_s=pred['start_s'],
                            confirmation_s=pred['confirmation_s'],
                            start_error_s=pred['start_s']-ref['start_s'],
                            detection_delay_s=pred['confirmation_s']-ref['start_s']))
    return dict(minutes=minutes, reference_bouts=len(refs), predicted_bouts=len(predictions),
                matched=len(matches), missed=len(remaining),
                precision=len(matches)/len(predictions) if predictions else None,
                recall=len(matches)/len(refs) if refs else None,
                details=matches, unmatched_predictions=false,
                missed_references=[refs[j] for j in sorted(remaining)],
                first_reference=refs[0] if refs else None,
                first_prediction=predictions[0] if predictions else None)
