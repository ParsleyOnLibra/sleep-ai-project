"""Causal W->N1 alerts; retrospective scoring is separate from detection."""
import numpy as np


def reference_events(y, times, valid):
    return [i for i in range(1,len(y)) if valid[i-1] and valid[i]
            and times[i]-times[i-1]==30 and y[i-1]==0 and y[i]==1]


def detect(p, times, valid, threshold=.5, wake_epochs=2, n1_epochs=2):
    """Alerts at epoch END. Arm after predicted wake, confirm consecutive N1.
    No labels, future epochs, whole-night normalization, or backdated decisions.
    """
    alerts=[]; wake=0; n1=0; armed=False
    for i, row in enumerate(p):
        if not valid[i] or (i and times[i]-times[i-1]!=30):
            wake=n1=0; armed=False
            if not valid[i]:continue
        if row.argmax()==0:
            wake+=1; n1=0
            if wake>=wake_epochs:armed=True
        elif armed and row[1]>=threshold:
            wake=0; n1+=1
            if n1>=n1_epochs:
                alerts.append(i); armed=False; n1=0
        else:
            wake=n1=0; armed=False
    return alerts


def evaluate(y, p, times, valid, threshold=.5, tolerance=120):
    refs=reference_events(y,times,valid)
    alerts=detect(p,times,valid,threshold)
    unmatched=set(refs); matched=[]; false=[]
    for a in alerts:
        decision=times[a]+30
        candidates=[r for r in unmatched if r <= a and 0<decision-times[r]<=tolerance
                    and np.all(valid[r:a+1]) and np.all(np.diff(times[r:a+1])==30)]
        if candidates:
            r=min(candidates,key=lambda r: decision-times[r]); unmatched.remove(r)
            matched.append({'reference_start_s':float(times[r]),'decision_s':float(decision),
                            'delay_s':float(decision-times[r]), 'evidence_end_s':float(times[a]+30)})
        else:false.append(float(decision))
    hours=float(valid.sum()*30/3600)
    tp=len(matched); fp=len(false); fn=len(unmatched)
    return dict(reference_events=len(refs),alerts=len(alerts),matched=tp,missed=fn,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                unmatched_alerts_per_valid_hour=fp/hours if hours else None,
                median_delay_s=float(np.median([x['delay_s'] for x in matched])) if matched else None,
                valid_hours=hours,details=matched,unmatched_decision_s=false,
                missed_reference_start_s=[float(times[i]) for i in sorted(unmatched)])
