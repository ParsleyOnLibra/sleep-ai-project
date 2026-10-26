"""Re-score saved predictions without training; preserves the old selection."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from transitions import evaluate
from sustained import evaluate_sustained

CLASSES = ['Wake', 'N1', 'N2', 'N3', 'REM']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use a new output directory')
    threshold = json.loads((args.input/'selection.json').read_text())['threshold']
    groups = defaultdict(list)
    with (args.input/'predictions.csv').open() as f:
        for row in csv.DictReader(f):
            groups[(row['subject'], row['night'])].append(row)
    result = []
    for (subject, night), rows in groups.items():
        times = np.array([float(r['epoch_start_s']) for r in rows])
        if np.any(np.diff(times) <= 0):
            raise ValueError('Epoch times must increase within each record')
        y = np.array([CLASSES.index(r['true_stage']) for r in rows])
        valid = np.array([bool(int(r['qc_valid'])) for r in rows])
        p = np.array([[float(r['p_'+c]) for c in CLASSES] for r in rows])
        result.append(dict(subject=subject, night=night,
                           transitions=evaluate(y,p,times,valid,threshold),
                           sustained=[evaluate_sustained(y,p,times,valid,m) for m in [2,5,10]]))
    args.out.mkdir(parents=True, exist_ok=True)
    data = dict(exploratory=True, retrained=False, threshold=threshold,
                limitations='Old validation-selected threshold retained. QC reasons require raw EDF re-extraction. Sustained matching: onset within 120s; confirmation no earlier than reference confirmation and no later than its bout end. All criteria reported; none selected on test.',
                records=result)
    (args.out/'review.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
    print(json.dumps({key:sum(r['transitions'][key] for r in result)
                      for key in ['reference_events','alerts','matched','missed']},indent=2))

if __name__ == '__main__':
    main()
