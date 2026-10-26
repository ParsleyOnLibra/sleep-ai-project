"""Render the preserved experimental results; no EDF download or model training."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'results' / 'classification'
STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM']
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': '#fafbfc'})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    metrics = json.loads((DATA / 'metrics.json').read_text())['test']
    variants = ['baseline', 'multimodal', 'causal']
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (key, label, color) in enumerate([
        ('accuracy', 'Accuracy', '#283b63'),
        ('balanced_accuracy', 'Balanced accuracy', '#278f91'),
        ('macro_f1', 'Macro F1', '#d58639'),
    ]):
        bars = ax.bar(np.arange(3) + (i - 1) * .25,
                      [metrics[v][key] * 100 for v in variants], .25,
                      label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)
    ax.set_xticks(range(3), ['EEG baseline', 'Multimodal', 'Multimodal + past context'])
    ax.set_ylim(0, 110)
    ax.set_ylabel('Score (%)')
    ax.set_title('Held-out subjects 5 & 6 | QC-passed epochs only (4,275 / 5,492)', pad=16)
    ax.legend(loc='lower right')
    save(fig, 'model_comparison.png')

    cm = np.array(metrics['causal']['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap='Blues')
    for i in range(5):
        for j in range(5):
            ratio = cm[i, j] / cm[i].sum()
            ax.text(j, i, f'{cm[i,j]}\n{ratio:.1%}', ha='center', va='center',
                    color='white' if ratio > .55 else '#182235', fontsize=10)
    ax.set_xticks(range(5), STAGES)
    ax.set_yticks(range(5), STAGES)
    ax.set_xlabel('Predicted stage')
    ax.set_ylabel('Reference stage')
    ax.set_title('Past-context model | Counts + row percentages')
    fig.colorbar(im, ax=ax, label='Fraction within reference stage')
    save(fig, 'confusion_matrix.png')

    rows = list(csv.DictReader((DATA / 'predictions.csv').open()))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, subject in zip(axes, ['5', '6']):
        r = [x for x in rows if x['subject'] == subject]
        times = np.array([float(x['epoch_start_s']) / 3600 for x in r])
        valid = np.array([int(x['qc_valid']) == 1 for x in r])
        truth = np.array([STAGES.index(x['true_stage']) for x in r])
        pred = np.array([STAGES.index(x['prediction']) if x['prediction'] in STAGES else np.nan for x in r], dtype=float)
        pred[~valid] = np.nan
        ax.step(times, truth, where='post', label='Reference', lw=1.4, color='#243e63')
        ax.step(times, pred, where='post', label='Prediction (QC passed)', lw=.9, color='#d4772f', alpha=.85)
        for t in times[~valid]:
            ax.axvspan(t, t + 30/3600, color='#b8bec8', alpha=.35, lw=0)
        ax.set_yticks(range(5), STAGES)
        ax.invert_yaxis()
        ax.set_title(f'Subject {subject} | elapsed time from recording start')
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, 1.4), ncol=2)
    axes[1].set_xlabel('Elapsed hours (not clock time)')
    fig.tight_layout()
    save(fig, 'hypnogram.png')

    qc = json.loads((ROOT / 'results/quality/quality_audit_summary.json').read_text())['records']
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([str(x['subject']) for x in qc],
                  [x['legacy_equivalent_excluded_epochs'] for x in qc], color='#b96245')
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, 1400)
    ax.set_xlabel('Subject')
    ax.set_ylabel('30-second PSG records')
    ax.set_title('Legacy exclusion criterion | All PSG records, before label alignment')
    save(fig, 'quality_exclusions.png')

    curve = json.loads((DATA / 'learning_curve.json').read_text())
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [x['n_subjects'] for x in curve]
    ys = [100*x['subject_mean_macro_f1'] for x in curve]
    ax.plot(xs, ys, 'o-', color='#278f91')
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.2f}%', (x,y), xytext=(0,10), textcoords='offset points', ha='center')
    ax.set_xticks(xs)
    ax.set_ylim(65,90)
    ax.set_xlabel('Number of training subjects')
    ax.set_ylabel('Mean per-subject Macro F1 (%)')
    ax.set_title('Fixed validation subjects 0 & 7 | exploratory learning curve')
    save(fig, 'learning_curve.png')


if __name__ == '__main__':
    main()
