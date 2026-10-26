#!/usr/bin/env python3
"""Apple Health XML/ZIP -> local sleep CSV and HTML report (stdlib only)."""
import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

SLEEP = 'HKCategoryTypeIdentifierSleepAnalysis'
HEART = 'HKQuantityTypeIdentifierHeartRate'
HRV = 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN'
STAGES = {
 'HKCategoryValueSleepAnalysisAwake':'Awake',
 'HKCategoryValueSleepAnalysisAsleepCore':'Core',
 'HKCategoryValueSleepAnalysisAsleepDeep':'Deep',
 'HKCategoryValueSleepAnalysisAsleepREM':'REM',
 'HKCategoryValueSleepAnalysisAsleepUnspecified':'Unspecified',
 'HKCategoryValueSleepAnalysisAsleep':'Unspecified',
 'HKCategoryValueSleepAnalysisInBed':'InBed',
}
FIELDS = ['date','source','sleep_hours','core_hours','deep_hours','rem_hours','unspecified_hours','awake_hours','heart_rate_avg_bpm','hrv_avg_ms']


def parse_date(s):
    if not s: return None
    try: return datetime.strptime(s, '%Y-%m-%d %H:%M:%S %z')
    except ValueError: return None


def wake_date(dt):
    # Shift noon boundary: 23:00 and next-day 02:00 belong to same night.
    return (dt + timedelta(hours=12)).date().isoformat()


@contextmanager
def open_export(path):
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.namelist() if m == 'export.xml' or m.endswith('/export.xml')]
            if not members: raise ValueError('ZIP 안에 export.xml이 없습니다.')
            with archive.open(members[0]) as fh: yield fh
    else:
        with path.open('rb') as fh: yield fh


def union_hours(intervals):
    if not intervals: return 0.0
    intervals = sorted(intervals)
    merged = []
    for start,end in intervals:
        if not merged or start > merged[-1][1]: merged.append([start,end])
        else: merged[-1][1] = max(end, merged[-1][1])
    return sum((e-s).total_seconds() for s,e in merged)/3600


def extract(path):
    sleeps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    metrics = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    with open_export(path) as f:
        for _, element in ET.iterparse(f, events=('end',)):
            if element.tag != 'Record':
                element.clear()
                continue
            a = element.attrib
            typ = a.get('type')
            if typ not in (SLEEP, HEART, HRV):
                element.clear()
                continue
            st, en = parse_date(a.get('startDate')), parse_date(a.get('endDate'))
            if not st or not en or en <= st:
                element.clear()
                continue
            source = a.get('sourceName','Unknown')
            night = wake_date(st)
            if typ == SLEEP:
                stage = STAGES.get(a.get('value'))
                if stage:
                    # split stages that cross the noon-boundary if needed
                    cur = st
                    while cur < en:
                        next_boundary = cur.replace(hour=12,minute=0,second=0,microsecond=0)
                        if cur >= next_boundary: next_boundary += timedelta(days=1)
                        stop = min(next_boundary,en)
                        sleeps[wake_date(cur)][source][stage].append((cur,stop))
                        cur = stop
                    counts['sleep'] += 1
            else:
                try: val = float(a.get('value',''))
                except ValueError: val = None
                unit = a.get('unit','')
                if val is not None and val >= 0 and ((typ == HEART and unit in ('count/min','bpm')) or (typ == HRV and unit == 'ms')):
                    key = 'hr' if typ == HEART else 'hrv'
                    metrics[night][key].append(val)
                    counts[key] += 1
            element.clear()
    rows=[]
    for night, sources in sorted(sleeps.items()):
        if not sources: continue
        # One source per night, to avoid double-counting Apple Watch + third-party apps.
        def score(item):
            _, stages = item
            return sum(union_hours(stages.get(k,[])) for k in ('Core','Deep','REM','Unspecified'))
        source, stage_map = max(sources.items(),key=score)
        durations = {k:union_hours(stage_map.get(k,[])) for k in ('Core','Deep','REM','Unspecified','Awake')}
        asleep_intervals = sum((stage_map.get(k,[]) for k in ('Core','Deep','REM','Unspecified')),[])
        total_sleep = union_hours(asleep_intervals) # independent of InBed overlaps
        # Avoid interpreting missing measurements as zero.
        row = {'date':night,'source':source,'sleep_hours':round(total_sleep,3) if asleep_intervals else '',
               'core_hours':round(durations['Core'],3) if stage_map.get('Core') else '',
               'deep_hours':round(durations['Deep'],3) if stage_map.get('Deep') else '',
               'rem_hours':round(durations['REM'],3) if stage_map.get('REM') else '',
               'unspecified_hours':round(durations['Unspecified'],3) if stage_map.get('Unspecified') else '',
               'awake_hours':round(durations['Awake'],3) if stage_map.get('Awake') else '',
               'heart_rate_avg_bpm':round(statistics.mean(metrics[night]['hr']),1) if metrics[night]['hr'] else '',
               'hrv_avg_ms':round(statistics.mean(metrics[night]['hrv']),1) if metrics[night]['hrv'] else ''}
        rows.append(row)
    return rows,counts


def render_report(rows, output):
    recent = rows[-30:]
    bar_rows=[]
    for r in recent:
        segments=[]
        for label,color,key in [('Deep','#425bba','deep_hours'),('Core','#6ba6ee','core_hours'),('REM','#ab6cd5','rem_hours'),('Unspecified','#8e9ca5','unspecified_hours')]:
            val=float(r[key]) if r[key] != '' else 0
            if val: segments.append(f'<span title="{label}: {val:.2f}h" style="width:{min(val/12*100,100):.2f}%;background:{color}"></span>')
        bar_rows.append(f'<div class="night"><span>{escape(r["date"])}</span><div class="bar">{"".join(segments)}</div><span>{r["sleep_hours"] if r["sleep_hours"] != "" else "—"}h</span></div>')
    tr=''.join('<tr>'+''.join('<td>'+escape(str(r[k]))+'</td>' for k in FIELDS)+'</tr>' for r in rows[-100:])
    nums=[float(r['sleep_hours']) for r in rows if r['sleep_hours'] != '']
    avg=f'{statistics.mean(nums):.2f}시간' if nums else '계산 불가'
    page=f'''<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sleep AI Research</title><style>
    body{{font:15px system-ui,-apple-system,sans-serif;background:#0d1422;color:#e9edf6;max-width:1000px;margin:40px auto;padding:0 20px}}
    h1{{font-size:28px}}p{{color:#a8b6cd}}.card{{background:#172235;border-radius:16px;padding:22px;margin:18px 0}}.night{{display:grid;grid-template-columns:110px 1fr 70px;gap:12px;align-items:center;margin:10px 0;font-size:13px}}.bar{{display:flex;height:20px;background:#2a3546;border-radius:4px;overflow:hidden}}.bar span{{display:block}}.legend span{{padding-right:18px}}.tablewrap{{overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid #364052;text-align:left}}small{{color:#a8b6cd}}
    </style></head><body><h1>Sleep AI Research</h1><p>Apple Health 로컬 분석 · 파일은 외부 서버로 전송되지 않습니다.</p><div class="card"><strong>수면 기록 {len(rows)}일</strong><p>기록이 있는 밤의 평균 수면 시간: {avg}</p><small>Apple Watch의 추정 단계입니다. 진단용이 아니며, 다른 앱/기기 기록은 밤별 대표 출처 하나만 반영합니다.</small></div><div class="card"><h2>최근 30회 수면 단계</h2><p class="legend"><span>🟦 Deep</span><span>🔹 Core</span><span>🟪 REM</span><span>⬜ Unspecified</span></p>{''.join(bar_rows) or '<p>수면 기록이 없습니다.</p>'}<small>막대의 전체 길이는 12시간을 기준으로 합니다. Awake / InBed는 수면 시간에 포함되지 않습니다.</small></div><div class="card"><h2>최근 100회 요약</h2><div class="tablewrap"><table><thead><tr>{''.join('<th>'+escape(k)+'</th>' for k in FIELDS)}</tr></thead><tbody>{tr}</tbody></table></div></div><p><small>날짜는 현지 시각 정오를 기준으로 묶습니다. 심박/HRV는 동일 날짜 구간의 단순 산술평균이며 수면 중 수치만 따로 추출한 것은 아닙니다. 데이터 누락은 0이 아닌 빈 값으로 표시합니다.</small></p></body></html>'''
    output.write_text(page, encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description='Apple Health ZIP/XML 수면 분석')
    parser.add_argument('input', type=Path,help='apple_health_export.zip 또는 export.xml')
    parser.add_argument('--out',type=Path,default=Path('output'))
    args=parser.parse_args()
    if not args.input.is_file(): parser.error(f'파일을 찾을 수 없습니다: {args.input}')
    try: rows,counts=extract(args.input)
    except (ET.ParseError,ValueError,zipfile.BadZipFile,OSError) as e: parser.error(f'파일 처리 실패: {e}')
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'sleep_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    render_report(rows,args.out/'report.html')
    print(f'완료: {len(rows)}일 수면 기록 | 수면 레코드 {counts["sleep"]:,}개 | 심박 {counts["hr"]:,}개 | HRV {counts["hrv"]:,}개')
    print('CSV:',(args.out/'sleep_summary.csv').resolve())
    print('HTML:',(args.out/'report.html').resolve())
    if not rows: print('수면 기록을 찾지 못했습니다. Apple Health에 수면 단계가 기록되어 있는지 확인하세요.')

if __name__=='__main__': main()
