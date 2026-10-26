# 결과 출처와 범위

정리 날짜: 2026-10-26. 새로운 성능 개선 실험 없이 이전 산출물을 복원해 포트폴리오 형태로 구성했습니다.

| 내용 | 원본 산출물 | 이번 구성 |
|---|---|---|
| Apple Health 파서 | sleep_ai_research.zip | analyze.py만 포함, 개인 기록 제외 |
| EEG 학습·평가 코드 | sleep_ai_update_v041.zip | eeg/, tests/, manifest 복원 |
| 10명 분류 결과 | metrics.json, predictions.csv, learning_curve.json | results/classification/ |
| validation 모델 선택 | selection(1).json | selection.json으로 이름 정리 |
| 수정된 전환 평가 | transitions(1).json | transitions.json으로 이름 정리 |
| 지속 수면 평가 | sustained_sleep.json | 원 결과 보존 |
| 실행 환경 | run.json | 경로만 eeg_data로 치환한 run_metadata.json |
| 품질 감사 코드 | sleep_ai_quality_audit_v050.zip | eeg/audit_quality.py, 기존 테스트 |
| 감사 결과 | quality_audit_summary.json, quality_audit_channels.csv, quality_reasons.json | results/quality/ |
| 파형 검토 | participant6_wave_review.zip, waveform_review_35.zip | 원 PNG와 findings.json |

## 서로 다른 실행 시점

분류 지표는 v0.3 계열 기존 실행의 보존 결과입니다. 포함된 학습 코드 v0.4.1은 이후 전환 경계·품질 이유 로그·지속 탐지 등을 반영했습니다. 두 가지를 동일 실행의 코드/결과라고 주장하지 않습니다. `run_metadata.json`의 코드 해시는 이전 실행의 것으로 현 소스 해시와 다를 수 있습니다. status=started라는 원 값은 완료 증거가 아니며 그대로 보존했습니다.

v0.4 전환 수정은 기준 사건을 관측하기 전 경보가 매칭되던 경계 문제를 수정했습니다. 예전 6건 매칭 대신 수정된 4건 매칭만 대표 결과로 사용합니다. v0.5.0-audit는 모델 재학습이나 품질 필터 교체가 아니라 별도 진단 도구입니다.

## 해석 제한

- 사람별 분할을 했지만 평가 인원은 2명으로 작고 한 번의 분할입니다.
- full-window 평가에 Wake가 많습니다. Accuracy와 Macro F1, 단계별 지표를 함께 봐야 합니다.
- 학습곡선은 고정된 검증 2명에서 학습 인원을 늘린 비교입니다. 평가 인원이 늘어나서 정확도가 높아졌다는 뜻이 아닙니다.
- 품질 통과 4,275/5,492구간의 분류 지표이며, 제외 구간의 성능은 알 수 없습니다.
- 품질 감사는 전체 PSG 27,476구간, 학습은 라벨과 정렬한 27,464구간입니다.
- 감사에서 한계값 접촉이 있다고 즉시 자동 제외하거나 센서 고장으로 확정하지 않았습니다.
- 기존 평가 참가자 기록을 이후 오류 진단에 사용했습니다. 새 기준의 최종 평가에는 새 독립 자료가 필요합니다.
- 개인 수면 진단, EEG 없는 N1/N2 판별, 수면 유도 효과를 검증하지 않았습니다.

## 공개 준비 변경

모델 알고리즘은 변경하지 않았습니다. README, 실행·출처 문서, 결과 기반 그림 생성 스크립트, .gitignore를 추가했습니다. 개인 Apple Health 기록, 로컬 경로, EDF 원본, 모델 바이너리 및 캐시는 공개 패키지에서 제외했습니다. 그래프의 수치는 저장된 실험 결과에서 읽으며 임의 성능 데이터를 만들지 않습니다.
