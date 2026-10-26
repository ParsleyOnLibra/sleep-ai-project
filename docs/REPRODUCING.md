# 실행 및 검증 방법

결과를 보는 데 실행은 필요하지 않습니다. README의 그래프와 `results/`에 실제 실행 결과를 보존했습니다.

## 환경 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eeg.txt
```

Windows에서는 가상환경 활성화 명령을 `.venv\Scripts\activate`로 바꿉니다. 의존성은 범위 지정이며 완전히 고정된 환경이 아닙니다. 기존 실행 환경은 `results/classification/run_metadata.json`을 참고하세요.

## 기존 결과로 그래프 재생성

```bash
python scripts/build_figures.py
```

추가 EDF 다운로드나 모델 학습 없이 보존된 JSON/CSV를 시각화합니다.

## 저장된 예측 재평가

```bash
python eeg/review_predictions.py --input results/classification --out output/review_portfolio
```

기존 validation에서 고른 임계값을 유지합니다. 재학습하지 않으며 수정된 Wake→N1 경계 정의와 2/5/10분 지속 수면 탐지를 재평가합니다. 출력 폴더는 새 이름을 사용하세요.

## 테스트

```bash
python -m unittest discover -s tests -v
```

합성 자료로 분할·문맥·전환 경계·다운로드 검사·감사 지표·통합 흐름을 검증합니다. 실제 사람의 수면 분류 정확도를 입증하는 테스트가 아닙니다.

## 선택적 10명 재학습

```bash
python eeg/eeg_train.py --learning-curve --out output/experiment_new
```

최초 실행은 공개 EDF를 내려받으므로 네트워크와 저장 공간이 필요합니다. 이미 자료가 있다면 `eeg_data/physionet-sleep-data/` 아래에 두고 `--offline`을 추가하세요. 첫 번째 밤 10명의 PSG·Hypnogram 쌍이 필요하며, `eeg/age_records.csv`의 체크섬으로 확인합니다.

## 원본 신호 품질 감사

```bash
python eeg/audit_quality.py --data-dir eeg_data --out output/quality_audit_new
```

학습 코드와 독립적인 진단 도구입니다. 전체 PSG의 native sampling rate를 유지하며, Hypnogram 라벨 필터를 적용하지 않습니다. 원본 EDF는 저장소에 포함하지 않았습니다.

## Apple Health 추출 코드

```bash
python analyze.py --help
```

개인 export는 로컬에서만 사용하세요. EEG 분류 모델과 연결하는 기능은 아직 없습니다. 기존 파서는 날짜별 HR/HRV를 요약하므로 수면 중 생리 지표로 간주하면 안 됩니다.
