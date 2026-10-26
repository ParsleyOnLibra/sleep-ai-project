# 데이터 출처 및 고지

## Sleep-EDF Expanded

본 포트폴리오의 공개 EEG 평가 지표, 파형 그림, 예측 및 감사 결과는 **PhysioNet Sleep-EDF Database Expanded v1.0.0**에서 파생되었습니다.

- 데이터셋: https://physionet.org/content/sleep-edfx/1.0.0/
- 이용 조건: Open Data Commons Attribution License v1.0 — https://physionet.org/content/sleep-edfx/view-license/1.0.0/
- 기록: sleep-cassette 참가자 0–9의 첫 번째 밤. 개별 파일 및 SHA-256은 results/quality/quality_audit_summary.json 참조.
- 원본 수집 기관/기여자에 대한 표기 및 권리는 위 데이터셋 페이지를 따릅니다. 본 프로젝트 작성자가 원본 수면 데이터를 수집한 것은 아닙니다.

원본 EDF는 재배포하지 않습니다. 결과와 파형은 데이터셋에서 계산·시각화한 파생 산출물이며 원본 라벨이나 데이터를 새로 수집한 결과가 아닙니다.

## 코드 및 외부 자료

`eeg/age_records.csv`는 기존 프로젝트에 포함된 Sleep-EDF 기록 manifest입니다. 함께 보존한 `docs/MNE_LICENSE.txt`의 고지를 유지합니다. Python 및 외부 패키지는 각각의 라이선스를 따릅니다. 이 정리 과정에서 프로젝트 자체에 새 오픈소스 라이선스를 임의 부여하지 않았습니다.

개인 Apple Health export, 개인 수면 요약표, 심박·HRV 원본 및 개인 보고서는 포함하지 않습니다.
