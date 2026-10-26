"""Sleep-EDF SC manifest, verified cache and interruptible atomic downloads."""
import csv
import hashlib
import time
import urllib.request
from pathlib import Path

BASE = 'https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest():
    with Path(__file__).with_name('age_records.csv').open() as f:
        return list(csv.DictReader(f))


def acquire(row, root, offline=False, retries=3, timeout=60):
    target = Path(root) / 'physionet-sleep-data' / row['fname']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target, 'sha1') == row['sha']:
            return target
        raise ValueError(f'캐시 체크섬 불일치: {target}. 원본을 보관한 뒤 해당 파일을 다른 폴더로 옮기고 재실행하세요.')
    if offline:
        raise FileNotFoundError(f'오프라인 캐시 없음: {target}')
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(retries):
        try:
            print(f"다운로드 {row['fname']} ({attempt + 1}/{retries})", flush=True)
            with urllib.request.urlopen(BASE + row['fname'], timeout=timeout) as response, partial.open('wb') as f:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    f.write(block)
            if digest(partial, 'sha1') != row['sha']:
                raise OSError('다운로드 체크섬 불일치')
            partial.replace(target)
            return target
        except (OSError, TimeoutError) as error:
            if attempt == retries - 1:
                raise RuntimeError(f'{target.name} 다운로드 실패. 완료 파일은 유지됩니다. 재실행 또는 수동 다운로드: {BASE + row["fname"]}') from error
            time.sleep(min(2 ** attempt, 8))
    raise ValueError('retries must be positive')


def get_nights(subjects, nights, root, **kwargs):
    rows = manifest()
    result = []
    for subject in subjects:
        for night in nights:
            pair = [r for r in rows if int(r['subject']) == subject and int(r['night']) == night]
            by_type = {r['record type']: r for r in pair}
            if set(by_type) != {'PSG', 'Hypnogram'}:
                raise ValueError(f'공식 기록 없음: subject={subject}, night={night}. 참가자/박 목록을 명시적으로 수정하세요.')
            paths = [acquire(by_type[k], root, **kwargs) for k in ['PSG', 'Hypnogram']]
            result.append((subject, night, *paths))
    # Identical PSG bytes under different subjects must never cross partitions.
    hashes = [digest(p[2]) for p in result]
    if len(hashes) != len(set(hashes)):
        raise ValueError('중복 PSG 발견; 중복 데이터 분리 누출 위험')
    return result
