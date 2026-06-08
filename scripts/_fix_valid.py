"""valid.json을 label별 균등(6개×5=30) 으로 재구성하는 임시 스크립트."""
import json, random, shutil
from pathlib import Path
from collections import Counter

random.seed(42)
BASE = Path(__file__).parent.parent

train_path = BASE / "data" / "train.json"
valid_path = BASE / "data" / "valid.json"
backup_path = BASE / "data" / "valid_backup.json"

train = json.load(open(train_path, encoding="utf-8"))
valid = json.load(open(valid_path, encoding="utf-8"))

print("Before train:", dict(sorted(Counter(d["label"] for d in train).items())))
print("Before valid:", dict(sorted(Counter(d["label"] for d in valid).items())))

# label 3, 4가 4개 → 6개로 맞추기 위해 train에서 부족분 이동
# 전략: train에서 label 3,4 각 2개를 valid로 이동 (train 균등성 유지하려면 label 3,4를 30→28)
# 대신 label 0,1,2는 valid 6개 유지 → 전체 valid 30개 균등

by_label_train = {i: [d for d in train if d["label"] == i] for i in range(5)}
by_label_valid = {i: [d for d in valid if d["label"] == i] for i in range(5)}

TARGET_VALID = 6  # 각 label당 valid 목표 개수

new_valid = []
new_train = []

for label in range(5):
    tv = by_label_train[label]
    vv = by_label_valid[label]
    have = len(vv)
    need = TARGET_VALID - have   # 부족분

    if need > 0:
        # train에서 need개 가져오기 (앞에서 가져오면 데이터 순서 의존 → 뒤에서)
        moved = tv[-need:]
        remaining = tv[:-need]
        vv = vv + moved
        tv = remaining
    new_valid.extend(vv[:TARGET_VALID])
    new_train.extend(tv)

print("After  train:", dict(sorted(Counter(d["label"] for d in new_train).items())))
print("After  valid:", dict(sorted(Counter(d["label"] for d in new_valid).items())))

# 백업 후 저장
shutil.copy(valid_path, backup_path)
json.dump(new_valid, open(valid_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(new_train, open(train_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"\nOK: valid.json {len(new_valid)}, train.json {len(new_train)}")
print("backup: data/valid_backup.json")
