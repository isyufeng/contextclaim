import json
import random
from collections import Counter

random.seed(42)

INPUT_FILE = "data/CIKM/paper_dev.jsonl"
OUTPUT_FILE = "data/CIKM/fever_sampled_1000.jsonl"
SAMPLE_SIZE = 1000

# 1. Load and filter out NOT ENOUGH INFO
claims_by_label = {"SUPPORTS": [], "REFUTES": []}

with open(INPUT_FILE, "r") as f:
    for line in f:
        item = json.loads(line)
        label = item["label"]
        if label in claims_by_label:
            claims_by_label[label].append(item)

# 2. Print original distribution
print("=== Original Distribution (excluding NOT ENOUGH INFO) ===")
total = sum(len(v) for v in claims_by_label.values())
for label, items in claims_by_label.items():
    print(f"  {label}: {len(items)} ({len(items)/total*100:.1f}%)")
print(f"  Total: {total}")

# 3. Stratified sampling - preserve original label ratio
sample_counts = {}
for label, items in claims_by_label.items():
    sample_counts[label] = round(SAMPLE_SIZE * len(items) / total)

# Adjust rounding to ensure exactly SAMPLE_SIZE
diff = SAMPLE_SIZE - sum(sample_counts.values())
if diff != 0:
    largest_label = max(sample_counts, key=sample_counts.get)
    sample_counts[largest_label] += diff

# 4. Sample
sampled = []
for label, count in sample_counts.items():
    sampled.extend(random.sample(claims_by_label[label], count))

random.shuffle(sampled)

# 5. Save
with open(OUTPUT_FILE, "w") as f:
    for item in sampled:
        f.write(json.dumps(item) + "\n")

# 6. Print sampled distribution
print(f"\n=== Sampled Distribution ({SAMPLE_SIZE} total) ===")
sampled_counts = Counter(item["label"] for item in sampled)
for label, count in sorted(sampled_counts.items()):
    print(f"  {label}: {count} ({count/SAMPLE_SIZE*100:.1f}%)")
print(f"\nSaved to {OUTPUT_FILE}")