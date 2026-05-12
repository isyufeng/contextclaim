from pathlib import Path

current_dir = Path('./data/keywords')
for file in current_dir.iterdir():
    if 'CT22_keywords_claim_train' in file:
        print(file)