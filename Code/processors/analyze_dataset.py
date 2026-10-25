import json
from collections import Counter

def analyze_dataset(file_path="Data/lok_sabha1_final.jsonl"):
    total_segments = 0
    lang_counts = Counter()
    dates = Counter()
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            total_segments += 1
            lang_counts[entry['lang']] += 1
            dates[entry['date']] += 1
            
    print(f"--- Dataset Quality Audit ---")
    print(f"Total Segments: {total_segments}")
    print(f"Language Distribution: {dict(lang_counts)}")
    print(f"Date Coverage: {len(dates)} unique dates processed")

if __name__ == "__main__":
    analyze_dataset()