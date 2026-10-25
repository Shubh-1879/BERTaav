import json
import collections

def analyze_full_corpus(file_path="Data/corpus.jsonl"):
    total_entries = 0
    total_hindi_chars = 0
    total_non_hindi_chars = 0
    broken_entries = []
    
    # Unicode range for Devanagari
    # U+0900 to U+097F
    
    print("Scanning corpus...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                text = entry.get("text", "")
                total_entries += 1
                
                if not text:
                    broken_entries.append(entry.get("filename", "unknown"))
                    continue
                
                # Count Devanagari vs everything else
                for char in text:
                    if '\u0900' <= char <= '\u097f':
                        total_hindi_chars += 1
                    else:
                        total_non_hindi_chars += 1
                        
            except json.JSONDecodeError:
                continue

    print(f"\n--- Corpus Analysis Report ---")
    print(f"Total files processed: {total_entries}")
    print(f"Files with zero length: {len(broken_entries)}")
    print(f"Total Hindi characters found: {total_hindi_chars}")
    print(f"Total non-Hindi characters found: {total_non_hindi_chars}")
    
    if total_entries > 0:
        hindi_ratio = total_hindi_chars / (total_hindi_chars + total_non_hindi_chars + 1)
        print(f"Hindi content ratio: {hindi_ratio:.2%}")
    
    if broken_entries:
        print(f"\nFiles needing urgent OCR review (first 5): {broken_entries[:5]}")

if __name__ == "__main__":
    analyze_full_corpus()