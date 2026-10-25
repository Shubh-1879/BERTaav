import json
import re
import os

# --- Configuration ---
INPUT_FILE = "Data/lok_sabha1.jsonl"       # Your raw OCR output
OUTPUT_FILE = "Data/lok_sabha1_final.jsonl" # Cleaned & Structured
CATEGORY = "Lok Sabha Debates"         # You can change this for other datasets

def clean_text(text):
    """Clean OCR noise and structure."""
    # Replace newlines and multiple spaces with a single space
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common OCR artifacts (customize these patterns as needed)
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    
    return text.strip()

def get_lang(text):
    """Deterministic script-based detection."""
    hindi_chars = len(re.findall(r'[\u0900-\u097f]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total = hindi_chars + latin_chars
    if total < 10: return "other" # Too short to detect
    return "hi" if (hindi_chars / total) > 0.15 else "en"

def parse_date(filename):
    match = re.search(r'\d{2}-\d{2}-\d{4}', filename)
    return match.group(0) if match else "unknown"

def run_pipeline():
    print(f"Starting final processing for: {CATEGORY}...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            try:
                data = json.loads(line)
                raw_text = data.get("text", "")
                
                # 1. Clean
                cleaned_text = clean_text(raw_text)
                
                # 2. Filter (Minimum 30 characters of meaningful content)
                if len(cleaned_text) < 30:
                    continue
                
                # 3. Enrich
                final_entry = {
                    "text": cleaned_text,
                    "category": CATEGORY,
                    "lang": get_lang(cleaned_text),
                    "date": parse_date(data.get("source_image", "")),
                    "source_file": data.get("source_image", ""),
                    "token_count": len(cleaned_text.split())
                }
                
                f_out.write(json.dumps(final_entry, ensure_ascii=False) + "\n")
                
            except json.JSONDecodeError:
                continue
                
    print(f"Success! Dataset saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()