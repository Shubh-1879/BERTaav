import os, json, glob, re
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUT_PATH = os.path.join(SCRIPT_DIR, "domain_vocab_frequencies.tsv")

# Devanagari block is \u0900-\u097F. Danda (।, U+0964) and double danda (॥, U+0965)
# are sentence punctuation but fall inside this range, so we explicitly carve them
# out — otherwise words get glued to a trailing danda (e.g. "है।" instead of "है").
# Matras (vowel signs, e.g. ा U+093E) are untouched — they're well outside 0964-0965.
TOKEN_RE = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")

def tokenize(text):
    return TOKEN_RE.findall(text)

# Pure grammar / function words — filtered because they're common in ALL Hindi,
# not specifically informative about domain.
STOPWORDS = {
    "है", "के", "की", "का", "में", "को", "से", "और", "यह", "इस", "पर",
    "एक", "हैं", "था", "थी", "थे", "कि", "जो", "वह", "गया", "गई", "गए",
    "कर", "किया", "करने", "लिए", "साथ", "भी", "ने", "तो", "या", "अपने",
    "हो", "होगा", "होगी", "नहीं", "सभी", "बाद", "आदि", "यदि", "जब",
}

# Genre-specific procedural/bureaucratic jargon from Lok Sabha Q&A format
# specifically — high-frequency in YOUR corpus because of document TYPE
# (parliamentary question-answer transcripts), not because they signal
# "political content" in the broader sense we want for filtering Sangraha.
# Excluded separately from STOPWORDS so it's easy to review/adjust this list
# on its own without touching core grammar filtering.
PROCEDURAL_JARGON = {
    "ब्यौरा", "तत्संबंधी", "अतारांकित", "कृपा", "बताने", "दिनांक",
    "तारांकित", "प्रश्न", "उत्तर", "सभा", "लोक", "करेंगे", "जाने",
    "विवरण", "अनुसार", "हेतु", "संख्या", "कुल",
}

EXCLUDE = STOPWORDS | PROCEDURAL_JARGON

def main():
    counter = Counter()
    doc_count = 0

    for fp in glob.glob(os.path.join(PROCESSED_DIR, "*.json")):
        if fp.endswith("quality_report.json"):
            continue
        with open(fp, encoding="utf-8") as f:
            r = json.load(f)
        text = r.get("text", "")
        tokens = [t for t in tokenize(text) if t not in EXCLUDE and len(t) > 2]
        counter.update(tokens)
        doc_count += 1
        if doc_count % 10000 == 0:
            print(f"processed {doc_count} docs...")

    print(f"Total docs: {doc_count}")
    print(f"Unique tokens: {len(counter)}")
    print("\nTop 50 terms by frequency (preview):")
    for term, freq in counter.most_common(50):
        print(f"{term}\t{freq}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for term, freq in counter.most_common(3000):
            f.write(f"{term}\t{freq}\n")
    print(f"\nSaved top 3000 terms to {OUT_PATH}")

if __name__ == "__main__":
    main()
