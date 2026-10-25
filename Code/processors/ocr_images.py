import pytesseract
from PIL import Image
import os
import json

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image(image_path):
    img = Image.open(image_path)
    # Using 'hin+eng' for multilingual support
    return pytesseract.image_to_string(img, lang='hin+eng', config='--psm 3')

def process_all_nested_images(root_dir="Data/images", output_file="Data/corpus.jsonl"):
    print(f"Walking through directory: {root_dir}")
    
    with open(output_file, "a", encoding="utf-8") as f_out:
        # os.walk is the key: it visits root AND all subfolders
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dirpath, filename)
                    print(f"Processing: {filename} from {dirpath}")
                    
                    try:
                        text = ocr_image(img_path)
                        
                        entry = {
                            "text": text,
                            "source_image": filename,
                            "folder": os.path.basename(dirpath),
                            "lang": "mixed"
                        }
                        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"Failed {filename}: {e}")

if __name__ == "__main__":
    process_all_nested_images()