import requests
import json
import os
import time
import logging
import random
import browser_cookie3

# Configuration
MANIFEST_FILE = "Data/pdf_manifest.jsonl"
PDF_DIR = "Data/raw_pdfs"
META_FILE = "Data/meta/session_dates.json"
API_BASE = "https://sansad.in/api_ls/debate/text-of-debate"
LANDING_PAGE = "https://sansad.in/ls/debates/text-of-debates"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Referer": "https://sansad.in/ls/debates/text-of-debates",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BERTaavAPI")

def run_harvest():
    # 1. Attempt to load cookies while Chrome is CLOSED
    session = requests.Session()
    session.headers.update(HEADERS)
    
    try:
        logger.info("Attempting to load Chrome cookies (Ensure Chrome is CLOSED)...")
        # Use domain_name to target specifically
        cj = browser_cookie3.chrome(domain_name='sansad.in')
        session.cookies.update(cj)
        logger.info("Cookies loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load cookies: {e}. Check if Chrome is closed.")

    # 2. Prime the session
    session.get(LANDING_PAGE)

    with open(META_FILE, "r") as f:
        data = json.load(f)

    for ls_entry in data[::-1]:
        ls_no = ls_entry['loksabha']
        for session_entry in ls_entry['sessions'][::-1]:
            # Use string conversion if sessionNo is already a number
            roman_session = str(session_entry['sessionNo'])
            
            for date_str in session_entry['dates'][::-1]:
                safe_date = date_str.replace('/', '-')
                file_name = f"ls{ls_no}_s{roman_session}_{safe_date}.pdf"
                save_path = os.path.join(PDF_DIR, file_name)
                
                if os.path.exists(save_path): continue

                logger.info(f"Checking: LS {ls_no}, Session {roman_session}, Date {date_str}")
                
                params = {
                    "loksabha": ls_no,
                    "sessionNo": roman_session,
                    "debateDate": date_str,
                    "locale": "en"
                }
                
                try:
                    resp = session.get(API_BASE, params=params, timeout=20)
                    if resp.status_code == 200:
                        content = resp.json()
                        pdf_url = content.get("pdfUrl")
                        if pdf_url:
                            d_resp = session.get(pdf_url, stream=True, timeout=60)
                            with open(save_path, 'wb') as f:
                                for chunk in d_resp.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            logger.info(f"Downloaded: {file_name}")
                        else:
                            # If this is still empty, the server truly has no PDF for this date
                            logger.debug(f"Empty API response for {date_str}")
                    time.sleep(random.uniform(2.0, 4.0))
                except Exception as e:
                    logger.error(f"Error {date_str}: {e}")

if __name__ == "__main__":
    run_harvest()