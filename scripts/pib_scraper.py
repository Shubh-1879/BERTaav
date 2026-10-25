import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
from bs4 import BeautifulSoup
import json

async def get_pib_content(url):
    async with async_playwright() as p:
        # Launch with arguments to mask automation flags
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        
        page = await context.new_page()
        # Navigate and wait for content
        await page.goto(url, wait_until="networkidle")
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Fallback to finding by ID or generic paragraph extraction
        content_container = soup.find(id='divContent') or soup.find('div', class_='PRDetail')
        
        if content_container:
            text_content = content_container.get_text(separator=' ', strip=True)
        else:
            # Fallback: Get all paragraphs on the page if container fails
            paragraphs = soup.find_all('p')
            text_content = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
            
        await browser.close()
        return text_content
# Example call
if __name__ == "__main__":
    url = "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2240912&reg=48&lang=2"
    print(asyncio.run(harvest_pib_article(url)))