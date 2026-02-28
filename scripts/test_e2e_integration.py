import asyncio
import subprocess
import time
import os
import sys
from pyppeteer import launch

CHROME_PATH = os.path.expanduser("~/chromium/src/out/Default/chrome")
TEST_PAGE = "file://" + os.path.abspath("scripts/test_page.html")
MOCK_SERVER = os.path.abspath("scripts/mock_e2e_server.py")

async def run_test():
    print("Step 1: Launching Mock E2E Server...")
    server_proc = subprocess.Popen([sys.executable, MOCK_SERVER])
    time.sleep(3) # Wait for server to bind

    try:
        print(f"Step 2: Launching Custom Chrome: {CHROME_PATH}")
        browser = await launch(
            executablePath=CHROME_PATH,
            headless=True,
            args=[
                '--no-sandbox',
                '--force-renderer-accessibility',
                '--enable-logging=stderr',
                '--v=1'
            ]
        )
        
        page = await browser.newPage()
        
        print(f"Step 3: Navigating to {TEST_PAGE}...")
        await page.goto(TEST_PAGE)
        
        print("Step 4: Waiting for VLM Actuator to fire (max 15s)...")
        success = False
        for i in range(15):
            bg_color = await page.evaluate("document.body.style.backgroundColor")
            print(f"Current Background Color: {bg_color}")
            if bg_color == "green":
                success = True
                break
            await asyncio.sleep(1)
        
        if success:
            print("SUCCESS: VLM Actuator physical event injection verified!")
        else:
            print("FAILURE: VLM Actuator did not fire correctly.")
            sys.exit(1)

        await browser.close()
    finally:
        print("Step 6: Cleaning up subprocesses...")
        server_proc.terminate()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run_test())
