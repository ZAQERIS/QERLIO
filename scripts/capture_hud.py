import os
import sys
from playwright.sync_api import sync_playwright

def capture_dashboard_screenshot():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    html_file = os.path.join(base_dir, "server", "local_api", "dashboard.html")
    
    # Screenshots can expose host details, so they are written outside the
    # repository tree (override with QERLIO_HUD_CAPTURE_DIR). Never commit them.
    output_dir = os.environ.get(
        "QERLIO_HUD_CAPTURE_DIR",
        os.path.join(os.path.expanduser("~"), ".qerlio", "hud_captures"),
    )
    try:
        inside_repo = os.path.commonpath([os.path.abspath(output_dir), base_dir]) == base_dir
    except ValueError:  # different drives on Windows
        inside_repo = False
    if inside_repo:
        raise SystemExit("Refusing to write HUD screenshots inside the repository tree.")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qerlio_hud_dashboard.png")

    file_url = f"file:///{html_file.replace(os.sep, '/')}"
    print(f"Loading: {file_url}")

    with sync_playwright() as p:
        browser = None
        # Try default chromium or system installed chrome/msedge
        for channel in [None, "msedge", "chrome"]:
            try:
                kwargs = {"channel": channel} if channel else {}
                browser = p.chromium.launch(**kwargs)
                print(f"Launched browser with channel: {channel or 'default'}")
                break
            except Exception as e:
                print(f"Could not launch with channel {channel}: {e}")

        if not browser:
            raise RuntimeError("No suitable browser could be launched by Playwright.")

        page = browser.new_page(viewport={"width": 1440, "height": 920}, device_scale_factor=2)
        page.goto(file_url)
        page.wait_for_timeout(1000)  # Wait for JS to populate tables

        # Populate an explicit unloaded preview rather than claiming a live model reply
        page.evaluate("""
            document.getElementById('chat-output').textContent = 
                'User: Ping gateway stub.\\n\\n' +
                'Assistant: [Simulated UI preview: no model is currently loaded into the engine.]';
        """)

        page.screenshot(path=output_path, full_page=True)
        browser.close()

    print(f"Screenshot successfully saved to: {output_path}")

if __name__ == "__main__":
    capture_dashboard_screenshot()
