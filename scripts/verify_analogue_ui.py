"""Desktop visual regression checks using the repository's Selenium/Chrome setup."""
import argparse
import json
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/game_concept/analogue_upgrade/ui"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8502")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,1100")
    results = {}
    with webdriver.Chrome(options=options) as browser:
        browser.set_page_load_timeout(60)
        browser.get(args.url)
        wait = WebDriverWait(browser, 60)
        wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, '[data-testid="stPlotlyChart"]')) >= 8)

        def capture(name):
            path = OUTPUT / f"{name}.png"
            browser.save_screenshot(str(path))
            with Image.open(path) as image:
                assert image.convert("RGB").entropy() > 1, "Blank screenshot"
            probe = browser.execute_script("""return {
                width: innerWidth, documentWidth: document.documentElement.scrollWidth,
                errors: document.querySelectorAll('[data-testid="stException"]').length,
                plots: [...document.querySelectorAll('[data-testid="stPlotlyChart"]')].filter(e=>e.getClientRects().length).length,
                headings: [...document.querySelectorAll('h1,h2,h3')].filter(e=>e.getClientRects().length).map(e=>({text:e.innerText, width:e.clientWidth, scroll:e.scrollWidth}))
            };""")
            assert probe["errors"] == 0, probe
            assert probe["documentWidth"] <= probe["width"] + 1, probe
            assert all(h["scroll"] <= h["width"] + 1 for h in probe["headings"]), probe
            results[name] = probe

        def scroll_heading(text):
            candidates = [e for e in browser.find_elements(By.CSS_SELECTOR, 'h1,h2,h3') if e.is_displayed() and e.text == text]
            assert len(candidates) == 1, text
            browser.execute_script("arguments[0].scrollIntoView({block:'start'});", candidates[0])

        for width in [1440, 1920]:
            browser.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": width, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
            scroll_heading("Прогноз игровой концепции")
            capture(f"forecast_{width}")
            scroll_heading("Сопоставимые игры")
            capture(f"analogues_{width}")
            landscapes = browser.find_elements(By.CSS_SELECTOR, '.st-key-analogue_landscape')
            assert len(landscapes) == 1
            browser.execute_script("arguments[0].scrollIntoView({block:'start'});", landscapes[0])
            browser.execute_script("document.querySelector('[data-testid=stMain]').scrollBy(0, -100);")
            capture(f"analogue_charts_{width}")
        picker = [e for e in browser.find_elements(By.CSS_SELECTOR, '[data-testid="stSelectbox"]') if e.is_displayed() and "Подробности игры" in e.text]
        assert len(picker) == 1
        browser.execute_script("arguments[0].scrollIntoView({block:'start'});", picker[0])
        wait.until(lambda d: any(i.is_displayed() and int(i.get_attribute("naturalWidth")) > 0 for i in d.find_elements(By.CSS_SELECTOR, '[data-testid="stImage"] img')))
        capture("game_detail")
        expanders = [e for e in browser.find_elements(By.CSS_SELECTOR, "summary") if e.is_displayed() and "Формат игры и ориентир" in e.text]
        assert len(expanders) == 1
        browser.execute_script("arguments[0].scrollIntoView({block:'center'});", expanders[0])
        expanders[0].click()
        search = wait.until(lambda d: next((e for e in d.find_elements(By.CSS_SELECTOR, '.st-key-analogue_anchor_query input') if e.is_displayed()), None))
        browser.execute_script("arguments[0].scrollIntoView({block:'center'});", search)
        search.send_keys("Stardew Valley", Keys.ENTER)
        picker = wait.until(lambda d: d.find_element(By.CSS_SELECTOR, '.st-key-analogue_anchor [role=combobox]'))
        picker.click()
        option = wait.until(lambda d: next((e for e in d.find_elements(By.CSS_SELECTOR, '[role=option]') if e.text.strip() == "Stardew Valley"), None))
        option.click()
        wait.until(lambda d: "Coral Island" in d.find_element(By.CSS_SELECTOR, '.st-key-analogue_detail').text)
        browser.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
        landscape = browser.find_element(By.CSS_SELECTOR, '.st-key-analogue_landscape')
        browser.execute_script("arguments[0].scrollIntoView({block:'start'}); document.querySelector('[data-testid=stMain]').scrollBy(0,-100);", landscape)
        capture("anchor_1440")
        for title in ["Данные Steam", "Исследование"]:
            tabs = [e for e in browser.find_elements(By.CSS_SELECTOR, '[role="tab"]') if e.text == title]
            assert len(tabs) == 1
            browser.execute_script("arguments[0].scrollIntoView({block:'center'});", tabs[0])
            tabs[0].click()
            heading = "Корпус игровых элементов" if title == "Данные Steam" else "Какие игры модель распознаёт хуже"
            wait.until(lambda d: any(e.is_displayed() and e.text == heading for e in d.find_elements(By.CSS_SELECTOR, 'h1,h2,h3')))
            scroll_heading(heading)
            capture("data" if title == "Данные Steam" else "model_errors")
        (OUTPUT / "checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
