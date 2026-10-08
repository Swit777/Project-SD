"""Optional visual smoke test; requires locally installed Selenium and Chrome."""
import json
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/game_concept/ui_checks"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,1100")
    options.add_argument("--disable-dev-shm-usage")
    results = {}
    with webdriver.Chrome(options=options) as browser:
        browser.set_page_load_timeout(60)
        browser.get("http://127.0.0.1:8501")
        wait = WebDriverWait(browser, 60)
        wait.until(lambda driver: len(driver.find_elements(By.CSS_SELECTOR, '[data-testid="stPlotlyChart"]')) > 0)
        wait.until(lambda driver: any(image.get_attribute("naturalWidth") != "0" for image in driver.find_elements(By.CSS_SELECTOR, '[data-testid="stImage"] img')))
        for mode, width, height in [("desktop", 1440, 1100), ("wide_desktop", 1920, 1080)]:
            browser.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
                "width": width, "height": height, "deviceScaleFactor": 1, "mobile": mode == "mobile"})
            wait.until(lambda driver: driver.execute_script("return window.innerWidth") == width)
            browser.save_screenshot(str(OUTPUT / f"{mode}.png"))
            probe = browser.execute_script("""
                return {
                    viewport: [window.innerWidth, window.innerHeight],
                    documentWidth: document.documentElement.scrollWidth,
                    errors: document.querySelectorAll('[data-testid="stException"]').length,
                    images: [...document.querySelectorAll('[data-testid="stImage"] img')].map(i => ({loaded: i.complete && i.naturalWidth > 0})),
                    chartCount: document.querySelectorAll('[data-testid="stPlotlyChart"]').length,
                    headings: [...document.querySelectorAll('h1,h2,h3')].filter(e => e.getClientRects().length).map(e => ({text: e.innerText, width: e.clientWidth, scrollWidth: e.scrollWidth}))
                };
            """)
            assert probe["errors"] == 0, probe
            assert probe["documentWidth"] <= width + 1, probe
            assert all(h["scrollWidth"] <= h["width"] + 1 for h in probe["headings"]), probe
            results[mode] = probe
        browser.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
            "width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
        def open_tab(label):
            matches = [element for element in browser.find_elements(By.CSS_SELECTOR, '[role="tab"]') if element.text == label]
            assert len(matches) == 1, f"Tab not unique: {label}"
            browser.execute_script("arguments[0].scrollIntoView({block: 'center'});", matches[0])
            matches[0].click()
        def expand(label):
            matches = [element for element in browser.find_elements(By.CSS_SELECTOR, "summary") if element.is_displayed() and label in element.text]
            assert len(matches) == 1, f"Section not unique: {label}"
            matches[0].click()
        open_tab("Данные Steam")
        expand("Ручная проверка извлечения")
        wait.until(lambda driver: any(button.is_displayed() and "Сохранить разметку" in button.text for button in driver.find_elements(By.TAG_NAME, "button")))
        game_picker = [element for element in browser.find_elements(By.CSS_SELECTOR, '[data-testid="stSelectbox"]') if element.is_displayed() and "Проверяемая игра" in element.text]
        assert len(game_picker) == 1
        browser.execute_script("arguments[0].scrollIntoView({block: 'center'});", game_picker[0])
        browser.save_screenshot(str(OUTPUT / "annotation_desktop.png"))
        assert not browser.find_elements(By.CSS_SELECTOR, '[data-testid="stException"]')
        open_tab("Исследование")
        expand("Проверка вероятностей")
        picker = [element for element in browser.find_elements(By.CSS_SELECTOR, '[data-testid="stSelectbox"]') if element.is_displayed() and "Граница аудитории" in element.text]
        assert len(picker) == 1
        browser.execute_script("arguments[0].scrollIntoView({block: 'center'});", picker[0])
        browser.save_screenshot(str(OUTPUT / "calibration_desktop.png"))
        assert not browser.find_elements(By.CSS_SELECTOR, '[data-testid="stException"]')
        results["annotation_and_calibration"] = {"errors": 0, "screenshots": ["annotation_desktop.png", "calibration_desktop.png"]}
        (OUTPUT / "checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
