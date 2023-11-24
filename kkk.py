from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def fetch_stock_data(stock_symbol):
    url = f"https://finance.yahoo.com/quote/{stock_symbol}"
    
    # Set up Chrome options
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)

    # Create a Chrome webdriver
    service = ChromeService()
    driver = webdriver.Chrome(service=service)

    # Fetch the Yahoo Finance page
    driver.get(url)
    
    wait = WebDriverWait(driver, 10)

    # texts = wait.until(
    #     EC.presence_of_element_located(
    #         (By.CSS_SELECTOR, "h1.D(ib).Fz(18px)")
    #     )
    # )
    # print(texts.text)
    price_element = wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "[data-test='qsp-price']")
        )
    )
    price = price_element.text

    percentage_increase = wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "[data-test='qsp-price-change']")
        )
    )
    percentage = percentage_increase.text
    
    print(price)
    print(percentage)
   
fetch_stock_data('GOOG')