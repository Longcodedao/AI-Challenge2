import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def fetch_stock_data(stock_symbol):
    url = f"https://finance.yahoo.com/quote/{stock_symbol}"
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)

    # Create a Chrome webdriver
    service = ChromeService()
    driver = webdriver.Chrome(service=service, options = chrome_options)

    # Fetch the Yahoo Finance page
    driver.get(url)    
    wait = WebDriverWait(driver, 10)

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
    
    return {"Price" : price, "Percentage": percentage}


st.set_page_config(
    page_title = "GOOG", page_icon = "📈"
)

@st.cache_data
def fetch_historical_data(dataset ,period= 1):
    
    end_date = datetime.now()                              
    start_date = end_date - relativedelta(months = period)
    filtered_data = dataset[(dataset.index >= start_date) & 
                                    (dataset.index <= end_date)]
    return filtered_data

infos = fetch_stock_data('GOOG')
st.markdown('## Alphabet Inc. (GOOG)')
st.write('NasdaqGS - NasdaqGS Real Time Price. Currency in USD')
# st.markdown(f"### {infos['Price']}")

color_percent = ''
if infos['Percentage'][0] == '+':
    color_percent = 'green'
else:
    color_percent = 'red'

st.markdown(
    f"""
    <div style="display: flex; justify-content: flex-start; flex-direction: row;  align-items: center;">
        <h1 style="font-weight:bold">{infos['Price']}</h1>
        <h2 style='color:{color_percent}; font-weight:bold'>{infos['Percentage']}%</h2>
    </div>
    """,
    unsafe_allow_html=True
)
# Periods = 1
periods = 3
st.write('At close: 04:00PM EST')
data_total = pd.read_csv(f'../data/GOOG.csv',
                                  index_col = ['Date'], parse_dates = True)
historical_data = fetch_historical_data(data_total, periods)


st.subheader(f"Historical Data for {periods} month")
st.dataframe(historical_data)


st.subheader(f"Ploting the Close Price")
st.line_chart(data_total['Close'])

if st.button('Plot Prediction'):
    predict = pd.read_csv('../Predict_30_GOOG.csv',
                          index_col = ['Date'],
                          parse_dates = True)
    st.subheader("Prediction of the Graph")
    st.line_chart(predict, color = ("#0000FF", "#8B0000"))