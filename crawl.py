from selenium import webdriver
import csv
from datetime import datetime, timedelta
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import configparser
import os
from selenium.webdriver.common.action_chains import ActionChains

# website = "https://finance.yahoo.com/quote/BTC-USD"

class CrawlData(webdriver.Chrome):
    def __init__(self, stock, teardown = False):
        super(CrawlData, self).__init__()

        def _init_configuration():
            config = configparser.ConfigParser()
            config.read('config.ini')

            return config
        
        def _collect_data(past_date):
            current_date = int(self.current_time.timestamp())
            past_date = int(past_date.timestamp())

            self.website = f"https://finance.yahoo.com/quote/{self.stock}/history?period1={past_date}&period2={current_date}"

        def _read_datetime(date):
            formatted_date = datetime.strptime(date, "%b %d, %Y")
            return formatted_date

        self.teardown = teardown
        self.config = _init_configuration() 
        self.stock = self.config.get('Stocks', stock)
        self.past_time = _read_datetime(self.config.get('Date', f'past_date_{self.stock}'))
        self.current_time = datetime.now()
        # self.current_time = datetime(2023, 1, 1, 0, 0)

        _collect_data(self.past_time)
        self.waiter = WebDriverWait(self, 15)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.teardown:
            self.quit()

    def land_first_page(self):
        self.get(self.website)
        # print(self.website)
        time.sleep(2)

    def crawl_data(self):
        
        def _check_file(path_file):            
            if os.path.exists(path_file):
                return True
            return False


        initial_file = os.path.join(self.config.get('Data', 'path_file'), 
                                     f"{self.stock}.csv")

        table = WebDriverWait(self, 10).until(
            EC.presence_of_element_located(
                (By.TAG_NAME, 'table')
            )
        )   
    
        self.config.set('Date', f'past_date_{self.stock}', (self.current_time + timedelta(days = 1)).strftime("%b %d, %Y"))

        with open('config.ini', mode = 'w') as config_file:
            self.config.write(config_file)

      
        # if not _check_file(initial_file):
        header_cells = table.find_elements(By.TAG_NAME, 'th')
    
        k = 0

        if _check_file(initial_file):
            initial_file = os.path.join(self.config.get('Data', 'path_file'), 
                                     f"{self.stock}_BONUS.csv")
        
        with open(initial_file, mode = 'w+', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            header_cell = [cell.text for cell in header_cells]
            csv_writer.writerow(header_cell)
            
            repeated_date = self.current_time
    
            while True:
                
                rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
                if rows:
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if (datetime.strptime(cells[0].text, "%b %d, %Y") < repeated_date):
                            body_cell = [cell.text for cell in cells]
                            csv_writer.writerow(body_cell)
                        
                    last_date = datetime.strptime(rows[-1].find_elements(By.TAG_NAME, "td")[0].text, "%b %d, %Y")
                    
                    if last_date <= self.past_time or last_date == repeated_date:
                        break
                    else:
                        repeated_date = last_date
                
                    while True:
                        action = ActionChains(self)
                        action.send_keys(Keys.END).perform()
                        time.sleep(0.1)
                        k += 1

                        self.waiter.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody tr")))
                        rows_scroll = table.find_elements(By.CSS_SELECTOR, "tbody tr")
                        date_lol = datetime.strptime(rows_scroll[-1].find_elements(By.TAG_NAME, "td")[0].text, "%b %d, %Y")

                        print(date_lol)
                        # print(last_date)
                        # print()
                        if (date_lol <= last_date):
                            break
                else:
                    break
        print(k)                 

                