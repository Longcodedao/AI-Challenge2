from crawl import CrawlData
from datetime import datetime
import time

# with CrawlData('GOOG') as web:
#     web.land_first_page()
#     web.crawl_data()

with CrawlData('GOOG') as web:
    web.land_first_page()
    web.crawl_data()