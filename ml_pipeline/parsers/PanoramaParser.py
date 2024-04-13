import logging

import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta, datetime
import pandas as pd
from fake_useragent import UserAgent

class PanoramaParser:
    def get_links_from_feed(self, date):
        url = f'https://panorama.pub/news/{date}'
        response = requests.get(url, headers={'User-Agent': UserAgent().chrome})
        tree = BeautifulSoup(response.content, 'html.parser')
        books_widgets = tree.find_all('a', {'class':'flex flex-col rounded-md hover:text-secondary hover:bg-accent/[.1] mb-2'})
        return [f'https://panorama.pub{x.get("href")}' for x in books_widgets]
    
    def get_article_content(self, url):
        response = requests.get(url)
        tree = BeautifulSoup(response.content, 'html.parser')
        body = tree.find('div', {'itemprop':'articleBody'}).find_all('p')
        return ' '.join(body[i].text for i in range(len(body)))
    
    def get_articles(self, dateFrom, dateTo, n_articles):
        dateFrom = datetime.strptime(dateFrom, '%Y-%m-%d')
        dateTo = datetime.strptime(dateTo, '%Y-%m-%d')
        out = pd.DataFrame(columns = ['text'])
        counter = 0
        while dateFrom <= dateTo:
            logging.info(f'Panorama: parsing articles dated {dateFrom}')
            links = self.get_links_from_feed(dateFrom.strftime('%d-%m-%Y'))
            for link in links:
                content = self.get_article_content(link)
                if content:
                    out.loc[len(out)] = content
            dateFrom += timedelta(days = 1)
            counter += 1
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        out = out.sample(n=min(n_articles, len(out)), axis=0)
        out.to_csv(f'data/panorama_{date_time_string}.csv', index=False)
        return out






