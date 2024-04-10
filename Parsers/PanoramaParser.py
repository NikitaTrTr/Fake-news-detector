import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
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
    
    def get_articles(self, dateFrom, dateTo):
        articles = pd.DataFrame(columns = ['text'])
        counter = 0
        while dateFrom <= dateTo:
            print(f'Parsing articles dated {dateFrom}')
            links = self.get_links_from_feed(dateFrom.strftime('%d-%m-%Y'))
            for link in links:
                content = self.get_article_content(link)
                if content:
                    articles.loc[len(articles)] = content
            if counter % 10 == 0:
                articles.to_csv('panorama.csv', index = False)
            dateFrom += timedelta(days = 1)
            counter += 1
        return articles


parser = PanoramaParser()
dateFrom, dateTo = date(2023, 4, 8), date(2024, 4, 8)
df = parser.get_articles(dateFrom, dateTo)
df.to_csv('../Data/panorama.csv', index = False)





