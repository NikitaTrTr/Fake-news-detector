import requests
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.common.by import By
import time


class LapshaParser:
    def scroll_pages(self):
        driver = webdriver.Firefox()
        ref = 'https://lapsha.media/novoe/'
        driver.get(ref)
        time.sleep(30)
        try:
            button = driver.find_element(By.CLASS_NAME, 'closebanner')
            button.click()
            time.sleep(0.5)
        except:
            pass
        try:
            button = driver.find_element(By.CLASS_NAME, 'cn-buttons-container')
            button.click()
            time.sleep(0.5)
        except:
            pass
        while True:
            time.sleep(0.2)
            try:
                button = driver.find_element(By.CLASS_NAME, 'lenta-loader')
                button.click()
            except:
                break
        return driver

    def get_article_text(self, url):
        response = requests.get(url, headers={'User-Agent': UserAgent().chrome})
        tree = BeautifulSoup(response.content, 'html.parser')
        title = tree.find('h1').text
        body = tree.find('div', {'class': 'bialty-container'})
        text = [title]
        for p in body:
            if 'Как на самом деле' in p.text:
                break
            text.append(p.text)
        return ' '.join(text)

    def get_articles(self, n_artircles=float('inf')):
        driver = self.scroll_pages()
        tree = BeautifulSoup(driver.page_source, 'html.parser')
        lenta = tree.find('div', {'class': 'lenta'})
        links_to_articles = []
        for post in lenta:
            if post.find('a', {'class': 'postblock-title__tag'}).text == 'Фейки':
                link = post.find('meta').get('content')
                links_to_articles.append(link)
            if len(links_to_articles) == n_artircles:
                break
        df = pd.DataFrame(columns=['text'])
        for link in links_to_articles:
            print(f'Parsing {link}')
            df.loc[len(df)] = self.get_article_text(link)
        return df

parser = LapshaParser()
df = parser.get_articles()
df.to_csv('../../Data/panorama.csv', index = False)