import requests as rq
import pandas as pd
from datetime import datetime, timedelta
from fake_useragent import UserAgent
import logging

class LentaParser:
    def _get_url(self, param_dict: dict) -> str:
        url = 'https://lenta.ru/search/v2/process?' \
              + 'size=700&' \
              + 'domain=1&' \
              + 'modified%2Cformat=yyyy-MM-dd&' \
              + 'modified%2Cfrom={}&'.format(param_dict['dateFrom']) \
              + 'modified%2Cto={}'.format(param_dict['dateTo'])
        return url

    def _get_search_table(self, param_dict: dict) -> pd.DataFrame:
        url = self._get_url(param_dict)
        r = rq.get(url, headers={'User-Agent': UserAgent().chrome})
        search_table = pd.DataFrame(r.json()['matches'])
        return search_table

    def get_articles(self,
                     param_dict,
                     n_articles,
                     time_step=10) -> pd.DataFrame:
        param_copy = param_dict.copy()
        time_step = timedelta(days=time_step)
        dateFrom = datetime.strptime(param_copy['dateFrom'], '%Y-%m-%d')
        dateTo = datetime.strptime(param_copy['dateTo'], '%Y-%m-%d')
        if dateFrom > dateTo:
            raise ValueError('dateFrom should be less than dateTo')
        out = pd.DataFrame()
        while dateFrom <= dateTo:
            param_copy['dateTo'] = (dateFrom + time_step).strftime('%Y-%m-%d')
            if dateFrom + time_step > dateTo:
                param_copy['dateTo'] = dateTo.strftime('%Y-%m-%d')
            logging.info('Lenta: parsing articles from ' \
                  + param_copy['dateFrom'] + ' to ' + param_copy['dateTo'])
            out = pd.concat([out, self._get_search_table(param_copy)], ignore_index=True)
            dateFrom += time_step + timedelta(days=1)
            param_copy['dateFrom'] = dateFrom.strftime('%Y-%m-%d')
        out = out.dropna(subset=['text'])
        out = out[out['text'] != '']
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        out = out.sample(n=min(n_articles, len(out)), axis=0)
        out.to_csv(f'data/lenta_{date_time_string}.csv', index=False)
        return out

