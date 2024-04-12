import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem


class Preprocessor:
    def __init__(self):
        nltk.download('punkt')
        s = stopwords.words('russian')
        s.extend(['фото', 'новость', 'риа', 'лента', 'тасс', 'коммерсантъ'])
        self.stop_words = set(s)
        self.mystem = Mystem()

    def preprocess(self):
        lapsha = pd.read_csv('../../Data/lapsha.csv', sep='\t')
        panorama = pd.read_csv('../../Data/panorama.csv', sep='\t')
        lenta = pd.read_csv('../../Data/lenta.csv')

        lapsha['label'] = 1  # is fake
        panorama['label'] = 1  # is fake
        lenta['label'] = 0  # is not fake

        result = pd.concat([lapsha[['text', 'label']], panorama[['text', 'label']], lenta[['text', 'label']]])
        result = result.dropna(subset=['text'])
        result = result[result['text'] != '']

        result['text'] = result['text'].apply(self.clean_text)
        result = result.sample(frac=1).reset_index(drop=True)  # shuffle result
        # result.to_csv('../../Data/preprocessed_data.csv', index=False)
        return result

    def clean_text(self, phrase):
        cleared_text = re.sub(r'[^а-яА-ЯёЁ]', ' ', phrase)
        lower_text = cleared_text.lower()
        words = lower_text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        filtered_text = ' '.join(filtered_words)
        lemmatized_words = self.mystem.lemmatize(filtered_text, )
        lemmatized_text = ''.join(lemmatized_words).replace('\n', ' ')

        return lemmatized_text.strip()