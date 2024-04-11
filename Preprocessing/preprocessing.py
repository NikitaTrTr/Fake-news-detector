import pandas as pd
import numpy as np
import re
from pymystem3 import Mystem
from tqdm import tqdm
from nltk.tokenize import word_tokenize

from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
s = stopwords.words('russian')
s.extend(['фото', 'новость', 'риа', 'лента', 'тасс', 'коммерсантъ'])
stop_words = set(s)


def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub('[^а-яА-ЯёЁ]', ' ', text)  # remove not russian letters
    text = word_tokenize(text.lower())  # tokenize by words aka split
     # normal form of words
     text = [morph.normal_forms(token)[0] for token in text
            if token not in stop_words and len(token) > 2]
    text = " ".join(text) # return string
    return text


lapsha = pd.read_csv('../Data/lapsha.csv', sep='\t')
panorama = pd.read_csv('../Data/panorama.csv', sep='\t')
lenta = pd.read_csv('../Data/lenta.csv')

lapsha['label'] = 1  # is fake
panorama['label'] = 1  # is fake
lenta['label'] = 0  # is not fake

result = pd.concat([lapsha[['text', 'label']], panorama[['text', 'label']], lenta[['text', 'label']]])
result = result.dropna(subset = ['text'])
result = result[result['text'] != '']



result['text'] = result['text'].apply(clean_text)

result = result.sample(frac = 1).reset_index(drop=True)  # shuffle result

result.to_csv('../Data/preprocessed_data.csv', index=False)
