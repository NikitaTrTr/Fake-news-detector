import streamlit as st
import pandas as pd
import warnings
import re
from pymystem3 import Mystem
from nltk.tokenize import word_tokenize
from pymorphy2 import MorphAnalyzer
import pickle
warnings.filterwarnings('ignore')


def get_mean_w2v_vector(sentence):
    model = pickle.load(open('../Models/word2vecmodel.pkl', 'rb'))
    sums = 0
    count = 0

    try:
        words = str(sentence).split()
    except TypeError:
        words = []

    for w in words:
        if w in model.wv:
            sums += model.wv[w]
            count += 1

    if count == 0:
        return 0

    return sums / count


def clean_text(text):
    morph = MorphAnalyzer()
    text = re.sub('[^а-яёА-ЯЁ]', ' ', text)
    text = word_tokenize(text.lower())
    text = [morph.normal_forms(token)[0] for token in text
            if len(token) > 2]
    text = " ".join(text)
    return text


def vectorize(text, vectors_dim = 300):

  HIDDEN = vectors_dim

  NewCols = ['col'+str(i) for i in range(HIDDEN)]

  text_vectors = text.map(get_mean_w2v_vector)
  text = pd.concat([text, text_vectors], axis = 1)

  text.columns.values[0] = "text"
  text.columns.values[1] = "vectors"

  text[NewCols] = pd.DataFrame(text['vectors'].tolist(), index=text.index)
  text.drop(['text', 'vectors'], axis=1, inplace=True)

  return text


def rubert_predict_proba(query, model):
    tokenizer = AutoTokenizer.from_pretrained('=..Models/rubert/', return_tensors='pt')
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)
    
    tokenized_query = tokenizer(query, truncation=True, max_length=50, padding=True, add_special_tokens = True)
    prediction = pipe(tokenized_query.tolist(), return_all_scores=True)
    
    return prediction
