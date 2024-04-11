import pandas as pd
import warnings
import re
import pickle
warnings.filterwarnings('ignore')
from transformers import TextClassificationPipeline, AutoTokenizer
from pymystem3 import Mystem
mystem = Mystem()

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
s = stopwords.words('russian')
s.extend(['фото', 'новость', 'риа', 'лента', 'тасс', 'коммерсантъ'])
stop_words = set(s)

def get_mean_w2v_vector(sentence):
    model = pickle.load(open('./Models/word2vecmodel.pkl', 'rb'))
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


def clean_text(phrase):
    cleared_text = re.sub(r'[^а-яА-ЯёЁ]', ' ', phrase)
    lower_text = cleared_text.lower()

    words = lower_text.split()
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = ' '.join(filtered_words)

    lemmatized_words = mystem.lemmatize(filtered_text, )

    # Удаляем лишние пробелы и символ переноса строки, оставшиеся после лемматизации
    lemmatized_text = ''.join(lemmatized_words).replace('\n', ' ')

    return lemmatized_text.strip()


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
    tokenizer = AutoTokenizer.from_pretrained('./Models/rubert/', return_tensors='pt')
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    
    # tokenized_query = tokenizer(query, truncation=True, max_length=50, padding=True, add_special_tokens = True)
    prediction = pipe(query, return_all_scores=True)
    
    return prediction
