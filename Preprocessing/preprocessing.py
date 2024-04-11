import pandas as pd
from functions import clean_text


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
result = result.sample(frac=1).reset_index(drop=True)  # shuffle result

result.to_csv('../Data/preprocessed_data.csv', index=False)
