import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import Pool, CatBoostClassifier
import multiprocessing
from gensim.models import Word2Vec
import pickle
from datetime import datetime

class Trainer:
    def __init__(self):
        cores = multiprocessing.cpu_count()
        data = pd.read_csv('../Data/preprocessed_data.csv')
        X = data.text
        y = data.label
        self.model = Word2Vec(min_count=20,
                              window=5,
                              vector_size=300,
                              sample=6e-5,
                              alpha=0.03,
                              min_alpha=0.0007,
                              negative=20,
                              workers=cores - 1)
        self.sent = [row.split() for row in X]
        self.model.build_vocab(self.sent, progress_per=10000)

    def get_mean_w2v_vector(self, sentence):
        Sum = 0
        Count = 0
        try:
            words = str(sentence).split()
        except TypeError:
            words = []
        for w in words:
            if w in self.model.wv:
                Sum += self.model.wv[w]
                Count += 1
        if Count == 0:
            return 0
        return Sum / Count

    def prepare_data(self, X, y, vectors_dim=300):
        HIDDEN = vectors_dim
        self.model.train(self.sent, total_examples=self.model.corpus_count, epochs=30, report_delay=1)
        NewCols = ['col' + str(i) for i in range(HIDDEN)]
        X_vectors = X.map(self.get_mean_w2v_vector)
        X = pd.concat([X, X_vectors], axis=1)
        X.columns.values[0] = "text"
        X.columns.values[1] = "vectors"
        BadIdxTrain = []
        for ix, row in X.iterrows():
            if not isinstance(row['vectors'], np.ndarray):
                BadIdxTrain.append(ix)
        X.drop(index=BadIdxTrain, inplace=True)
        y = y.drop(index=BadIdxTrain)
        X[NewCols] = pd.DataFrame(X['vectors'].tolist(), index=X.index)
        X.drop(['text', 'vectors'], axis=1, inplace=True)
        return X, y

    def get_trained_model(self, X, y, model_name, **parameters):
        if not model_name in ['Logistic_Regression', 'Random_Forest', 'SVM', 'Catboost']:
            raise ValueError(
                f'Model type not supported. Supported types are {["Logistic Regression", "Random Forest", "SVM", "Catboost"]:}')
        if model_name == 'Logistic Regression':
            model = LogisticRegression(**parameters)
        if model_name == 'Random Forest':
            model = RandomForestClassifier(**parameters)
        if model_name == 'SVM':
            model = SVC(**parameters)
        if model_name == 'Catboost':
            model = CatBoostClassifier(**parameters)
        model.fit(self.prepare_data(X, y))
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        pickle.dump(model, open(f'{model_name}_{date_time_string}.pkl', 'wb'))
        return model

