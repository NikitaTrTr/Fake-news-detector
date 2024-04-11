import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
import torch
import warnings
import multiprocessing
from gensim.models import Word2Vec
import pickle

warnings.filterwarnings('ignore')

cores = multiprocessing.cpu_count()

data = pd.read_csv('../Data/preprocessed_data.csv')

X = data.text
y = data.label

model = Word2Vec(min_count=20,
                      window=5,
                      vector_size=300,
                      sample=6e-5,
                      alpha=0.03,
                      min_alpha=0.0007,
                      negative=20,
                      workers=cores-1)

sent = [row.split() for row in X]
model.build_vocab(sent, progress_per=10000)


def get_mean_w2v_vector(sentence):
    Sum = 0
    Count = 0

    try:
        words = str(sentence).split()
    except TypeError:
        words = []

    for w in words:
        if w in model.wv:
            Sum += model.wv[w]
            Count += 1

    if Count == 0:
        return 0

    return Sum / Count


def prepare_data(X, y, vectors_dim = 300):

  HIDDEN = vectors_dim

  model.train(sent, total_examples=model.corpus_count, epochs=30, report_delay=1)

  NewCols = ['col'+str(i) for i in range(HIDDEN)]

  X_vectors = X.map(get_mean_w2v_vector)

  X = pd.concat([X, X_vectors], axis=1)

  X.columns.values[0] = "text"
  X.columns.values[1] = "vectors"

  IdxTrain = []

  for ix, row in X.iterrows():
      if not isinstance(row['vectors'],np.ndarray):
          IdxTrain.append(ix)

  X.drop(index=IdxTrain, inplace=True)
  y = y.drop(index=IdxTrain)

  X[NewCols] = pd.DataFrame(X['vectors'].tolist(), index=X.index)

  X.drop(['text', 'vectors'], axis=1, inplace=True)

  return X, y


X_full, y_full = prepare_data(X, y)


pickle.dump(model, open('word2vecmodel.pkl', 'wb'))


logreg = LogisticRegression()
logreg.fit(X_full, y_full)
pickle.dump(logreg, open('logreg.pkl', 'wb'))


tree = DecisionTreeClassifier()
tree.fit(X_full, y_full)
pickle.dump(tree, open('tree.pkl', 'wb'))


random_forest = RandomForestClassifier()
random_forest.fit(X_full, y_full)
pickle.dump(random_forest, open('random_forest.pkl', 'wb'))

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  {'probability': [True]},
 ]
clf = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=3,  n_jobs=-1)
clf.fit(X_full, y_full)
svc_best = clf.best_estimator_
svc_best.fit(X_full, y_full)
pickle.dump(svc_best, open('svc.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(data.text, data.label, test_size=0.25, random_state=42)

def prepare_x_train_test_data(X_train, X_test, y_train, vectors_dim = 300):

  HIDDEN = vectors_dim

  model.train(sent, total_examples=model.corpus_count, epochs=30, report_delay=1)

  NewCols = ['col'+str(i) for i in range(HIDDEN)]

  X_train_vectors = X_train.map(get_mean_w2v_vector)
  X_test_vectors = X_test.map(get_mean_w2v_vector)

  X_train = pd.concat([X_train, X_train_vectors], axis = 1)
  X_test = pd.concat([X_test, X_test_vectors], axis = 1)

  X_train.columns.values[0] = "text"
  X_train.columns.values[1] = "vectors"

  X_test.columns.values[0] = "text"
  X_test.columns.values[1] = "vectors"

  IdxTrain = []

  for ix, row in X_train.iterrows():
      if not isinstance(row['vectors'],np.ndarray):
          IdxTrain.append(ix)

  X_train.drop(index=IdxTrain, inplace=True)
  y_train = y_train.drop(index=IdxTrain)

  for ix, row in X_test.iterrows():
      if not isinstance(row['vectors'],np.ndarray):
          row['vectors'] = np.array([1/HIDDEN]*HIDDEN)


  X_train[NewCols] = pd.DataFrame(X_train['vectors'].tolist(), index=X_train.index)
  X_test[NewCols] = pd.DataFrame(X_test['vectors'].tolist(), index=X_test.index)

  X_train.drop(['text', 'vectors'], axis=1, inplace=True)
  X_test.drop(['text', 'vectors'], axis=1, inplace=True)

  return X_train, X_test, y_train

X_full, X_exam, y_full = prepare_x_train_test_data(X_train, X_test, y_train)

task_type = "GPU" if torch.cuda.is_available() else "CPU"
train_pool = Pool(data=
    X_full,
    label=y_full
)

valid_pool = Pool(data=
    X_exam,
    label=y_test
)
model_catboost = CatBoostClassifier(verbose=500, use_best_model=True, task_type=task_type)
model_catboost.fit(train_pool, eval_set=valid_pool)
pickle.dump(model_catboost, open('catboost.pkl', 'wb'))