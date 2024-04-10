import pickle


def random_forest(X_test):
  pickled_model = pickle.load(open('random_forest.pkl', 'rb'))
  pred = pickled_model.predict(X_test)
  return pred
