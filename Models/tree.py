import pickle


def tree(X_test):
  pickled_model = pickle.load(open('tree.pkl', 'rb'))
  pred = pickled_model.predict(X_test)
  return pred
