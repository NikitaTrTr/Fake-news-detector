import pickle


def logreg(X_test):
  pickled_model = pickle.load(open('logreg.pkl', 'rb'))
  pred = pickled_model.predict(X_test)
  return pred
