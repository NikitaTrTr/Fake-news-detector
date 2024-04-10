import pickle


def pac(X_test):
  pickled_model = pickle.load(open('pac.pkl', 'rb'))
  pred = pickled_model.predict(X_test)
  return pred
