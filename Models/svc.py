import pickle


def svc(X_test):
  pickled_model = pickle.load(open('svc.pkl', 'rb'))
  pred = pickled_model.predict(X_test)
  return pred
