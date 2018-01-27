import sys
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.externals import joblib


def train_model(X_train, y_train):
    regr = LogisticRegression(solver='sag', multi_class='multinomial', max_iter=40, verbose=1)
    regr.fit(X_train, y_train)
    return regr

if __name__ == '__main__':
    # read the program arguments
    feature_vecs_file = sys.argv[1]
    model_file = sys.argv[2]
    X_train, y_train = load_svmlight_file(feature_vecs_file)
    trained_model = train_model(X_train, y_train)

    y_pred = trained_model.predict(X_train)
    y_true = y_train
    print metrics.accuracy_score(y_true, y_pred)
    # save the model to disk
    joblib.dump(trained_model, model_file)
