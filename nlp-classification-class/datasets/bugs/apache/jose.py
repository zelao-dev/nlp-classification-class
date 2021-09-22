import json
import builtins
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



filename00 = './class0.txt'
filename01 = './class1.txt'

def extract_descriptions (filename):
	bug_text = open(filename).read()
	bug_texts = bug_text.split('\n')

	return [ json.loads(bug)['DESCRIPTION'] for bug in bug_texts if len(bug) > 0 ]

# CARREGAMENTO DOS DADOS
bugs00 = extract_descriptions(filename00)
bugs01 = extract_descriptions(filename01)
bugs_all = bugs00 + bugs01
y = [0 for bug in bugs00 ] + [1 for bug in bugs01]
###

# extracao das caracteristicas
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(bugs_all)
###

# TREINAMENTO
#classifier = DecisionTreeClassifier()
#results = cross_validate(classifier, X, y, cv=10)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# RESULTADOS
final = []

# Decision Tree
decisiontree = DecisionTreeClassifier()
parameters = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random'),'min_samples_split': [2,5,10], 'min_samples_leaf':[1,5,10]

}

classifier = GridSearchCV(decisiontree, parameters, cv=10)
classifier.fit(X_train, y_train)
results = classifier.score(X_test, y_test)

final.append({'algorithm:': classifier.estimator,'best_params': classifier.best_params_, 'best_score': classifier.score(X_test, y_test)})
#final.append(classifier.score(X_test, y_test))
#print(final)


# LinearSVC
linearsvc = LinearSVC()
parameters = { 'C': [0.1, 1, 1.5, 2.0, 10, 20, 100], 'max_iter': (1000, 1500, 2000)}
classifier = GridSearchCV(linearsvc, parameters, cv=10)
classifier.fit(X_train, y_train)
results = classifier.score(X_test, y_test)

final.append({'algorithm:': classifier.estimator, 'best_params': classifier.best_params_, 'best_score': classifier.score(X_test, y_test)})
#final.append(classifier.score(X_test, y_test))
print(final)
#

# Random Forest
random = RandomForestClassifier()
parameters = {'n_estimators': [100, 200], 'max_features': ['sqrt', 'log2'], 'min_samples_leaf': [1, 2, 4]}
classifier = GridSearchCV(random, parameters, cv=10)
classifier.fit(X_train, y_train)
results = classifier.score(X_test, y_test)

final.append({'algorithm:': classifier.estimator, 'best_params': classifier.best_params_, 'best_score': classifier.score(X_test, y_test)})
#print(final)

# LogisticRegression
logistic = LogisticRegression()
parameters = {'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'C': [100, 10, 1.0, 0.1, 0.01]}
classifier = GridSearchCV(logistic, parameters, cv=10)
classifier.fit(X_train, y_train)
results = classifier.score(X_test, y_test)

final.append({'algorithm:': classifier.estimator, 'best_params': classifier.best_params_, 'best_score': classifier.score(X_test, y_test)})
#print(final)

# Gaussian Naive Bayes
# ajustes na matriz dos dados X
Xarray = X.toarray()
X_train, X_test, y_train, y_test = train_test_split(Xarray, y)

gaussiannb = GaussianNB()
parameters = {'var_smoothing': np.logspace(0,-9, num=100)}
classifier = GridSearchCV(gaussiannb, parameters, cv=10)
classifier.fit(X_train, y_train)
results = classifier.score(X_test, y_test)

final.append({'algorithm:': classifier.estimator, 'best_params': classifier.best_params_, 'best_score': classifier.score(X_test, y_test)})
print(final, "porém esta issue aqui diz que não faz muito sentido utilizar GaussianNB em classifacação de textos: https://github.com/scikit-learn/scikit-learn/issues/6440")

