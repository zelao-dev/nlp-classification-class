[	{'algorithm:': DecisionTreeClassifier(), 'best_params': {'criterion': 'entropy'
, 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}, 'best_scor
e': 0.7777777777777778},
	{'algorithm:': LinearSVC(), 'best_params': {'C': 2.0, '
max_iter': 1000}, 'best_score': 0.8333333333333334},
	{'algorithm:': SVC(), 'best_params': {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'},
'best_score': 0.75}],
	{'algorithm:': RandomForestClassifier(), 'best_params': {'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 200}, 'best_score': 0.8611111111111112},
	{'algorithm:': LogisticRegression(), 'best_params': {'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}, 'best_score': 0.75},
	{'algorithm:': GaussianNB(), 'best_params': {'var_smoothing': 0.001}, 'best_score': 0.6111111111111112}
] porém esta issue aqui diz que não faz muito sentido utilizar GaussianNB em classifacação de textos: https://github.com/scikit-learn/scikit-learn/issues/6440
