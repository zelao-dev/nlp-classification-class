<h2>Meu código python é o jose.py está em nlp-classification-class/datasets/bugs/apache/jose.py</h2>
<h2>Resultados estão logo abaixo neste README e também na pasta nlp-classification-class/datasets/bugs/apache/README.MD.</h2>

Tarefa
======
Implementar scripts com Python e a biblioteca SKLEARN que façam a classificação de textos dos datasets propostos em sala de aula. Deve ser submetido um repositório no Github, contendo os arquivos do projeto. No projeto, deve ser apresentado no arquivo README a combinação de classificadores e configurações que geraram os melhores valores no procedimento de 10-fold cross-validation.

Como classificadores, considere utilizar:
* Decision Tree
* Linear SVC
* SVC
* Random Forest
* Logistic Regression
* Gaussian Naive Bayes

Bibliotecas interessantes que podem ser consideradas:
* Pipeline
* GridSearch

Referências para serem utilizadas
=================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text

DATASETS e ALUNOS
-----------------
* apache - José
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
  
* linux - Willian
* xia - Ademir
* games - Lucas
* illiterate - Nariane
* mdwe - Sidnei
* ontologies
* pair
* slr
* testing
* xbi - Rene
