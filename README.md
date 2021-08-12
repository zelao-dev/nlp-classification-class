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
