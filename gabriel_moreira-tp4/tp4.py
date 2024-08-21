# Trabalho Prático 4 - Aprendizagem de Máquina

# INF 420 - Inteligência Artificial I
# Universidade Federal de Viçosa
# 1º Semestre de 2024
# Professor: Julio Cesar Soares dos Reis

# Aluno: Gabriel Moreira Marques
# Matrícula: 108207

# Importando as bibliotecas necessárias
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carregando o dataset
df = pd.read_csv("koi_data.csv")

# Visualizando as primeiras linhas do dataset
print(df.head())

# Separando features e rótulos
X = df.drop(columns=['KOI', 'LABEL'])  # Substitua 'LABEL' pelo nome correto da coluna de rótulos
y = df['LABEL']

# Dividindo o dataset em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Naive Bayes (Baseline)
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# 2. Decision Tree
# (Adicionar variação da altura máxima da árvore e resultados gráficos)

# 3. Support Vector Machines (SVM)
# (Adicionar comparação entre kernels linear e RBF)

# 4. k-Nearest Neighbors (k-NN)
# (Adicionar variação do número k de vizinhos e resultados gráficos)

# 5. Random Forest
# (Adicionar variação do número de árvores e resultados gráficos)

# 6. Gradient Tree Boosting
# (Adicionar variação do número de iterações e resultados gráficos)

# 7. Multi-layer Perceptron (MLP)
# (Adicionar variação da função de ativação e resultados gráficos)

# Comparação final entre os métodos
# (Incluir matriz de confusão, precisão, revocação e F1)
