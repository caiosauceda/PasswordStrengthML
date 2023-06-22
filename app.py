import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Mensagens explicativas para as classes de força de senha
print("0 significa: a força da senha é fraca;"
      "1 significa: a força da senha é média;"
      "2 significa: a força da senha é forte;")

# Lendo apenas as primeiras 1000 linhas do arquivo CSV
data = pd.read_csv("data.csv", on_bad_lines='skip', nrows=1000)
print(data.head())

print("--------------------------------------------------")

# Removendo registros com valores ausentes e mapeando as classes de força de senha
data = data.dropna()
data["strength"] = data["strength"].map({0: "Fraca",
                                         1: "Média",
                                         2: "Forte"})

# Função para tokenizar cada caractere da senha
def word(password):
    character = []
    for i in password:
        character.append(i)
    return character

x = np.array(data["password"])
y = np.array(data["strength"])

# Criando uma matriz de recursos usando TfidfVectorizer
tdif = TfidfVectorizer(tokenizer=word)
x = tdif.fit_transform(x)

# Dividindo os dados em conjuntos de treinamento e teste
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.05,
                                                random_state=42)

# Treinando o modelo de classificação RandomForest
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# Avaliando a precisão do modelo nos dados de teste
print(model.score(xtest, ytest))

# Obtendo a senha do usuário usando a função input()
user = input("Digite a senha: ")

# Convertendo a senha em uma matriz de recursos e fazendo a previsão de força
data = tdif.transform([user]).toarray()
output = model.predict(data)

# Imprimindo a força prevista da senha
print(output)
