from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

dataset_path = "./breast+cancer+wisconsin+diagnostic/wdbc.data"

breasts_cancer_db = pd.read_csv(dataset_path, header=None)

X, y = breasts_cancer_db.drop(columns=[0, 1]), breasts_cancer_db[1]

print("Proporção entre classes:")
print(f"(%) B = {(y[y == "B"].count() / y.count()) * 100 } | (%) M = {(y[y == "M"].count() / y.count()) * 100}")
print()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

knn = KNeighborsClassifier()

print("K = 5")
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
acc = accuracy_score(y_test, predictions)
conf_matrix = 
print(f"Acurácia: {acc}")
print("Matriz de confusão: ")
print()
