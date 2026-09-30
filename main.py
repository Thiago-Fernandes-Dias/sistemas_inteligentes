from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

def main():
    dataset_path = "./breast+cancer+wisconsin+diagnostic/wdbc.data"

    breasts_cancer_db = pd.read_csv(dataset_path, header=None)

    X, y = breasts_cancer_db.drop(columns=[0, 1]), breasts_cancer_db[1]

    print("Proporção entre classes:")
    print(f"(%) B = {(y[y == "B"].count() / y.count()) * 100 } | (%) M = {(y[y == "M"].count() / y.count()) * 100}")
    print()

    print("K = 5, normalização por mínimos e máximos")

    evaluate_setting(X, y, MinMaxScaler(), 5)

    print("K = 5, normalização por Z-score")

    evaluate_setting(X, y, StandardScaler(), 5)

    print("K = 21, normalização por mínimos e máximos")
    
    evaluate_setting(X, y, MinMaxScaler(), 21)

    print("k = 21, normalização por Z-score")

    evaluate_setting(X, y, StandardScaler(), 21)

def evaluate_setting(X: pd.DataFrame, y: pd.Series, scaler, k: int):
    X_normalized = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, train_size=469, random_state=32)

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print(f"Acurácia: {acc}")
    print("Matriz de confusão: ")
    print(conf_matrix)
    print()

if __name__ == "__main__":
    main()
