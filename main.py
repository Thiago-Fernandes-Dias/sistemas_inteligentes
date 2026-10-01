from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabulate import tabulate
import pandas as pd
import numpy as np

def main():
    dataset_path = "./breast+cancer+wisconsin+diagnostic/wdbc.data"

    breasts_cancer_db = pd.read_csv(dataset_path, header=None)

    X, y = breasts_cancer_db.drop(columns=[0, 1]), breasts_cancer_db[1]

    print("Proporção entre classes:")
    print(f"(%) B = {(y[y == "B"].count() / y.count()) * 100 } | (%) M = {(y[y == "M"].count() / y.count()) * 100}")
    print()

    ks = [1, 5, 11, 15, 21, 27]

    min_max_k_results = {}
    z_score_k_results = {}
    for k in ks:
        min_max_k_results[k] = evaluate_setting(X, y, MinMaxScaler(), k)
        z_score_k_results[k] = evaluate_setting(X, y, StandardScaler(), k)
    
    print("Normalização por mínimos e máximos:")
    print()
    print(create_table(min_max_k_results))
    print()

    print("Normalização por Z-score: ")
    print()
    print(create_table(z_score_k_results))
    print()
    


def evaluate_setting(X: pd.DataFrame, y: pd.Series, scaler, k: int) -> np.ndarray:
    X_normalized = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, train_size=469, random_state=32)

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    return confusion_matrix(y_pred=predictions, y_true=y_test)

def create_table(k_results: dict[int, np.ndarray]) -> str:
    tb_headers = ["K", "# True positive", "# False positives", "# False negatives", "# True negatives", "Acurácia"]

    tb_data = []
    for k, cm in k_results.items():
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]
        acc = acc_from_cm(cm)
        tb_data.append([k, tp, fp, fn, tn, acc])

    return tabulate(tb_data, headers=tb_headers, floatfmt=".5f")


def acc_from_cm(cm: np.ndarray) -> float:
    return cm.trace() / cm.sum()


if __name__ == "__main__":
    main()
