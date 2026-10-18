from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabulate import tabulate
import pandas as pd
import numpy as np


def main():
    dataset_path = "./credit/credit.csv"

    credit_db = pd.read_csv(dataset_path)

    X, y = credit_db.drop(columns=['default']), credit_db['default']

    print("Proporção entre classes:")
    print(
        f"(%) 2 - Yes {(y[y == 2].count() / y.count()) * 100} | (%) 1 - No = {(y[y == 1].count() / y.count()) * 100}"
    )
    print()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.9)

    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(X_train, y_train)
    
    preds = dt.predict(X_test)

    print(accuracy_score(y_test, preds))



    




# def evaluate_setting(X: pd.DataFrame, y: pd.Series, scaler, k: int) -> np.ndarray:
#     X_normalized = scaler.fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_normalized, y, train_size=469, random_state=32
#     )

#     knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

#     knn.fit(X_train, y_train)
#     predictions = knn.predict(X_test)

#     return confusion_matrix(y_pred=predictions, y_true=y_test)


# def create_table(k_results: dict[int, np.ndarray]) -> str:
#     tb_headers = [
#         "K",
#         "# True positive",
#         "# False positives",
#         "# False negatives",
#         "# True negatives",
#         "Acurácia",
#     ]

#     tb_data = []
#     for k, cm in k_results.items():
#         tp = cm[1][1]
#         fp = cm[0][1]
#         fn = cm[1][0]
#         tn = cm[0][0]
#         acc = acc_from_cm(cm)
#         tb_data.append([k, tp, fp, fn, tn, acc])

#     return tabulate(tb_data, headers=tb_headers, floatfmt=".5f")


# def acc_from_cm(cm: np.ndarray) -> float:
#     return cm.trace() / cm.sum()


if __name__ == "__main__":
    main()
