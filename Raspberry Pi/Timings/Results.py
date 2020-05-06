import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time

start_time = time.time()

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = train.drop("file_name", 1)
test = test.drop("file_name", 1)

X_train = train.drop(columns=["ripeness_index"])
y_train = train["ripeness_index"].values

X_test = test.drop(columns=["ripeness_index"])
y_test = test["ripeness_index"].values

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(accuracy)

print("Time taken: %0.5f seconds" % (time.time() - start_time))
# from sklearn import metrics
# k_range = list(range(1, 21))
# scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     accuracy = metrics.accuracy_score(y_test, y_pred)*100
#     scores.append(accuracy)
#     print("k =",k,", acc =",accuracy)