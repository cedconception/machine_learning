import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Iris dataset
dataset = pd.read_csv("iris.csv")

# I separe features (X) and target variable (y)
X = dataset[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
y = dataset["variety"]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)

#Confusion Matrices (3x3 for Iris dataset)
knn_confusion_matrix = confusion_matrix(y_test, knn_predictions)
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)

# Print Confusion Matrices
print("KNN Confusion Matrix:")
print(knn_confusion_matrix)

print("\nDecision Tree Confusion Matrix:")
print(dt_confusion_matrix)

# Evaluate model accuracy (using accuracy_score)
knn_accuracy = accuracy_score(y_test, knn_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"KNN Accuracy: {knn_accuracy:.4f}")
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")


# df = pd.DataFrame({'Y_connu': y_test, 'Y_predit': knn_predictions})
# print(df)
