from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

cross_val_score
# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset loaded successfully.")
print("Features:", X.shape)
print("Target:", y.shape)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
model = KNeighborsClassifier(n_neighbors=3)


# Train model
model.fit(X_train, y_train)

print("Model trained successfully.")


# Predictions
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Cross-validation
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Scores:")
print(cv_scores)

print("Mean CV Accuracy:", cv_scores.mean())
print("Mean CV Accuracy (%):", cv_scores.mean() * 100)