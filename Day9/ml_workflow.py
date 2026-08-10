from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset loaded successfully.")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)


# Create baseline model
model = KNeighborsClassifier(n_neighbors=3)


# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully.")


# Make predictions
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual labels:", y_test)


# Evaluate model
accuracy = model.score(X_test, y_test)

print("Baseline Accuracy:", accuracy)
print("Baseline Accuracy (%):", accuracy * 100)