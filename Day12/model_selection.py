from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset loaded successfully.")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# --------------------------------------------------
# 2. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 3. Model 1 - Logistic Regression
# --------------------------------------------------

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

lr_param_grid = {
    "model__C": [0.1, 1, 10],
    "model__solver": ["liblinear", "lbfgs"]
}

lr_grid = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=5,
    scoring="accuracy"
)

lr_grid.fit(X_train, y_train)

print("\nLogistic Regression")
print("Best parameters:", lr_grid.best_params_)
print("Best CV score:", lr_grid.best_score_)


# --------------------------------------------------
# 4. Model 2 - Random Forest
# --------------------------------------------------

rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [None, 3, 5]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=5,
    scoring="accuracy"
)

rf_grid.fit(X_train, y_train)

print("\nRandom Forest")
print("Best parameters:", rf_grid.best_params_)
print("Best CV score:", rf_grid.best_score_)


# --------------------------------------------------
# 5. Compare models on test data
# --------------------------------------------------

lr_pred = lr_grid.predict(X_test)
rf_pred = rf_grid.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nModel Comparison")
print("----------------")
print("Logistic Regression Test Accuracy:", lr_accuracy)
print("Random Forest Test Accuracy:", rf_accuracy)


# --------------------------------------------------
# 6. Select best model
# --------------------------------------------------

if lr_accuracy >= rf_accuracy:
    best_model = lr_grid.best_estimator_
    best_model_name = "Logistic Regression"
else:
    best_model = rf_grid.best_estimator_
    best_model_name = "Random Forest"

print("\nSelected Model:", best_model_name)


# --------------------------------------------------
# 7. Save complete fitted pipeline
# --------------------------------------------------

model_file = "iris_best_model.joblib"

joblib.dump(best_model, model_file)

print("Model saved successfully:", model_file)


# --------------------------------------------------
# 8. Load saved model
# --------------------------------------------------

loaded_model = joblib.load(model_file)

print("Model loaded successfully.")


# --------------------------------------------------
# 9. Inference test
# --------------------------------------------------

sample = X_test[0].reshape(1, -1)

prediction = loaded_model.predict(sample)

print("\nInference Test")
print("Input:", sample)
print("Predicted class:", prediction[0])
print("Actual class:", y_test[0])

if prediction[0] == y_test[0]:
    print("Inference test: PASSED")
else:
    print("Inference test: FAILED")