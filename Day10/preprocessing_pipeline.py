import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Create sample employee dataset
data = {
    "age": [25, 30, 35, None, 28, 40, 32, None, 27, 45],
    "salary": [30000, 45000, 55000, 60000, None, 80000, 65000, 50000, 40000, 90000],
    "department": [
        "IT",
        "HR",
        "IT",
        "Finance",
        "HR",
        "IT",
        None,
        "Finance",
        "HR",
        "IT"
    ],
    "experience": [1, 3, 5, 7, 2, 10, 6, 4, 2, 12],
    "promoted": [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)


# Separate features and target
X = df.drop("promoted", axis=1)
y = df["promoted"]

print("Features:")
print(X)

print("\nTarget:")
print(y)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# Define columns
numeric_features = ["age", "salary", "experience"]
categorical_features = ["department"]


# Numeric preprocessing
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# Complete ML pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ]
)


# Train model
model_pipeline.fit(X_train, y_train)

print("\nPipeline trained successfully.")


# Predictions
y_pred = model_pipeline.predict(X_test)

print("Predictions:", y_pred)
print("Actual labels:", y_test.to_numpy())


# Accuracy
accuracy = model_pipeline.score(X_test, y_test)

print("\nPipeline Accuracy:", accuracy)
print("Pipeline Accuracy (%):", accuracy * 100)