# train.py
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

# Load training data
train_df = pd.read_csv("data/train.csv")
X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]

# Train model (Random Forest)
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("✅ Model trained and saved to model/model.pkl")