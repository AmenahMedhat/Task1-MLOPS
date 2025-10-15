# validate.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os

# Load model and test data
model = joblib.load("model/model.pkl")
test_df = pd.read_csv("data/test.csv")

X_test = test_df.drop("Survived", axis=1)
y_test = test_df["Survived"]

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Save metrics
metrics = {"accuracy": acc}
os.makedirs("results", exist_ok=True)
with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save confusion matrix plot
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")

print(f"✅ Validation done. Accuracy: {acc:.2f}")
