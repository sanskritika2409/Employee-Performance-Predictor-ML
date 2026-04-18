import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.data_generator import generate_data
from src.preprocessing import preprocess_data
from src.model import train_model, evaluate_model
from src.visualization import plot_confusion_matrix, plot_feature_importance

# -------------------------
# 1. Generate Data
# -------------------------
data = generate_data(500)
print("\nDataset Preview:\n", data.head())
import matplotlib.pyplot as plt

# Save dataset preview as image
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=data.head().values,
                 colLabels=data.columns,
                 loc='center')

plt.savefig("images/dataset_preview.png")
plt.close()

# Save dataset
data.to_csv("data/employee_data.csv", index=False)

# -------------------------
# 2. Preprocess
# -------------------------
X, y = preprocess_data(data)

# -------------------------
# 3. Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 4. Train
# -------------------------
model = train_model(X_train, y_train)

# -------------------------
# 5. Evaluate
# -------------------------
acc, report, cm, y_pred = evaluate_model(model, X_test, y_test)

print("\nAccuracy:", acc)
print("\nReport:\n", report)

# -------------------------
# 6. Visualize
# -------------------------
plot_confusion_matrix(cm)
plot_feature_importance(model, X.columns)

# -------------------------
# 7. Sample Prediction
# -------------------------
sample = np.array([[30, 8, 1, 50000, 60, 6]])
pred = model.predict(sample)

print("\nSample Prediction:", pred)
import joblib

joblib.dump(model, "models/model.pkl")
print("Model saved successfully!")