import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from scipy.sparse import load_npz

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../Dataset')
model_dir = os.path.join(script_dir, '../Models')
vis_dir = os.path.join(script_dir, '../Visualisations')
os.makedirs(vis_dir, exist_ok=True)

# Load data
X_test = load_npz(os.path.join(data_dir, 'X_test_vec.npz'))
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()

# Model names and file paths
models = {
    "Logistic Regression": "Logistic_Model.pkl",
    "Naive Bayes": "Naive_Bayes_Model.pkl",
    "SVM": "SVM_Model.pkl",
    "Random Forest": "Random_Forest_Model.pkl",
    "XGBoost": "XGBoost_Model.pkl",
    "Passive Aggressive": "Passive_Aggressive_Model.pkl"
}

# Store metrics for plotting
results = []

for name, file in models.items():
    model_path = os.path.join(model_dir, file)
    model = joblib.load(model_path)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})
    
    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{name.replace(' ', '_')}_confusion_matrix.png"))
    plt.close()

# Create dataframe and barplot
df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x="F1 Score", y="Model", data=df, palette="viridis")
plt.title("Model Comparison - F1 Scores")
plt.xlim(0.94, 1.00)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "model_f1_comparison.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x="Accuracy", y="Model", data=df, palette="rocket")
plt.title("Model Comparison - Accuracy")
plt.xlim(0.94, 1.00)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "model_accuracy_comparison.png"))
plt.close()

print("Model comparison visualisations saved.")


# Save metrics
docs_dir = os.path.join(script_dir, '../Docs')
os.makedirs(docs_dir, exist_ok=True)

with open(os.path.join(docs_dir, "model_scores.txt"), "w") as f:
    f.write("Model Accuracy and F1 Scores\n")
    f.write("===================================\n")
    for entry in results:
        line = f"{entry['Model']}:\n  Accuracy: {entry['Accuracy']:.4f}\n  F1 Score: {entry['F1 Score']:.4f}\n\n"
        f.write(line)

print("Model scores saved to Docs/model_scores.txt")