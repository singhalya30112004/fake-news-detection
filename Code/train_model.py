import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import load_npz
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
X_train = load_npz(os.path.join(script_dir, '../Dataset/X_train_vec.npz'))
X_test = load_npz(os.path.join(script_dir, '../Dataset/X_test_vec.npz'))
y_train = pd.read_csv(os.path.join(script_dir, '../Dataset/y_train.csv')).values.ravel()
y_test = pd.read_csv(os.path.join(script_dir, '../Dataset/y_test.csv')).values.ravel()

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, os.path.join(script_dir, '../Models/Logistic_Model.pkl'))
print("\nModel trained and saved as Logistic_Model.pkl")

# Save evaluation
with open(os.path.join(script_dir, '../Docs/evaluation_report.txt'), "w") as f:
    f.write("\n\n====================\n")
    f.write("Logistic Regression Results\n")
    f.write("====================\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
print("Evaluation report saved to evaluation_report.txt")


# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, os.path.join(script_dir, '../Models/Naive_Bayes_Model.pkl'))
print("\nModel trained and saved as Naive_Bayes_Model.pkl")

# Append to evaluation report
with open(os.path.join(script_dir, '../Docs/evaluation_report.txt'), "a") as f:
    f.write("\n\n====================\n")
    f.write("Multinomial Naive Bayes Results\n")
    f.write("====================\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
print("Evaluation report updated with Naive Bayes results.")