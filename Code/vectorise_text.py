import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load cleaned data
df = pd.read_csv(os.path.join(script_dir, '../Dataset/Cleaned_News.csv'))

# Features and Labels
X = df['clean_text']
y = df['label']

# Replace NaN with empty strings
X = X.fillna("")

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise TF-IDF vectoriser
vectoriser = TfidfVectorizer(max_features=5000)  # top 5000 words

# Fit and transform the training set
X_train_vec = vectoriser.fit_transform(X_train)

# Transform the test set
X_test_vec = vectoriser.transform(X_test)

# Save the vectorised data (as .npz files)
from scipy.sparse import save_npz
save_npz(os.path.join(script_dir, '../Dataset/X_train_vec.npz'), X_train_vec)
save_npz(os.path.join(script_dir, '../Dataset/X_test_vec.npz'), X_test_vec)

# Save labels
y_train.to_csv(os.path.join(script_dir, '../Dataset/y_train.csv'), index=False)
y_test.to_csv(os.path.join(script_dir, '../Dataset/y_test.csv'), index=False)

# Save the TF-IDF vectoriser
joblib.dump(vectoriser, os.path.join(script_dir, '../Models/tfidf_vectoriser.pkl'))

print("TF-IDF vectorisation complete and saved.")