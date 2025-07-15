import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

nltk.download('stopwords')

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
df = pd.read_csv(os.path.join(script_dir, '../Dataset/Combined_News.csv'))

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text cleaning function
def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and apply stemming
    cleaned = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned)

# Apply cleaning
df['clean_text'] = df['text'].astype(str).apply(clean_text)

# Save cleaned data
df.to_csv(os.path.join(script_dir, '../Dataset/Cleaned_News.csv'), index=False)

print("Text preprocessing complete. Saved to: Dataset/Cleaned_News.csv")