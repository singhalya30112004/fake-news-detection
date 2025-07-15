import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../Dataset/Cleaned_News.csv')
vis_dir = os.path.join(script_dir, '../Visualisations')
os.makedirs(vis_dir, exist_ok=True)

# Load cleaned data
df = pd.read_csv(data_path)

# Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, palette='pastel')
plt.title('Real vs Fake News Count')
plt.xticks([0, 1], ['Fake (0)', 'Real (1)'])
plt.xlabel('News Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'class_distribution.png'))
plt.close()

# Text Length Analysis
df['text_length'] = df['clean_text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True, palette='muted', element='step')
plt.title('Distribution of Article Lengths')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.xlim(0, 2000)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'text_length_distribution.png'))
plt.close()

# Most Common Words (Top 20)
def get_top_words(df, label_val, n=20):
    words = ' '.join(df[df['label'] == label_val]['clean_text'].dropna().astype(str)).split()
    return Counter(words).most_common(n)

fake_top_words = get_top_words(df, 0)
real_top_words = get_top_words(df, 1)

# Convert to DataFrame
fake_words_df = pd.DataFrame(fake_top_words, columns=['word', 'count'])
real_words_df = pd.DataFrame(real_top_words, columns=['word', 'count'])

# Plot fake
plt.figure(figsize=(8, 4))
sns.barplot(x='count', y='word', data=fake_words_df, palette='Reds_r')
plt.title('Top 20 Words in Fake News')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'top_words_fake.png'))
plt.close()

# Plot real
plt.figure(figsize=(8, 4))
sns.barplot(x='count', y='word', data=real_words_df, palette='Blues_r')
plt.title('Top 20 Words in Real News')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'top_words_real.png'))
plt.close()

print("EDA visuals saved in Visualisations")