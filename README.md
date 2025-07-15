# Fake News Detection System

A machine learning–powered web app that detects whether a news article is **Real** or **Fake**, using Natural Language Processing (NLP) and a **Random Forest classifier**. Built using real-world news data, this project showcases the **entire ML lifecycle** — from preprocessing and model evaluation to **deployment and feedback learning**.

## Live App:
https://singhalya3011-fake-news-detection.streamlit.app

## Dataset

Used the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset), containing:

- `text`: Full article content  
- `title`: News headline  
- `label`: `real` or `fake`  

→ **Total: ~9,000 labeled articles**

## Workflow

### 1. Preprocessing (`Code/preprocess_text.py`)
- Lowercased, removed URLs, punctuation, digits
- Removed stopwords & applied stemming using NLTK
- Tokenized and joined clean words
- Saved cleaned dataset to CSV
  
### 2. Exploratory Data Analysis (`Code/eda.py`)
- Class distribution
- Text length distribution
- Frequent words in fake/real news  
→ All visuals saved in `Visualisations/`

### 3. Feature Extraction
- Compared `CountVectoriser` and `TF-IDF`
- Final model used **CountVectoriser** based on performance

### 4. Multiple Model Training (`Code/train_model.py`)
Trained and evaluated 6 ML classifiers:
- Logistic Regression  
- Multinomial Naive Bayes  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost  
- Passive Aggressive Classifier  

Evaluated using:
- Precision
- Recall
- F1-score
- Confusion Matrix

## Model Evaluation Highlights
| Model                | Accuracy | F1 Score |
|----------------------|----------|----------|
| Logistic Regression  | 98.66%   | 98.60%   |
| Naive Bayes          | 94.31%   | 94.08%   |
| SVM                  | 99.35%   | 99.32%   |
| Random Forest        | 99.81%   | 99.80%   |
| XGBoost              | 99.77%   | 99.75%   |
| Passive Aggressive   | 99.42%   | 99.39%   |

**Best Model:** `Random Forest + CountVectorizer`  
→ Saved in `/Models/final_model.pkl` and used in the app

## App Interface (`app.py`)
- Built using **Streamlit**
- Paste any news article or headline
- Shows:
  - Prediction (`Real` or `Fake`)
  - Confidence %
- User feedback system:
  - "Was this prediction correct?"
  - Option to confirm or correct label

## Continuous Learning System

### Feedback System Features:
- Saves user feedback to `Feedback_Data.csv`
- Appends it to `Combined_News.csv` (used for future training)
- Automatically removes duplicates
- Clears feedback file after every submission

### Retrain:
- `Combined_News.csv` grows with every use
- Enables smooth future retraining based on live feedback

## Tech Stack

- `Python`
- `Pandas`, `NumPy`
- `Scikit-learn`, `XGBoost`
- `NLTK` (text preprocessing)
- `Streamlit` (web app)
- `Joblib` (model saving)
- `Matplotlib`, `Seaborn` (EDA/plots)

## How to Run

```bash
# Clone this repository
git clone https://github.com/singhalya30112004/fake-news-detection
cd fake-news-detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
cd Code
streamlit run app.py
```

## Author

Alya Singh  
[LinkedIn](https://www.linkedin.com/in/alya-singh/)  
[GitHub: @singhalya30112004](https://github.com/singhalya30112004)


## License
MIT License – feel free to use, share, and modify!
