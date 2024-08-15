# Sentiment Analysis Project Documentation

## 1. Introduction

**Objective**: To build a machine learning model that classifies text into positive, negative, or neutral sentiments. This documentation covers the data preprocessing steps, model development, and evaluation results.

## 2. Data Preprocessing

### 2.1 Data Collection

**Source**: Describe where the data is sourced from, e.g., movie reviews, social media posts, etc.

```python
import pandas as pd

# Load dataset
data = pd.read_csv('sentiment_data.csv')
```

### 2.2 Data Exploration

**Exploration**: Understand the dataset by inspecting its structure and distribution.

```python
# Display basic information
print(data.info())

# Show the first few rows
print(data.head())
```

**Visualization**: Plot class distribution to understand sentiment distribution.

```python
import matplotlib.pyplot as plt

# Plot class distribution
data['sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```

### 2.3 Text Cleaning

**Text Cleaning Steps**:
- Remove punctuation
- Convert to lowercase
- Remove stop words
- Tokenization

```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stop words
    return ' '.join(tokens)

data['cleaned_text'] = data['text'].apply(clean_text)
```

### 2.4 Data Splitting

**Train-Test Split**: Split the data into training and test sets.

```python
from sklearn.model_selection import train_test_split

X = data['cleaned_text']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3. Model Development

### 3.1 Feature Extraction

**TF-IDF Vectorization**: Convert text data into numerical features using TF-IDF.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### 3.2 Model Selection

**Model Choice**: For this project, we'll use a Logistic Regression model.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
```

### 3.3 Hyperparameter Tuning

**Grid Search**: Optimize hyperparameters using GridSearchCV.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

best_model = grid_search.best_estimator_
```

## 4. Model Evaluation

### 4.1 Evaluation Metrics

**Metrics**: Use accuracy, precision, recall, and F1-score to evaluate the model.

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = best_model.predict(X_test_tfidf)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
```

**Confusion Matrix**: Visualize the confusion matrix.

```python
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
```

### 4.2 Model Performance

**Analysis**: Summarize the performance metrics and discuss potential improvements.

## 5. Conclusion

**Summary**: Recap the key findings, the performance of the model, and future work.

- **Summary**: The Logistic Regression model achieved an accuracy of X% and performed well in distinguishing between different sentiments.
- **Future Work**: Possible improvements include experimenting with different models, incorporating more features, or using more advanced NLP techniques.
