# 1. Import Necessary Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

# Function to preprocess text
def preprocess_text(text):
    # Implement preprocessing steps like tokenization, removing stopwords, stemming
    pass



# 2. Load and Prepare the Data
df = pd.read_csv('tweet_emotions.csv')
df['Processed_Text'] = df['comment'].apply(preprocess_text)

# 3. Feature Extraction
cv = CountVectorizer(max_features=1500, ngram_range=(1, 2))
X = cv.fit_transform(df['Processed_Text']).toarray()
y = df['sentiment'].values  # Assuming the sentiment column is labeled 'sentiment'

# 4. Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# 5. Model Training
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 7. Prediction (Example)
new_feedback = ["This product was great!"]
processed_feedback = [preprocess_text(feedback) for feedback in new_feedback]
feedback_features = cv.transform(processed_feedback).toarray()
prediction = classifier.predict(feedback_features)
print("Sentiment Prediction:", "Positive" if prediction[0] == 1 else "Negative")

# 8. Save the Model and Vectorizer
joblib.dump(classifier, "sentiment_classifier.pkl")
joblib.dump(cv, "count_vectorizer.pkl")