# Spam-Email-Detection-
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sample dataset: Replace this with your own email dataset
# For this example, we'll use the SMS Spam Collection Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
df['cleaned'] = df['message'].apply(preprocess_text)

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].map({'ham': 0, 'spam': 1})  # 0 = Not Spam, 1 = Spam

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to classify new email
def predict_email(email):
    email_cleaned = preprocess_text(email)
    email_vector = vectorizer.transform([email_cleaned])
    prediction = model.predict(email_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Example
new_email = "Congratulations! You've won a free iPhone. Click here to claim."
print("Email:", new_email)
print("Prediction:", predict_email(new_email))
