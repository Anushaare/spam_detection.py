# SPAM EMAIL DETECTION USING MACHINE LEARNING

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Create Dataset
data = {
    'message': [
        'Congratulations you won a lottery',
        'Hi how are you',
        'Claim your free prize now',
        'Let us meet tomorrow',
        'You have won cash reward',
        'Call me later',
        'Free entry in contest',
        'Are you coming to class',
        'Win money now',
        'Good morning'
    ],

    'label': [
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham'
    ]
}

# Convert into DataFrame
df = pd.DataFrame(data)

print("DATASET\n")
print(df)


# Convert Labels into Numbers
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("\nDATASET AFTER LABEL CONVERSION\n")
print(df)


# Split Dataset
X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Convert Text into Numerical Features
vectorizer = CountVectorizer()

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)


# Train Model
model = MultinomialNB()

model.fit(X_train_features, y_train)


# Make Predictions
y_pred = model.predict(X_test_features)

print("\nPREDICTIONS\n")
print(y_pred)


# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print(accuracy)

print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_test, y_pred))


# Test with Custom Message
sample_message = ["Congratulations! You won free tickets"]

sample_features = vectorizer.transform(sample_message)

prediction = model.predict(sample_features)

print("\nCUSTOM MESSAGE PREDICTION\n")

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")