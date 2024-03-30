import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


df = pd.read_csv(r"C:\Users\ASHISH KUMAR\OneDrive\Desktop\Python\Dataset\IMDB Dataset.csv")


def remove_html_tag(text):
   soup = BeautifulSoup(text, "html.parser")
   return soup.get_text()

df['review'] = df['review'].apply(remove_html_tag)

# Remove Special Charcater

df['review'] = df['review'].str.replace("[^a-zA-Z]", ' ', regex = True)
df['review'] = df['review'].str.replace("\s+", ' ', regex = True).str.strip()

#Remove Punctuation

punctuation_pattern = f"[{string.punctuation}]"
df["review"] = df["review"].str.replace(punctuation_pattern, "", regex=True)

df["review"] = df["review"].str.lower()

df.isnull().sum()

#Tokenization

df['review'] = df['review'].apply(word_tokenize)

#stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(token_list):
    return [word for word in token_list if word not in stop_words]

df['review'] = df['review'].apply(remove_stopwords)

#Stemming

def stem_tokens(token_list):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in token_list]

df["review"] = df["review"].apply(stem_tokens)

#Frequency of Repeated words

# Extract high-frequency words
all_tokens = [token for sublist in df["review"] for token in sublist]
word_freq = Counter(all_tokens)
most_common_words = word_freq.most_common(20)
most_common_words

# Visualize it using wordcloud
wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Hide the axes
plt.show()


#Tf - IDF

# Join the tokens back into strings for the vectorizer
df["joined_review"] = df["review"].apply(" ".join)
text_data = df["joined_review"]
tfidf_vectorizer = TfidfVectorizer(max_features=2000)

# Fit the vectorizer to the text data and transform it into TF-IDF features
x_tfidf = tfidf_vectorizer.fit_transform(text_data)
x_tfidf

# Converting categorical labels to numerical form

df["sentiment_numeric"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_tfidf, df["sentiment_numeric"], test_size=0.2, random_state=1)

# Scale data
scaler = MaxAbsScaler()

# Scale the training and test sets
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train MLP model
mlp_model = MLPClassifier()
mlp_model.fit(x_train_scaled, y_train)


y_pred = mlp_model.predict(x_test_scaled)

# Calculate accuracy
mlp_accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {mlp_accuracy}")

# Print classification report for a detailed performance analysis
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()


def get_scores(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob[:, 1] >= threshold).astype(int)

    scores = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Roc_Auc': roc_auc_score(y_true, y_pred_prob[:, 1])
    }




# Sample text to predict sentiment
sample_text = [
    "This movie was a great watch with brilliant performances and a gripping plot!",  # Positive
    "An absolute waste of time, the worst movie I've seen in a long while.",  # Negative
    "I found the movie to be mediocre, not terrible but not great either.",  # Neutral
    "The cinematography was stunning, but the storyline was lacking and unoriginal.",  # Neutral/Negative
    "The film was a masterpiece with a perfect blend of drama and action, a must-watch!",  # Positive
    "It was an okay movie; I neither liked it nor disliked it particularly.",  # Neutral
    "The plot twist at the end was predictable and uninspired.",  # Negative
    "A stellar cast, but the film fell flat due to poor writing.",  # Negative
    "I loved the special effects, but the characters were not very compelling.",  # Neutral/Negative
    "The movie was well-received by critics but I didn't find it very interesting.",  # Neutral
    "This film is overrated, I had high expectations but was sadly disappointed.",  # Negative
    "What an entertaining experience, I was on the edge of my seat the whole time!"  # Positive
]

for text in sample_text:
    # Convert the text to TF-IDF features
    sample_tfidf = tfidf_vectorizer.transform([text])

    # Scale the features because we scaled during training
    sample_scaled = scaler.transform(sample_tfidf)

    # Make a prediction
    prediction_prob = mlp_model.predict_proba(sample_scaled)

    # Get the predicted probability of the positive class
    positive_prob = prediction_prob[0, 1]

    # Apply threshold to convert probabilities to class labels
    prediction = 1 if positive_prob > 0.5 else 0

    # Output the prediction
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"{text} ->  {sentiment}")

    import pickle

    pickle.dump(mlp_model, open('review.sav', 'wb'))
    # print(f"Predicted sentiment: {sentiment}")