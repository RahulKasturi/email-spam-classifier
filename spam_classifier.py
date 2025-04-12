import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("email_spam.csv")

# Combine title and text columns
df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Encode labels (spam = 1, not spam = 0)
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['type'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['combined_text'], df['label_encoded'], test_size=0.2, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Balance the data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

# Train Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Predictions and Evaluation
y_pred = clf.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))
