# email-spam-classifier
A Python-based email spam classifier using Random Forest and TF-IDF, with support for imbalanced datasets using SMOTE. Easily reproducible with clear setup instructions and example dataset.
# Email Spam Classifier using Random Forest

This project classifies emails as **spam** or **not spam** using a Random Forest classifier. It uses a TF-IDF approach and handles class imbalance using SMOTE.

## 📁 Dataset
- File: `email_spam.csv`
- Columns:
  - `title`: Subject line of the email
  - `text`: Body of the email
  - `type`: Label (`spam` or `not spam`)

## 🛠️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/RahulKasturi/email-spam-classifier.git
cd email-spam-classifier

