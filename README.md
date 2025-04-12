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
```
### 2.Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Classifier
```bash
python spam_classifier.py
```
