# Text-Classification-for-Spam-Detection
This project implements a Spam Detection System using machine learning. The system classifies email messagees as "Spam" or "Ham" based on their texual content. It incorporates TF-IDF Vectorization, Naive Bayes, Logistic Regression, XGBoost, and hyperparameter tuning to achieve high accuracy.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#Dataset)
- [Technologies Used](#technologies-used)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributions](#contributions)
- [Future Work](#future-works)

## Introduction
Email spam poses a major threat by exposing users to scams, phishing, and malware. Current filters often fail, leading to misclassified emails. This project aims to create a more reliable spam detection system using machine learning to develop a more dependable spam detection system.

## Dataset
- **Source:** [Spam Dataset](https://github.com/Apaulgithub/oibsip_taskno4/blob/main/spam.csv)
- The dataset contains email messages labeled as:
  - **`spam`**: Unwanted or junk emails.
  - **`ham`**: Legitimate emails.
- **Columns in the dataset**:
  - `Category`: Specifies whether the message is "spam" or "ham".
  - `Message`: The actual content of the email.
- **Preprocessing Steps**:
  1. Dropped unnecessary columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`).
  2. Handled missing values by removing rows with null values.
  3. Applied **text preprocessing**:
     - Lowercase the text.
     - Removed stopwords, numbers, and punctuation.
     - Applied lemmatization to reduce words to their base forms.

### Example Rows from the Dataset
| Category | Message                                                                                 |
|----------|-----------------------------------------------------------------------------------------|
| ham      | Go until jurong point, crazy.. Available only in bugis n great world la e buffet...     |
| spam     | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121... |
| ham      | Ok lar... Joking wif u oni...                                                           |

### Key Points
- The dataset has been split into **75% training** and **25% testing** sets.
- The **TF-IDF Vectorizer** was used to convert the textual data into numerical features for machine learning models.
  
## Technologies Used

- **Languages:** `Python`
 - **Libraries:**
    - Machine Learning: `scikit-learn`, `xgboost`
    - Text Processing: `nltk`, `re`
    - Visualization: `matplotlib`, `seaborn`, `wordcloud`
    - Utilities: `pandas`, `numpy`, `pickle`
    - Server: `Flask`
    - Web GUI: `HTML`, `CSS`

## Workflow

1. **Data Cleaning:** Remove unnecessary columns and handle missing values.
2. **Text Preprocessing:** Apply lemmatization and stopword removal.
3. **Dataset Splitting:** Divide data into training (75%) and testing (25%).
4. **TF-IDF Vectorization:** Convert textual data into numerical features.
5. **Model Training:**
  - Train models using:
      - Multinomial Naive Bayes.
      - Random Forest Regression.
      - XGBoost (with hyperparameter tuning using GridSearchCV).
6. **Model Evaluation:** Measure performance using metrics like accuracy, precision, recall, and F1-score.
7. **Save Model:** Save the trained model using pickle.
8. **Web Interface:** Build a Flask app for real-time email spam detection.
                         
## Installation
1. [Download Python if you don't have it installed on your system python version should be >=3.9](https://www.python.org/downloads/release/python-3130/)
2.  [Download the GitHub desktop If needed](https://desktop.github.com/download/)
3.  Clone the repository where you want:
```bash
  https://github.com/DoshiTirth/Spam_Detection_with_ML.git
```
4. Go inside the directory
```bash
  cd Spam_Detection_with_ML
```
5. Install the required dependencies:
```bash
  pip install -r requirements.txt
```
## Usage
1. Make sure you are still in the same directory.
2. start Flask server:
```bash
   python app.py
```
3. Open your browser and navigate to this:
```bash
http://127.0.0.1:5000
```
4. Enter email message in the GUI and check if it is spam or ham.

## Results
**Naive Bayes Model Metrics:**
- Accuracy: 95%
- Precision: 94%
- Recall: 93%
**Hyperparameter Tuned XGBoost Metrics:**
- Accuracy: 96%
- Precision: 96%
- Recall: 96%
Although both models performed well, hyperparameter tuning further optimized the XGBoost classifier.

## Contributions
- Applied hyperparameter tuning to XGBoost using GridSearchCV.
- Built a real-time spam detection system with a Flask web interface.
- Saved the trained model using pickle for reuse.
  
## Future Work
- Experiment with advanced NLP models like BERT or GPT for improved accuracy.
- Incorporate a multilingual spam detection feature.
- Deploy the system online using platforms like Heroku or AWS for public use.

