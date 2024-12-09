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

## **Introduction**
Fake news is a pressing issue in the digital age, where misinformation spreads rapidly. This project employs a machine learning-based approach to analyze and classify news articles.

---

## **Dataset**
- **Source:** [Dataset Link](#)
- **Details:** The dataset contains labeled news articles, where:
  - `1` represents fake news.
  - `0` represents real news.
- **Preprocessing Steps:**
  - Removed null values and unnecessary columns.
  - Applied lemmatization and stopword removal for text cleaning.

---

## **Technologies Used**
- **Languages:** Python
- **Libraries:** 
  - Machine Learning: `scikit-learn`
  - Text Processing: `nltk`, `re`, `string.punctuation`
  - Visualization: `matplotlib`, `seaborn`
  - Utilities: `pandas`, `numpy`, `joblib`

---

## **Workflow**
![Workflow Diagram](#) *(Include an image or describe the process)*

Steps include:
1. **Data Cleaning**: Handle missing values and remove unnecessary columns.
2. **Text Preprocessing**: Tokenize, lemmatize, and remove stopwords.
3. **Feature Engineering**: Compute length and punctuation features.
4. **TF-IDF Vectorization**: Convert cleaned text to numerical data.
5. **Model Training**: Train multiple models, including Decision Trees and Random Forests.
6. **Hyperparameter Tuning**: Optimize using `GridSearchCV`.
7. **Evaluation**: Measure model performance using metrics like accuracy and classification reports.

---

## **Installation**
1. Install Python and Git if not already installed.
2. Clone the repository:
   ```bash
   git clone <repository-link>```
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

