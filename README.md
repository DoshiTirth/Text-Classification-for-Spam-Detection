# Text-Classification-for-Spam-Detection
This project implements a Spam Detection System using machine learning. The system classifies email messagees as "Spam" or "Ham" based on their texual content. It incorporates `TF-IDF Vectorization`,`RandomForestClassifier`, `Logistic Regression`, `Naive Bayes`, `Decision Tree Classifier`, and hyperparameter tuning to achieve high accuracy.

## Table of Contents
- [Project Proposal](Proposal_for_Final_Project.docx)
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
- **Source:** [Dataset Link](spam.csv)
- **Details:** The dataset contains labeled news articles, where:
  - `1` represents `SPAM`.
  - `0` represents `HAM`.
- **Preprocessing Steps:**
  - Removed null values and unnecessary columns.
  - Applied lemmatization and stopword removal for text cleaning.

---

## **Technologies Used**
- **Languages:** `Python`
- **Libraries:** 
  - Machine Learning: `scikit-learn`
  - Text Processing: `nltk`, `re`, `string.punctuation`
  - Visualization: `matplotlib`, `seaborn`
  - Utilities: `pandas`, `numpy`, `joblib`

---

## **Workflow**
![Workflow Diagram](pipeline.png)

Steps include:
1. **Data Cleaning**: Handle missing values and remove unnecessary columns.
2. **Text Preprocessing**: Tokenize, lemmatize, and remove stopwords.
3. **Feature Engineering**: Compute length and punctuation features.
4. **TF-IDF Vectorization**: Convert cleaned text to numerical data.
5. **Standard Scaler**: Scaling ensures that all features contribute equally to the model's learning process
6. **Model Training**: Train multiple models, including Decision Trees and Random Forests.
7. **Hyperparameter Tuning**: Optimize using `GridSearchCV`.
8. **Evaluation**: Measure model performance using metrics like accuracy and classification reports.

---

## **Installation**
1. Install Python and Git if not already installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/DoshiTirth/Text-Classification-for-Spam-Detection.git
   ```
3. Navigate to the project directory:
```bash
  cd Fake-news-Detector
```
5. Install the required dependencies:
```bash
  pip install -r requirements.txt
```
## Usage
1. Make sure you are still in the same directory.
2. Train Models:
```bash
python train_models.py
```
3. start Flask server:
```bash
   python app.py
```
4. Use the GUI:
```bash
streamlit run GUI.py
```
---

## **Results**

### **Model Comparison**
A comparison of accuracy scores across different models trained on the dataset:

| Model                  | Accuracy | Precision | Recall |
|------------------------|----------|-----------|--------|
| Decision Tree          | 93%      | 92%       | 94%    |
| Random Forest          | 96%      | 96%       | 95%    |
| Support Vector Machine | 95%      | 94%       | 96%    |
| Multinomial Naive Bayes| 91%      | 89%       | 92%    |

---

### **Hyperparameter Tuned Model**
The best model after hyperparameter tuning:

- **Algorithm:** Random Forest Classifier
- **Best Hyperparameters:**
  - `n_estimators`: 200
  - `max_depth`: 30
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2

---

### **Performance Metrics**
The following metrics were observed for the tuned model on the test dataset:

- **Accuracy:** 96%
- **Precision:** 96%
- **Recall:** 96%
- **F1-Score:** 96%

---

### **Visualization**
#### Model Performance Plot
![Model Comparison](#) *(Insert or link to the image of your model comparison bar chart)*

#### Classification Report
![Classification Report](#) *(Insert or link to the image of your classification report)*

Although the overall metrics for the original and tuned models are the same, hyperparameter tuning validated the optimal settings and ensured robustness.

---

## Results
**Naive Bayes Model Metrics:**
- Accuracy: 95%
- Precision: 94%
- Recall: 93%
**Hyperparameter Tuned RandomforestClassifier Metrics:**
- Accuracy: 96%
- Precision: 96%
- Recall: 96%
Although both models performed well, hyperparameter tuning further optimized the XGBoost classifier.

## Contributions
- Applied hyperparameter tuning to XGBoost using GridSearchCV.
- Implemented reusable model and preprocessing pipelines.
- Saved trained models using `joblib`.
  
## Future Work
- Explore advanced NLP models like BERT for improved accuracy.
- Expand to multilingual fake news detection.
- Test on larger, more diverse datasets.
