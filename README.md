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

1. [Download and install python if not installed already](https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe)
2. [Download and Install Github if not installed Already](https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe)

3. Clone the repository:
```bash
   git clone https://github.com/DoshiTirth/Text-Classification-for-Spam-Detection.git
```
3. Navigate to the project directory:
```bash
  cd Text-Classification-for-Spam-Detection
```
5. Install the required dependencies:
```bash
  pip install -r requirements.txt
```
## **Usage**
1. Make sure you still in the project directory. 
2. Run the Python script to Start The Flask :
   ```python
   python Flask_server.py
   ```
3. Since the Flask server is running,Open another terminal and navigate to the Repository directory.
    ```python
    cd Text-Classification-for-Spam-Detection
    ```
5. Test the Saved Predict Model using API request and Steamlit GUI:
   ```streamlit
   streamlit run  GUI.py
   ```
   There should be a pop up in a web browser
6. To see the code on how the process of Training the models open the jupyter notebook
   ```Open With Jupyter notebook or Visual Studio code
   MLF_SPAM_DETECTION.ipynb
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
