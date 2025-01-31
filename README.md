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
A spam message is an unsolicited or irrelevant message, typically sent in bulk, often for advertising, phishing, or malicious purposes. This project employs a machine learning-based approch to create a robust and efficient solution for automated spam detection.

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
1. **Data Cleaning**: Address missing values and eliminate unneeded columns.
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
| Decision Tree          | 95%      | 98%       | 97%    |
| Random Forest          | 97%      | 97%       | 100%    |
| Support Vector Machine | 92%      | 92%       | 100%    |
| Multinomial Naive Bayes| 89%      | 89%       | 100%    |

---

### **Hyperparameter Tuned Model**
The best model after hyperparameter tuning:

- **Algorithm:** Random Forest Classifier
- **Best Hyperparameters:**
  - `n_estimators`: 100
  - `max_depth`: None
  - `min_samples_split`: 10
  - `min_samples_leaf`: 1

---

### **Performance Metrics**
The following metrics were observed for the tuned model on the test dataset:
-**Best_Score** `97%`
- **Accuracy:** `96%`
- **Precision:** `96%`
- **Recall:** `100%`
- **F1-Score:** `98%`
- **Cross validation Score** `97%`

---

### **Visualization**
#### Model Performance Plot
![Model Comparison](Model_Perfomance.png) 

#### Classification Report
![Classification Report](model_com.png) 

Although the overall metrics for the original and tuned models are the same, hyperparameter tuning validated the optimal settings and ensured robustness.

---

## Results
**RandomForestClassifier Model Metrics:**
- Accuracy: 97%
- Precision: 97%
- Recall: 100%
- f1-score: 97%

**Hyperparameter Tuned RandomforestClassifier Metrics:**
- Accuracy: 96%
- Precision: 96%
- Recall: 100%
- f1-score: 98%
Although both models performed well, hyperparameter tuning acuracy reduced

## Results Showcase
The ⁠ results includes all the essential outputs and artifacts generated during the project:

⁠*Saved Models*:
   - `Model.pkl`
   - `StandardsScalar.pkl`
   - `TFID.pkl`

## Contributions
- Conducted hyperparameter optimization for the RandomForestClassifier using GridSearchCV
- Developed resuable pipelines for model training and data preprocessing.
- Utilized `joblib` to save the trained models for future use.
  
## Future Work
- Explore advanced NLP models like BERT for improved accuracy.
- Expand to multilingual fake news detection.
- Test on larger, more diverse datasets.
