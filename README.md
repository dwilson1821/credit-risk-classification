# Credit Risk Classification

## Overview of the Analysis  

The goal of this analysis is to build and evaluate a machine learning model that can predict the credit risk of borrowers using historical lending data. Specifically, this project aims to determine if logistic regression can reliably classify loans as either **healthy** (low risk of default) or **high risk** (likely to default).  

By analyzing this dataset, the peer-to-peer lending company can better assess borrower creditworthiness, potentially reducing financial losses due to loan defaults.

---

## Repository Structure  

```  
credit-risk-classification/  
│  
├── Credit_Risk/  
│   ├── credit_risk_classification.ipynb   # Jupyter Notebook with model development 
│   └── lending_data.csv                   # Historical lending data  
│  
├── README.md                              # Project documentation  
└── Resources/                             # Additional resources (optional)  
```  

---

## Dataset  

The dataset, `lending_data.csv`, contains historical loan data with the following details:  

- **loan_status**: Target variable (0 = healthy loan, 1 = high-risk loan)  
- **Features**: Various borrower attributes used to determine credit risk. These attributes include loan size, interest rate, borrower income, debt to income, number of accounts, deragatory marks, and total debt   

---

## Analysis Steps  

1. **Data Preparation**  
   - Loaded the `lending_data.csv` file into a Pandas DataFrame.  
   - Created the **labels** (y) using the `loan_status` column.  
   - Created the **features** (X) using all other columns.  
   - Split the dataset into training (80%) and testing (20%) subsets using `train_test_split`.  

2. **Logistic Regression Model**  
   - Trained a logistic regression model on the training data.  
   - Used the testing data to predict loan classifications.  
   - Evaluated model performance using:  
     - Confusion Matrix  
     - Classification Report (accuracy, precision, and recall).  

3. **Model Performance**  
   - Analyzed the performance of the logistic regression model to determine how well it predicts **healthy** and **high-risk loans**.  

---

## Results  

### Logistic Regression Model Performance  

- **Accuracy Score**: 0.99 
- **Precision Score**:  
   - **Class 0 (Healthy Loans)**: 1.00  
   - **Class 1 (High-Risk Loans)**: 0.87  
- **Recall Score**:  
   - **Class 0 (Healthy Loans)**: 1.00  
   - **Class 1 (High-Risk Loans)**: 0.95  

### Confusion Matrix  
The confusion matrix provides the breakdown of predictions:  

| **Predicted / Actual** | **Healthy (0)** | **High-Risk (1)** |  
|-------------------------|-----------------|------------------|  
| **Healthy (0)**         | True Positives  | False Positives  |  
| **High-Risk (1)**       | False Negatives | True Negatives   |  

---

## Summary  

The logistic regression model demonstrates strong performance in predicting credit risk based on the evaluation metrics.  

- **Healthy Loans (0)**: The model achieved a precision of 1.00 and a recall of 1.00, indicating that it can correctly identify most healthy loans.  
- **High-Risk Loans (1)**: The model achieved a precision of 0.87 and a recall of 0.95, reflecting its ability to identify high-risk loans, though there may be some false positives.  

### Recommendation  

Given the results:  
- The logistic regression model is suitable for predicting credit risk and can be implemented by the company to evaluate borrower creditworthiness. However, th precision for high-risk loans (87%) is slightly lower, meaning that some loans predicted as high-risk are actually healthy. This could lead to conservative lending decisions where some borrowers are unnecessarily classified as risky. 
  
---

## Instructions to Run the Project  

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/dwilson1821/credit-risk-classification.git  
   cd credit-risk-classification  
   ```  

2. **Set Up Environment**  
   Install the required libraries:  
   ```bash  
   pip install pandas scikit-learn matplotlib jupyterlab  
   ```  

3. **Run the Notebook**  
   Launch Jupyter Lab and open `credit_risk_classification.ipynb`:  
   ```bash  
   jupyter lab  
   ```  

4. **Follow Along**  
   Execute each cell in the notebook to:  
   - Load and prepare the data.  
   - Train the logistic regression model.  
   - Evaluate model performance.  

---

## Tools and Technologies  

- **Python**: For data analysis and machine learning.  
- **Pandas**: Data manipulation and cleaning.  
- **scikit-learn**: Train-test splitting, logistic regression, and model evaluation.  
- **Jupyter Notebook**: Run and document the analysis.  
