# KKBOX User Churn Prediction

## Project Overview

This project aims to predict whether a user of KKBOX, a music streaming platform, will churn. Accurate churn predictions can help businesses implement targeted retention strategies, enhance user engagement, and increase revenue. The project utilizes four popular machine learning models: Random Forest, Logistic Regression, SVM, and XGBoost, testing each on both imbalanced and balanced datasets.

## Dataset

The dataset is available on Kaggle and can be accessed [here](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data).

## GitHub Repository

The complete project code and documentation are available in our [GitHub repository](https://github.com/SSaklecha/ChurnPred).

## Requirements

- Python Version: 3.10.6
- Dependencies: Install all required packages using the command : pip install -r requirements.txt
  
## Deployment Steps


Download and unzip the churnpred.zip file.
Navigate to the Code folder.
Install Dependencies: Run the following command to install the required packages. pip install -r requirements.txt

1. Run : <pip install -r requirements.txt> in your terminal to install the required packages. Ensure you are in the Code folder path (your_path/code).
2. Data Fetching:Run the getdata.py file (your_path/getdata.py) to fetch the dataset. If you prefer to use a pre-cleaned file, the final_merged file is available in the data folder. Skip to step 4 if using the pre-cleaned file.
3. EDA and Data Cleaning:Run the EDA_clean.py file to execute preprocessing steps and display EDA plots. Close each popup graph to view the subsequent plots.
4. Modeling and Evaluation:Run the main.py file (your_path/main.py).Use the iterative menu to choose which model to run next. For each model, its classification report and ROC curve are displayed.An evaluation of 1,000 unseen records in test.csv (located in the data folder) is performed, and the results are displayed.
5. Choose option 0 to exit or run a different model. Choose option 9 to see a comparison of all models. Choose 10 to run all models one after the other.
   
Note: Models are saved as pickle files in the model folder. Delete these files if re-running the models is required.

## File/Folder Descriptions
- data/: Contains data files fetched by getdata.py.
- model/: Contains pickle files of all models.
- requirements.txt: Contains all dependencies needed for the project.
- main.py: Main controller file.
- getdata.py: Script to source the original dataset.
- EDA_clean.py: Script for EDA and data cleaning.
- model.py: Contains the churnmodels class for model building and evaluation.
- feature_transformer.py: Contains the feature_transformer class for feature engineering.

## Team Contributions
- Business Understanding: Kashish Thakur, Yuti Khamker
- Data Collection: Swetha Sivakumar
- Data Preprocessing and Transformation: Yuti Khamker, Sourab Saklecha, Kashish Thakur
- EDA: Sourab Saklecha, Swetha Sivakumar
- Model Building and Evaluation: Yuti Khamker, Sourab Saklecha, Kashish Thakur, Swetha Sivakumar
- Modular Programming: Kashish Thakur, Sourab Saklecha
- Documentation: Kashish Thakur, Yuti Khamker
