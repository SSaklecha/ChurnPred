import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import numpy as np
from xgboost import XGBClassifier
import warnings
import pickle
import os
import tkinter as tk
from tkinter import ttk

def test_model_accuracy(model_filepath, test_data_filepath='./data/test.csv'):
    # Load the trained model from the pickle file
    print("Testing the model on new values from test.csv........")
    print("---------------------------------------")
    with open(model_filepath, 'rb') as file:
        model = pickle.load(file)

    # Load the test data
    test_data = pd.read_csv(test_data_filepath)
    total_count = test_data.shape[0]
    print(f"Total count of test records: {total_count}")

    # Separate features and the target label
    X_test = test_data.iloc[:, 1:]  # assuming features start from the second column
    y_test = test_data['is_churn']  # target labels

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy of the model: {accuracy}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print("---------------------------------------")

def test_model_accuracy_compare(model_filepath, test_data_filepath='./data/test.csv'):
    # Load the trained model from the pickle file
    print("Testing the model on new values from test.csv........")
    print("---------------------------------------")
    with open(model_filepath, 'rb') as file:
        model = pickle.load(file)

    # Load the test data
    test_data = pd.read_csv(test_data_filepath)
    total_count = test_data.shape[0]
    print(f"Total count of test records: {total_count}")

    # Separate features and the target label
    X_test = test_data.iloc[:, 1:]  # assuming features start from the second column
    y_test = test_data['is_churn']  # target labels

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return accuracy,precision,f1,recall


def compare_models(model_folder, test_data_filepath):
        # List of model files and their descriptive names
        models = {
            "Random Forest with Undersampling": "random_forest_model.pkl",
            "Random Forest without Sampling": "random_forest_model_wosample.pkl",
            "Log Regression with Upsampling": "log_reg_model.pkl",
            "Log Regression without Sampling": "log_reg_model_wosample.pkl",
            "XGBOOST with Undersampling": "XGBOOST_model.pkl",
            "XGBOOST without Sampling": "XGBOOST_model_wosample.pkl",
            "SVM with Undersampling": "SVM_model.pkl",
            "SVM without Sampling": "SVM_model_wosample.pkl"
        }
        
        # Check if all model files are present
        missing_models = [name for name, file in models.items() if not os.path.exists(os.path.join(model_folder, file))]
        if missing_models:
            print("Please ensure all models are trained. Missing models:")
            for model in missing_models:
                print(model)
            return
        
        # Dictionary to store performance metrics
        results = []
        
        # Evaluate each model
        for model_name, model_file in models.items():
            full_path = os.path.join(model_folder, model_file)
            accuracy,precision,f1,recall = test_model_accuracy_compare(full_path)
            results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        print(results_df)
        
        # Colors for the bars
        colors = ['#ADD8E6','#ADD8E6', '#87CEEB','#87CEEB', '#4682B4','#4682B4', '#4169E1','#4169E1']
        
        # Plotting the bar chart for accuracies
        plt.figure(figsize=(12, 6))
        bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=colors)
        
        # Adding the accuracy values on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Comparison of Model Accuracies')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        display_dataframe(results_df)

def display_dataframe(df):

    # Create a new top-level window
    popup = tk.Tk()
    popup.title('Model Comparison Results')
    popup.geometry('800x400')  # Set the size of the window

    # Configure the style of the Treeview for increased row height
    style = ttk.Style(popup)
    style.configure("Treeview", rowheight=40)  # Double the row height

    # Create a frame for the Treeview
    frame = ttk.Frame(popup)
    frame.pack(fill='both', expand=True)

    # Create a Treeview to display the data
    tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
    tree.pack(side='left', fill='both', expand=True)

    # Scrollbar for the Treeview
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)

    # Define headings
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=100)  # You can adjust column width here

    # Insert data into the Treeview
    for index, row in df.iterrows():
        tree.insert("", 'end', values=list(row))

    # Run the GUI
    popup.mainloop()

class churnmodels:

 
    def run_random_forest(X1_train, X1_test, y1_train, y1_test, save_model=True, model_path='./models/random_forest_model.pkl'):
        try:
            # Attempt to load an existing model
            with open(model_path, 'rb') as file:
                clf = pickle.load(file)
            print("Found a pre-trained model. Loaded successfully.")
        except (FileNotFoundError, EOFError):
            # If model is not found or file is corrupted, train a new one
            print("Training Random Forest Model....")
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X1_train, y1_train)
            
            # Optionally save the model
            if save_model:
                with open(model_path, 'wb') as file:
                    pickle.dump(clf, file)
                print(f"Model saved to {model_path}.")

        # Predict on the testing set
        y_pred = clf.predict(X1_test)
        y_prob = clf.predict_proba(X1_test)[:, 1]  # probabilities for the positive class

        # Evaluate the model
        accuracy = accuracy_score(y1_test, y_pred)
        print("Classification Report:")
        print(classification_report(y1_test, y_pred))

        # ROC Curve and AUC
        fpr, tpr, _ = roc_curve(y1_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plotting ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) for Random Forest with Undersampling')
        plt.legend(loc="lower right")
        plt.show()

        test_model_accuracy(model_path)

    def run_random_forest_wosample(X2_train, X2_test, y2_train, y2_test, save_model=True, model_path='./models/random_forest_model_wosample.pkl'):
               
        try:
            # Attempt to load an existing model
            with open(model_path, 'rb') as file:
                clf = pickle.load(file)
            print("Found a pre-trained model. Loaded successfully.")
        except (FileNotFoundError, EOFError):
            # If model is not found or file is corrupted, configure and train a new one
            print("Training Random Forest Model....")
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                random_state=42
            )
            clf.fit(X2_train, y2_train)
            
            # Optionally save the model
            if save_model:
                with open(model_path, 'wb') as file:
                    pickle.dump(clf, file)
                print(f"Model saved to {model_path}.")

        # Predict the response for test dataset
        y_pred = clf.predict(X2_test)

        # Evaluate the model
        print("Classification Report:\n", classification_report(y2_test, y_pred))

        # Calculate ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y2_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Random Forest without Sampling')
        plt.legend(loc="lower right")
        plt.show()

        test_model_accuracy(model_path)

    def run_log_reg(X3_train, X3_test, y3_train, y3_test, save_model=True, model_path='./models/log_reg_model.pkl'):
        warnings.filterwarnings('ignore')
        model = None
        try:
            # Load existing model if available
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Found a pre-trained model. Loaded successfully.")
        except (FileNotFoundError, EOFError):
            # Ignore warnings
            warnings.filterwarnings('ignore')
            print("Training Logistic Regression Model....")
            # Create and fit the logistic regression model with regularization
            model = LogisticRegression(C=0.1)  # Stronger regularization with C=0.1
            model.fit(X3_train, y3_train)

            # Optionally save the model
            if save_model:
                with open(model_path, 'wb') as file:
                    pickle.dump(model, file)
                print(f"Model saved to {model_path}.")

        # Generate predictions for the test set
        y_pred_proba = model.predict_proba(X3_test)[:, 1]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y3_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('RROC Curve for Log Regression with Upsampling')
        plt.legend(loc="lower right")

        # Define a range of training set sizes
        train_sizes = np.linspace(0.1, 1.0, 10)

        # Calculate the training and testing scores for each training set size
        train_scores = []
        test_scores = []
        for train_size in train_sizes:
            n_train = int(train_size * len(X3_train))
            X_train_subset = X3_train[:n_train]
            y_train_subset = y3_train[:n_train]
            model.fit(X_train_subset, y_train_subset)
            y_train_pred = model.predict(X_train_subset)
            y_test_pred = model.predict(X3_test)
            train_scores.append(accuracy_score(y_train_subset, y_train_pred))
            test_scores.append(accuracy_score(y3_test, y_test_pred))

        # Generate predictions for the test set
        y_pred = model.predict(X3_test)

        # Generate classification report
        print(classification_report(y3_test, y_pred))
        plt.show()
        test_model_accuracy(model_path)

    def run_log_reg_wosample(X2_train, X2_test, y2_train, y2_test, save_model=True, model_path='./models/log_reg_model_wosample.pkl'):
        warnings.filterwarnings('ignore')
        try:
            # Load existing model if available
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Found a pre-trained model. Loaded successfully.")
        except (FileNotFoundError, EOFError):
            warnings.filterwarnings('ignore')
            print("Training Logistic Regression Model....")
            # Create and fit the logistic regression model
            model = LogisticRegression()
            model.fit(X2_train, y2_train)

            # Optionally save the model
            if save_model:
                with open(model_path, 'wb') as file:
                    pickle.dump(model, file)
                print(f"Model saved to {model_path}.")

        # Generate predictions for the test set
        y_pred_proba = model.predict_proba(X2_test)[:, 1]

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y2_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Log Regression without Sampling')
        plt.legend(loc="lower right")
        # Generate predictions for the test dataset
        y_pred = model.predict(X2_test)
        # Generate classification report
        print(classification_report(y2_test, y_pred))
        plt.show()
        test_model_accuracy(model_path)

    def run_xgboost(X1_train, X1_test, y1_train, y1_test,save_model=True, model_path='./models/XGBOOST_model.pkl'):
        xgbc = None

        try:
            # Load existing model if available
            with open(model_path, 'rb') as file:
                xgbc = pickle.load(file)
            print("Found a pre-trained model. Loaded successfully.")
        except (FileNotFoundError, EOFError):
            print("Training XGBOOST Model....")
            xgbc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                colsample_bynode=1, colsample_bytree=1, gamma=0,
                                learning_rate=0.01, max_delta_step=0, max_depth=3,
                                min_child_weight=1, n_estimators=500, n_jobs=1,
                                objective='binary:logistic', random_state=123,
                                reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                                subsample=1)
            xgbc.fit(X1_train, y1_train)

            # Optionally save the model
            if save_model:
                with open(model_path, 'wb') as file:
                    pickle.dump(xgbc, file)
                print(f"Model saved to {model_path}.")

        # Predict probabilities for the test set
        y_pred_proba = xgbc.predict_proba(X1_test)[:, 1]

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y1_test, y_pred_proba)
        roc_auc = roc_auc_score(y1_test, y_pred_proba)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for XGBOOST with Undersampling')
        plt.legend(loc="lower right")
        

        # Generate predictions for the test set
        y_pred = xgbc.predict(X1_test)
        print('\nClassification Report:')
        print(classification_report(y1_test, y_pred))
        plt.show()
        test_model_accuracy(model_path)

    def run_xgboost_wosample(X2_train, X2_test, y2_train, y2_test,save_model=True, model_path='./models/XGBOOST_model_wosample.pkl'):
        xgbc = None
        try:
            # Load existing model if available
            with open(model_path, 'rb') as file:
                xgbc = pickle.load(file)
            print("Found a pre-trained model. Loaded successfully.")
        except (FileNotFoundError, EOFError):
            print("Training XGBOOST Model....")
            xgbc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                colsample_bynode=1, colsample_bytree=1, gamma=0,
                                learning_rate=0.01, max_delta_step=0, max_depth=3,
                                min_child_weight=1, n_estimators=500, n_jobs=1,
                                objective='binary:logistic', random_state=123,
                                reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                                subsample=1)
            xgbc.fit(X2_train, y2_train)

            # Optionally save the model
            if save_model:
                with open(model_path, 'wb') as file:
                    pickle.dump(xgbc, file)
                print(f"Model saved to {model_path}.")

        # Predict probabilities for the test set
        y_pred_proba = xgbc.predict_proba(X2_test)[:, 1]

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y2_test, y_pred_proba)
        roc_auc = roc_auc_score(y2_test, y_pred_proba)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for XGBOOST without Sampling')
        plt.legend(loc="lower right")
        

        # Generate predictions for the test dataset
        y_pred = xgbc.predict(X2_test)

        # Generate classification report
        print('\nClassification Report:')
        print(classification_report(y2_test, y_pred))
        plt.show()
        test_model_accuracy(model_path)

    def run_svm(X1_train1, X1_test, y1_train1, y1_test,save_model=True, model_path='./models/SVM_model.pkl'):
        # Initialize model variable
        svm_model = None
        X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train1, y1_train1, test_size=0.2, random_state=42)

        try:
            # Load existing model if available
            with open(model_path, 'rb') as file:
                svm_model = pickle.load(file)
            print("Found a pre-trained model. Loaded successfully.")
        except (FileNotFoundError, EOFError):
            print("Training SVM Model....")

            # Create SVM classifier with specified hyperparameters
            svm_model = SVC(kernel='linear', C=0.1, max_iter=100000, probability=True)  # Ensure probability is True for ROC curve
            svm_model.fit(X1_train, y1_train)

            # Optionally save the model
            if save_model:
                with open(model_path, 'wb') as file:
                    pickle.dump(svm_model, file)
                print(f"Model saved to {model_path}.")

        # Predictions for validation and test sets
        y_val_pred = svm_model.predict(X1_val)
        val_accuracy = accuracy_score(y1_val, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy}")

        y_test_pred = svm_model.predict(X1_test)
        test_accuracy = accuracy_score(y1_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy}")

        # Generate classification report
        report = classification_report(y1_test, y_test_pred)
        print("Classification Report:")
        print(report)

        # Generate ROC curve plot
        y_score = svm_model.decision_function(X1_test)  # Ensure decision_function is used for ROC
        fpr, tpr, _ = roc_curve(y1_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for SVM with Undersampling')
        plt.legend(loc="lower right")
        plt.show()
        test_model_accuracy(model_path)

    def run_svm_wosample(X2_train2, X2_test, y2_train2, y2_test,save_model=True, model_path='./models/SVM_model_wosample.pkl'):
        # Initialize model variable
        svm_model = None
        X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train2, y2_train2, test_size=0.2, random_state=42)

        try:
            # Load existing model if available
            with open(model_path, 'rb') as file:
                svm_model = pickle.load(file)
            print("Found a pre-trained model. Loaded successfully.")
        except (FileNotFoundError, EOFError):
            print("Training SVM Model....")
            # Create SVM classifier with specified hyperparameters
            svm_model = SVC(kernel='linear', C=0.1, max_iter=100000, probability=True)  # Ensure probability=True for ROC curve plotting
            

            # Train SVM model
            svm_model.fit(X2_train, y2_train)

            # Optionally save the model
            if save_model:
                with open(model_path, 'wb') as file:
                    pickle.dump(svm_model, file)
                print(f"Model saved to {model_path}.")

        # Validation and Test predictions
        y_val_pred = svm_model.predict(X2_val)
        val_accuracy = accuracy_score(y2_val, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy}")

        y_test_pred = svm_model.predict(X2_test)
        test_accuracy = accuracy_score(y2_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy}")

        # Generate classification report
        report = classification_report(y2_test, y_test_pred)
        print("Classification Report:")
        print(report)

        # Generate ROC curve plot
        y_score = svm_model.decision_function(X2_test)  # Ensure decision_function is used for ROC
        fpr, tpr, _ = roc_curve(y2_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for SVM without Sampling')
        plt.legend(loc="lower right")
        plt.show()
        test_model_accuracy(model_path)

    

