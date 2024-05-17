import pandas as pd
from feature_transform import feature_transformer as ft
from sklearn.utils import resample
from model import churnmodels as cm,compare_models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

def read_data(csv_path):
    # Read the CSV file into a Pandas DataFrame
    print("Reading Data.....")
    df = pd.read_csv(csv_path)
    # Drop duplicates and reset index
    df = df.drop_duplicates().reset_index(drop=True)
    # Drop 'msno' column if it's no longer needed and set new index
    df.drop('msno', axis=1, inplace=True)
    df.index.name = 'msno'
    df.reset_index(inplace=True)
    return df

def main():
    # Define the path to the CSV file
    csv_path = "./data/final_merged.csv"
    
    # Read and preprocess the data
    data = read_data(csv_path)
    data = ft.feature_engineering(data)
    data_encoded = ft.encode_features(data)
    data_scaled = ft.scale_features(data_encoded)
    data_selected=ft.select_features(data_scaled)
    data_downsampled=ft.downsample_data(data_selected)
  
    # Split the data into features and target
    X1 = data_downsampled.drop('is_churn', axis=1)
    y1 = data_downsampled['is_churn']

    # Split the data into training and testing sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)  # 70% training and 30% testing
    
    #without downsampled data
    X2 = data_selected.drop('is_churn', axis=1)
    y2 = data_selected['is_churn']

    # Split the data into training and testing sets
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)  # 70% training and 30% testing

    X3,y3=ft.upsample_data(X2,y2)

    # Split the resampled data into training and testing sets
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

    # Run the models menu
    while True:
        print("Choose a model to run:")
        print("1. Random Forest with Undersampling")
        print("2. Random Forest without Sampling")
        print("3. Log Regression with Upsampling")
        print("4. Log Regression without Sampling")
        print("5. XGBOOST with Undersampling")
        print("6. XGBOOST without Sampling")
        print("7. SVM with Undersampling")
        print("8. SVM without Sampling")
        print("9. Compare all models")
        print("10. Run all models one after the other.")
        print("0. Exit")
        choice = input("Enter your choice: ")

        if choice == '0':
            break

        if choice == '1':
            cm.run_random_forest(X1_train, X1_test, y1_train, y1_test)

        if choice == '2':
            cm.run_random_forest_wosample(X2_train, X2_test, y2_train, y2_test)
        
        if choice == '3':
            cm.run_log_reg(X3_train, X3_test, y3_train, y3_test)

        if choice == '4':
            cm.run_log_reg_wosample(X2_train, X2_test, y2_train, y2_test)

        if choice == '5':
            cm.run_xgboost(X2_train, X2_test, y2_train, y2_test)

        if choice == '6':
            cm.run_xgboost_wosample(X2_train, X2_test, y2_train, y2_test)

        if choice == '7':
            cm.run_svm(X1_train, X1_test, y1_train, y1_test)

        if choice == '8':
            cm.run_svm_wosample(X2_train, X2_test, y2_train, y2_test)

        if choice == '9':
            compare_models('./models', './data/test.csv')

        if choice == '10':
            cm.run_random_forest(X1_train, X1_test, y1_train, y1_test)
            cm.run_random_forest_wosample(X2_train, X2_test, y2_train, y2_test)
            cm.run_log_reg(X3_train, X3_test, y3_train, y3_test)
            cm.run_log_reg_wosample(X2_train, X2_test, y2_train, y2_test)
            cm.run_xgboost(X2_train, X2_test, y2_train, y2_test)
            cm.run_xgboost_wosample(X2_train, X2_test, y2_train, y2_test)
            cm.run_svm(X1_train, X1_test, y1_train, y1_test)
            cm.run_svm_wosample(X2_train, X2_test, y2_train, y2_test)
        else:
            print("Invalid choice. Please choose again.")


if __name__ == "__main__":
    main()