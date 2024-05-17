import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import json
import os

class feature_transformer:
    def feature_engineering(df):
        print("Performing Feature Engineering.....")
        cutoff_date = pd.to_datetime('2017-03-31')
        df['registration_init_date'] = pd.to_datetime(df['registration_init_year'].astype(int).astype(str) + '-' +
                                                    df['registration_init_month'].astype(int).astype(str) + '-' +
                                                    df['registration_init_date'].astype(int).astype(str))
        df['transaction_date'] = pd.to_datetime(df['transaction_date_year'].astype(int).astype(str) + '-' +
                                                df['transaction_date_month'].astype(int).astype(str) + '-' +
                                                df['transaction_date_date'].astype(int).astype(str))
        df['membership_expire_date'] = pd.to_datetime(df['membership_expire_date_year'].astype(int).astype(str) + '-' +
                                                    df['membership_expire_date_month'].astype(int).astype(str) + '-' +
                                                    df['membership_expire_date_date'].astype(int).astype(str))
        df['days_until_membership_expiration'] = (df['membership_expire_date'] - df['transaction_date']).dt.days
        df['transaction_quarter'] = df['transaction_date'].dt.quarter
        df['account_age_days_at_cutoff'] = (cutoff_date - df['registration_init_date']).dt.days
        df.drop(columns=['Unnamed: 0', 'is_cancel_sum'], axis=1, inplace=True, errors='ignore')
        datetime_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
        df.drop(datetime_cols, axis=1, inplace=True)
        return df

    def encode_features(df):
        print("Encoding Features.....")
        df_encoded = pd.get_dummies(df, columns=['city', 'registered_via', 'payment_method_id'],
                                    prefix=['city', 'registered_via', 'payment_method_id'])
        return df_encoded

    def scale_features(df):
        print("Scaling Features.....")
        df.set_index('msno', inplace=True)
        columns_to_scale = df.select_dtypes(include=['int64', 'float64', 'int32']).columns.drop(['is_churn'])
        df_scaled = df.copy()
        scaler = StandardScaler()
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
        return df_scaled

    def select_features(df):
        selected_features_file = 'selected_feature_list.json'
        if os.path.exists(selected_features_file):
            print("Performing feature selection...")
            with open(selected_features_file, 'r') as file:
                selected_features = json.load(file)
        else:
            print("Performing feature selection... This can take ~2 mins")
            X = df.drop('is_churn', axis=1)
            y = df['is_churn']
            rfc = RandomForestClassifier(random_state=42)
            rfc.fit(X, y)
            feature_importances = rfc.feature_importances_
            importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
            importances_df = importances_df.sort_values(by='Importance', ascending=False)
            top_20_features = importances_df.nlargest(20, 'Importance')['Feature'].tolist()
            selected_features = ['is_churn'] + top_20_features
            with open(selected_features_file, 'w') as file:
                json.dump(selected_features, file)
        final_selected = df[selected_features]
        return final_selected  

    def downsample_data(df):
        print("Preparing Undersampled data.....")
        majority_class = df[df['is_churn'] == 0]
        minority_class = df[df['is_churn'] == 1]
        majority_downsampled = resample(majority_class,
                                        replace=False,
                                        n_samples=len(minority_class),
                                        random_state=42)
        downsampled_data = pd.concat([majority_downsampled, minority_class])
        return downsampled_data

    def upsample_data(X2,y2):
        print("Preparing Oversampled data.....")
        # Use SMOTE to oversample the minority class
        smote = SMOTE(random_state=42)
        X3, y3 = smote.fit_resample(X2, y2)
        return X3,y3