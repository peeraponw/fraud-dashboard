import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


data_path = Path("..") / "init" / "Fraud detection.csv"

# Load the dataset
data = pd.read_csv(data_path)
# Convert 'Tx Datetime' to datetime object
data['Tx Datetime'] = pd.to_datetime(data['Tx Datetime'])

# Extract features from 'Tx Datetime'
data['Tx Year'] = data['Tx Datetime'].dt.year
data['Tx Month'] = data['Tx Datetime'].dt.month
data['Tx Day'] = data['Tx Datetime'].dt.day
data['Tx Hour'] = data['Tx Datetime'].dt.hour
data['Tx Minute'] = data['Tx Datetime'].dt.minute
data['Tx Second'] = data['Tx Datetime'].dt.second

# Drop unnecessary columns
data.drop(['Tx Datetime', 'Holder Name', 'Card Number', 'Remark', 'Payer IP'], axis=1, inplace=True)

# Convert 'label' to binary
data['label'] = data['label'].apply(lambda x: 1 if x == 'y' else 0)

# Separate features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical columns
categorical_cols = ['IP Country', 'Issuing Country', 'Currency']

# Define the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Amount', 'Tx Year', 'Tx Month', 'Tx Day', 'Tx Hour', 'Tx Minute', 'Tx Second']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create a pipeline with preprocessing and model
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
rf_pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump(rf_pipeline, 'rf_model.pkl')

# # # Predictions
y_pred_rf = rf_pipeline.predict(X_test)

# Evaluation
print("Random Forest Classifier Report")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))

# # # ------------------ # # #
# Train and save the XGBoost model
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

xgb_pipeline.fit(X_train, y_train)
joblib.dump(xgb_pipeline, 'xgb_model.pkl')

# Predictions
y_pred_xgb = xgb_pipeline.predict(X_test)

# Evaluation
print("XGBoost Classifier Report")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_xgb))