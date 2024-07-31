import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


data_path = Path("..") / "init" / "Fraud detection.csv"

# Load the dataset
data = pd.read_csv(data_path)


# Data Preprocessing
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

# Encode categorical variables
data = pd.get_dummies(data, columns=['IP Country', 'Issuing Country', 'Currency'], drop_first=True)

# Convert 'label' to binary
data['label'] = data['label'].apply(lambda x: 1 if x == 'y' else 0)

# Separate features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
# Using Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("Random Forest Classifier Report")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))

# Using XGBoost Classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation
print("XGBoost Classifier Report")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred_xgb))