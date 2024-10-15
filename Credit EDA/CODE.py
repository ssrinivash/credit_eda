# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

 #Importing the Dataset
application_data = pd.read_csv('application_data.csv')
previous_application_data = pd.read_csv('previous_application.csv')

#running Data Inspection
print(application_data.head())
print(application_data.info())

#Data Quality Checking and Imputation of Missing Values
imputer = SimpleImputer(strategy='mean')  # Using mean imputation for missing values
application_data.fillna(application_data.mean(), inplace=True)
print(application_data.isnull().sum())

#Validating Data Types and Converting if Needed
application_data['DAYS_BIRTH'] = abs(application_data['DAYS_BIRTH']) / 365  # Convert birth days to age
application_data['AMT_INCOME_TOTAL'] = application_data['AMT_INCOME_TOTAL'].astype(float)  # Ensure it's a float type

#Binning Continuous Variables
application_data['INCOME_BINNED'] = pd.cut(application_data['AMT_INCOME_TOTAL'], bins=5, labels=['Low', 'Medium', 'High', 'Very High', 'Super High'])

#Data Imbalance Checking
print(application_data['TARGET'].value_counts())
sns.countplot(application_data['TARGET'])
plt.title('Target Class Distribution')
plt.show()

# Univariate Analysis
# Categorical Variables
categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
for col in categorical_cols:
    sns.countplot(x=col, data=application_data)
    plt.show()

# Numerical Variables
numerical_cols = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH']
for col in numerical_cols:
    sns.histplot(application_data[col], kde=True)
    plt.show()

#   Bivariate & Multivariate Analysis
# Correlation between numerical variables
corr = application_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation between Numerical Variables')
plt.show()

# Reading Preveous Application CSV file
print(previous_application_data.head())
previous_application_data.fillna(previous_application_data.mean(), inplace=True)

# Merging the two datasets
merged_data = pd.merge(application_data, previous_application_data, on='SK_ID_CURR', how='left')

# Data Preprocesing for Machine Learning
# Encoding categorical variables
le = LabelEncoder()
application_data['CODE_GENDER'] = le.fit_transform(application_data['CODE_GENDER'])
application_data['FLAG_OWN_CAR'] = le.fit_transform(application_data['FLAG_OWN_CAR'])
application_data['FLAG_OWN_REALTY'] = le.fit_transform(application_data['FLAG_OWN_REALTY'])

# Splitting the data into training and testing sets
X = application_data.drop(columns=['TARGET'])
y = application_data['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training and Prediction using RandomForest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# doing Predictions
y_pred = clf.predict(X_test)

# Model Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

