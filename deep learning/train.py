import pandas as pd

# from keras import Sequential
# from keras.layers import Dense

df = pd.read_csv('costumer-churn-prediction/clasic machine learning/Data.csv')

print(df.head())
print(df.columns)
print(df.isnull().sum())

df = df.drop(columns=['gender','PaperlessBilling','customerID','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
numeric_columns = ['MonthlyCharges','tenure','TotalCharges']


df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

df['MultipleLines'] = df['MultipleLines'].map({'No internet service': 'No','No':'No','Yes':'Yes','No phone service':'No'})
    
print(df.head())
print(df.info())
for col in df.columns:
    print(df[col].value_counts)

