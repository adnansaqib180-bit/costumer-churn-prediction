import pandas as pd

# from keras import Sequential
# from keras.layers import Dense

df = pd.read_csv('costumer-churn-prediction/clasic machine learning/Data.csv')

print(df.head())
print(df.columns)
print(df.isnull().sum())

df = df.drop(columns=['gender','customerID'])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
numeric_columns = ['MonthlyCharges','tenure','TotalCharges']

un_cleaned_catagorical = []
for col in df.columns:
    if df[col].dtype == object or df[col].dtype == 'string':
        if df[col].nunique() > 2:
            un_cleaned_catagorical.append(col)
to_clean = ['MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for col in to_clean:
    df[col] = df[col].map({'No internet service': 'No','No':'No','Yes':'Yes','No phone service':'No'})


df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
x = df.drop('Churn',axis=1)
y= df['Churn']
