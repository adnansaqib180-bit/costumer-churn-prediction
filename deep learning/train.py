import pandas as pd

from keras import Sequential
from keras.layers import Dense

df = pd.read_csv('costumer-churn-prediction/clasic machine learning/Data.csv')

print(df.head())
print(df.columns)
print(df.isnull().sum())

df = df.drop(columns=['gender','PaperlessBilling','customerID','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['MultipleLines'] = df['MultipleLines'].map({'No internet service': 'No','No':'No','Yes':'Yes','No phone service':'No'})
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines']
for col in yes_no_columns:
    df[col] = df[col].map({'Yes':1,'No':0})
df['Contract'] = df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})
df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

df = pd.get_dummies(data=df,columns=['PaymentMethod'],drop_first=True,dtype=int)  
print(df.head())
print(df.info())
x = df.drop(columns= ['Churn'])
y = df['Churn']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42)

model = Sequential()
model.add(Dense(5,activation='relu',input_dim=13))
model.add(Dense(2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossantropy',optimizer='Adam',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=7,validation_split=.2)

from sklearn.metrics import confusion_matrix, f1_score ,accuracy_score

predictions = model.predict(x_test)[0]
threshold = 0.5
if predictions > threshold:
    predictions = 1
else :
    predictions = 0

print(confusion_matrix(predictions,y_test))
print(f1_score(predictions,y_test))
print(accuracy_score(predictions,y_test))

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])