import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,precision_score
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE 
import keras_tuner as kt


df = pd.read_csv('costumer-churn-prediction/clasic machine learning/Data.csv')
print(df.head())
print(df.columns)
print(df.isnull().sum())

df = df.drop(columns=['PaperlessBilling','customerID','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection'])

df['gender'] = df['gender'].map({'Male':1,'Female':0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['MultipleLines'] = df['MultipleLines'].map({'No internet service': 'No','No':'No','Yes':'Yes','No phone service':'No'})
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines']
for col in yes_no_columns:
    df[col] = df[col].map({'Yes':1,'No':0})
df['Contract'] = df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})
df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

df = pd.get_dummies(data=df,columns=['PaymentMethod'],drop_first=True,dtype=int)  
same_columns = ['TechSupport','StreamingMovies','StreamingTV']
for col in same_columns:
    df[col] = df[col].map({'Yes':1,'No':0,'No internet service':0})
x = df.drop(columns= ['Churn'])
y = df['Churn']

print(df.head())
print(df.info())

x,x_test,y_train,y_test = train_test_split(x,y,random_state=42)

scaler =  StandardScaler()
x_train = scaler.fit_transform(x)
x_test = scaler.transform(x_test)

# smote = SMOTE()
# x_train,y_train = smote.fit_resample(x_train,y_train)

#  =============== due to not good result i did'nt keep smote =========

def build_model(hp):
    model = Sequential()
    nodes = hp.Int('nodes',min_value= 20,max_value=128,step= 8)
    optimizer = hp.Choice('optimizer',values=['Adam','sgd'])
    activation = hp.Choice('activation',values=['tanh','sigmoid','relu'])
    num_layers = hp.Int('layers',min_value=2,max_value=5)
    for i in range (num_layers):
        if i == 1:
            model.add(Dense(units=nodes,activation=activation,input_shape=21))
        else:
            model.add(Dense(units=nodes,activation=activation))
        model.add(Dense(1,activation='sigmoid'))          
        model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['f1_score'])
    return model
tuner = kt.RandomSearch(build_model,objective='val_f1_score',max_trials=5)
tuner.search(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
print(tuner.get_best_hyperparameters()[0].values)
model = tuner.get_best_models(num_models=1)[0]
print(model.summary())
history = model.fit(x_train,y_train,epochs=50,validation_split = .24)
probabilities = model.predict(x_test)
threshold = 0.45
predictions = (probabilities > threshold).astype(int)

print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("Accuracy Score:", accuracy_score(y_test, predictions))
print('precion : ',precision_score(y_test, predictions))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["f1_score"], label="Train Accuracy")
plt.plot(history.history["val_f1_score"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()

#saving the model

# model.save('trained_nural_network.keras')
# print('model saved')