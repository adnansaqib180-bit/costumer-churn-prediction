import pandas as pd
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import keras 
    from keras import Sequential
    from keras.layers import Dense, Normalization
    # from imblearn.over_sampling import SMOTE 
    import keras_tuner as kt
    from keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping(monitor='val_recall', 
                                   patience=5,
                                   mode='max'  )



df = pd.read_csv('costumer-churn-prediction/Data.csv')

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


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42)

class_weight_dict = {0: 0.67, 1: 1.94}
if __name__ == '__main__':
    print(df.head())
    print(df.info())
    print(x.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(df['Churn'].value_counts())



# smote = SMOTE()
# x_train,y_train = smote.fit_resample(x_train,y_train)

#  =============== due to not good result i did'nt keep smote =========

if __name__ == '__main__':
    normalizer = Normalization()
    normalizer.adapt(x_train.values)
    def build_model(hp):
        model = Sequential()
        model.add(normalizer)
        nodes = hp.Int('nodes',min_value= 20,max_value=80,step= 8)
        optimizer = hp.Choice('optimizer',values=['Adam','sgd','adagrad','nadam'])
        activation = hp.Choice('activation',values=['tanh','sigmoid','relu','elu'])
        num_layers = hp.Int('layers',min_value=1,max_value=3)
        for i in range (num_layers):
            if i == 0:
                model.add(Dense(units=nodes,activation=activation,input_shape=(16,)))
            else:
                model.add(Dense(units=nodes,activation=activation))
        model.add(Dense(1,activation='sigmoid'))          
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=[keras.metrics.Recall(name='recall')])
        return model



    tuner = kt.RandomSearch(build_model,objective='val_recall',max_trials=10)
    tuner.search(x_train,
                 y_train,epochs=10,
                 validation_data=(x_test,y_test),
                 callbacks=[early_stopping],
                 class_weight=class_weight_dict
                 )


    print(tuner.get_best_hyperparameters()[0].values)
    model = tuner.get_best_models(num_models=1)[0]
    print(model.summary())


    history = model.fit(x_train,y_train,epochs=100,validation_split = .20)



    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(history.history["recall"], label="Train Recall")
    plt.plot(history.history["val_recall"], label="Validation Recall")
    plt.title("Model Recall")
    plt.ylabel("Recall")
    plt.xlabel("Epoch")
    plt.legend()


    plt.tight_layout()
    plt.show()

    #saving the model

    model.save('final_ann.keras')
    print('model saved')