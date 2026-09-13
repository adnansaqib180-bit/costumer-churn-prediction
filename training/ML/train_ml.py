import numpy as np,pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('Data.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df['Churn'].value_counts())
print (df.shape)
print(df.columns)

df = df.drop(columns=['gender','customerID'])
df.head()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(np.mean(df['TotalCharges']))
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

x['SeniorCitizen'] = x['SeniorCitizen'].map({0:'No',1:'Yes'})
to_Label_encode = []
to_onhot_encode = []
to_scale = numeric_columns
for col in x.columns:
    if x[col].nunique() == 2:
        to_Label_encode.append(col)
    elif x[col].nunique() == 3 or x[col].nunique() == 4:
        to_onhot_encode.append(col)
print(to_Label_encode,to_onhot_encode)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from xgboost import XGBClassifier
Preprocessor =  ColumnTransformer([
    ('scaler', StandardScaler(), to_scale),
    ('hot_encoder', OneHotEncoder(handle_unknown='ignore'), to_onhot_encode),
    ('binary_encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), to_Label_encode)
])
models = {
    'LogisticRegression':LogisticRegression(class_weight='balanced',max_iter=6000),
    'xgboost':XGBClassifier(),
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced')
}
param_grids = {
    'LogisticRegression': {
        'solver':  ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag'],
        'C': [ 0.1,0.01,1,30],
        'penalty': ['l1', 'l2']
    },
    'xgboost': {
        'n_estimators': [100, 200, 800],
        'max_depth': [3, 5, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],    
    },
    'DecisionTree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split':[2,5,10],
        'min_samples_leaf': [1, 2, 4]
    }
}

best_models = {}
for model_name in models:
    print(f'filhal ==> {model_name } <== trian ho rha he ...........')
    my_pipeline = Pipeline([
        ('preprocessor',Preprocessor),
        ('model',models[model_name])
    ])
    current_grid = {}
    for prams, val in param_grids[model_name].items():
        current_grid[f'model__{prams}'] = val
    search = RandomizedSearchCV(
        estimator=my_pipeline,
        param_distributions=current_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    best_models[model_name] = search.best_estimator_
    print(f"Optimal Parameters for {model_name}: {search.best_params_}")
    print(f"Highest CV Score: {search.best_score_:.4f}\n")


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix

print("======= FINAL TEST SCORES =======")

for model_name, trained_model in best_models.items():

    prediction = trained_model.predict(x_test)

    print(f"\n{model_name}")
    print("-" * 30)
    print("confusion_matrix :", confusion_matrix(y_test, prediction))
    print("Accuracy :", accuracy_score(y_test, prediction))
    print("Precision:", precision_score(y_test, prediction))
    print("Recall   :", recall_score(y_test, prediction))
    print("F1 Score :", f1_score(y_test, prediction))


import joblib

final_model = best_models['LogisticRegression']

joblib.dump(final_model,'trained_model.pkl')