from keras.models import load_model
from train import x_test, y_test 
from sklearn.metrics import (confusion_matrix, 
                             f1_score, accuracy_score,
                               precision_score,recall_score)

model =load_model('churn_ann.keras')

predictions = model.predict(x_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
print("Accuracy Score:", accuracy_score(y_test, predictions))
print('Precision : ',precision_score(y_test, predictions))
print('Recall : ',recall_score(y_test, predictions))
