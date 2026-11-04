print('libraries are loading....')
import torch
from torch.nn import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tf import preprocess_churn_data,drop_extra,x_train,y_train,x_test,y_test
import pandas as pd 

data = pd.read_csv('costumer-churn-prediction/Data.csv')
data = drop_extra(data)
data = preprocess_churn_data(data)

print(data.info())

x_train = torch.tensor(x_train.values,dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
x_test = torch.tensor(x_test.values,dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
train_dataset = TensorDataset(x_train,y_train)
test_dataset = TensorDataset(x_test,y_test)
train_loader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=32)

class ANN(Module):
    def __init__(self,input_dim,output_dim,nodes,num_layers,activation):
        super().__init__()
        layers = []
        activation_mappig = {
            'Tanh': Tanh(),
            'Sigmoid': Sigmoid(),
            'ReLU': ReLU(),
            'ELU': ELU()
        }
        for i in range(num_layers):
            layers.append(Linear(input_dim,nodes))
            layers.append(activation_mappig[activation])
            input_dim = nodes

        layers.append(Linear(nodes,output_dim))
        layers.append(Sigmoid())
        self.ann = Sequential(*layers)
    def forward(self,x):
        return self.ann(x)
    

def objective(trail):
    nodes = trail.suggest_int('nodes', 20,80,step= 8)
    suggested_optimizer = trail.suggest_categorical('optimizer',['Adam','sgd','adagrad','nadam'])
    activation = trail.suggest_categorical('activation', ['Tanh', 'Sigmoid', 'ReLU', 'ELU'])
    num_layers = trail.suggest_int('layers',1,3)
    learning_rate = trail.suggest_int('lr',1e-5,1e-1)
    input_dim = x_train.shape[1]
    output_dim = 1
    model = ANN(input_dim,output_dim,nodes,num_layers,activation)
    
    epochs = 20
    optimizers = {
        'Adam':optim.Adam(model.parameters(),lr=learning_rate),
        'sgd':optim.SGD(model.parameters(),lr=learning_rate),
        'adagrad':optim.Adagrad(model.parameters(),lr=learning_rate),
        'nadam':optim.NAdam(model.parameters(),lr=learning_rate)
    }
    optimizer = optimizers[suggested_optimizer]
    criterian = BCELoss()

    for _ in range(epochs):
        for x,y in train_loader:
            optimizer.zero_grad()
            predictions = model(x)
            loss = criterian(predictions,y)
            loss.backward()
            optimizer.step()
            
    
    # eval
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            predictions = model(x_batch)
            predicted_classes = (predictions >= 0.5).float()
            correct_predictions += (predicted_classes == y_batch).sum().item()
            total_samples += y_batch.size(0)
    test_accuracy = (correct_predictions / total_samples) * 100
    return test_accuracy


     
import optuna 
study =  optuna . create_study(direction='minimize')
study.optimize(objective,n_trials=5)

print(study.best_trial)
