print('libraries are loading....')
import torch
from torch.nn import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tf import preprocess_churn_data,drop_extra,x_train,y_train
import pandas as pd 

data = pd.read_csv('costumer-churn-prediction/Data.csv')
data = drop_extra(data)
data = preprocess_churn_data(data)

print(data.info())

class MyModel(Module):
    def __init__(self,input_dim):
        super().__init__()
        self.network = Sequential(
            Linear(input_dim,128),
            ReLU(),
            Linear(128,64),
            ReLU(),
            Linear(64,1),
            Sigmoid()
        )
    def forward(self,x):
        return self.network(x)

input_dim = x_train.shape[1]
model = MyModel(input_dim)
epochs = 20
optimizer = optim.Adam(model.parameters(),lr=0.001)
criterian = BCELoss()
model.train()
print('model created')

x_train = torch.tensor(x_train.values,dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(x_train,y_train)
data_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)

print('start training ......')
for _ in range(epochs):
    total_loss  = 0.0
    for x,y in data_loader:
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterian(predictions,y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss = total_loss/len(data_loader)
    print('current loss ===>', avg_loss)



