'''
Program flow: 
1. Design a model  by providing input size, output size and forward pass
2. Construct a loss and an optimizer. 
3. Implement a training loop where:
    forward pass: computes the prediction and loss. 
    backward pass: computes the gradient. 
    update our weights
'''
from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Preparing the dataset using breast cancer data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#conversion to torch tensors
X_train_scaled = torch.from_numpy(X_train_scaled.astype(np.float32))
X_test_scaled = torch.from_numpy(X_test_scaled.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#reshaping our target tensors
y_train_= y_train.view(y_train.shape[0], 1) 
y_test_= y_test.view(y_test.shape[0], 1) 
##Designing the model
# f = wx_b followed by sigmoid function
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)
##Set up loss and optimizer 
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
###Implement training loop
epochs = 100
for epoch in range(epochs):
    #implementing forward pass
    y_predicted = model(X_train_scaled)
    loss = criterion(y_predicted, y_train)
    #backward pass
    loss.backward()
    #updating the weights
    optimizer.step()
    #zero the gradients
    optimizer.zero_grad()
    if (epoch + 1) %10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test_scaled)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'Accuracy = {acc:.4f}')