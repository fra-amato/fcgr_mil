from torch import optim
from torch import nn
from torch.utils.data import DataLoader,random_split
from model import *
import data
import math

#parameters
kmers_len = 3
learning_rate = 0.001

#Create network model 
print('creating network...')
network = MilModel(kmers_len)

#Create datasets
print('creating dataset...')
dataset = data.FcgrBags(kmers_len)
train_size = math.ceil((70 * dataset.sample_number)/100)
validation_size = math.floor((10 * dataset.sample_number)/100)
test_size = math.floor((20 * dataset.sample_number)/100)
train,validation,test = random_split(dataset, [train_size, validation_size,test_size])
train_loader = DataLoader(train,batch_size=100)
validation_loader = DataLoader(validation)
test_loader = DataLoader(test)

#Loss
print('creating loss...')
binary_cross_entropy = nn.CrossEntropyLoss()

#optimizer
print('creating optimizer...')
optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=0.9)

#training loop
print('training loop...')
for epoch in range(50):
    print('-------- Start Epoch {} --------'.format(epoch + 1))
    for i,(bags,labels) in enumerate(train_loader):
        #Zero gradient parameters
        optimizer.zero_grad()

        #forward
        class_pred = network.forward(bags.unsqueeze(1))
        #loss
        loss = binary_cross_entropy(class_pred,labels)
        #Backward
        loss.backward()
        #update weights and bias
        optimizer.step()
        #if (i+1)%100 == 0:
        #   print( 'Batch {}/1000'.format(i+1))
    print('Epoch: {} - loss:{}'.format(epoch + 1, loss.item()))