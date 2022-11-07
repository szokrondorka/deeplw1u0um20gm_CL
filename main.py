import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset

#0-1
train0 = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test0 = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

#2-3
train1 = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test1 = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

#4-5
train2 = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test2 = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

#6-7
train3 = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test3 = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

#8-9
train4 = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test4 = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

#train
idx = (train0.targets==0) | (train0.targets==1)
train0.targets = train0.targets[idx]
train0.data = train0.data[idx]

idx = (train1.targets==2) | (train1.targets==3)
train1.targets = train1.targets[idx]
train1.data = train1.data[idx]

idx = (train2.targets==4) | (train2.targets==5)
train2.targets = train2.targets[idx]
train2.data = train2.data[idx]

idx = (train3.targets==6) | (train3.targets==7)
train3.targets = train3.targets[idx]
train3.data = train3.data[idx]

idx = (train4.targets==8) | (train4.targets==9)
train4.targets = train4.targets[idx]
train4.data = train4.data[idx]

#test
idx = (test0.targets==0) | (test0.targets==1)
test0.targets = test0.targets[idx]
test0.data = test0.data[idx]

idx = (test1.targets==2) | (test1.targets==3)
test1.targets = test1.targets[idx]
test1.data = test1.data[idx]

idx = (test2.targets==4) | (test2.targets==5)
test2.targets = test2.targets[idx]
test2.data = test2.data[idx]

idx = (test3.targets==6) | (test3.targets==7)
test3.targets = test3.targets[idx]
test3.data = test3.data[idx]

idx = (test4.targets==8) | (test4.targets==9)
test4.targets = test4.targets[idx]
test4.data = test4.data[idx]


# Data loader
train_loader0 = torch.utils.data.DataLoader(dataset=train0,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader0 = torch.utils.data.DataLoader(dataset=test0,
                                          batch_size=batch_size, 
                                          shuffle=False)

train_loader1 = torch.utils.data.DataLoader(dataset=train1,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader1 = torch.utils.data.DataLoader(dataset=test1,
                                          batch_size=batch_size, 
                                          shuffle=False)

train_loader2 = torch.utils.data.DataLoader(dataset=train2,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader2 = torch.utils.data.DataLoader(dataset=test2,
                                          batch_size=batch_size, 
                                          shuffle=False)

train_loader3 = torch.utils.data.DataLoader(dataset=train3,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader3 = torch.utils.data.DataLoader(dataset=test3,
                                          batch_size=batch_size, 
                                          shuffle=False)

train_loader4 = torch.utils.data.DataLoader(dataset=train4,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader4 = torch.utils.data.DataLoader(dataset=test4,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = [train_loader0,train_loader1,train_loader2,train_loader3,train_loader4]
test_loader = [test_loader0,test_loader1,test_loader2,test_loader3,test_loader4]

# Train the model
for i in range(0,5):
  total_step = len(train_loader[i])
  for epoch in range(num_epochs):
      for j, (images, labels) in enumerate(train_loader[i]):
          images = images.to(device)
          labels = labels.to(device)
          
          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)
          
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          if (j+1) % 100 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')
