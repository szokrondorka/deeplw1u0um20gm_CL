import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # No pictures displayed 

import data
import models

# Device configuration
if torch.cuda.is_available() is False:
    raise Exception("GPU device not found, runtime environment should be set to GPU")
device = torch.cuda.current_device()
print(f"Using GPU device: {torch.cuda.get_device_name(torch.cuda.current_device())}--------------------------------------------------")

# Hyper parameters
datadir = "/home/szokron/deeplw1u0um20gm_CL/"
batch_size = 100
num_tasks = 5
num_cycles = 1
num_classes = 10
learning_rate = 0.001
num_epochs = 20

# MNIST dataset
# Data loader
# data.py

data = data.Data_MNIST(datadir, batch_size, num_tasks, num_cycles, num_classes)

print(data.train_loaders[0])

# Convolutional neural network (two convolutional layers)
# models.py

model = models.ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses=[]
train_accu=[]
eval_losses=[]
eval_accu=[]

# Train the model
for i in range(0,num_tasks):
  total_step = len(data.train_loaders[i])
  for epoch in range(num_epochs):
      running_loss = 0
      correct = 0
      total = 0
      for j, (images, labels) in enumerate(data.train_loaders[i]):
          images = images.to(device)
          labels = labels.to(device)
          
          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)
          
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          running_loss += loss.item()
          
          _, predicted = outputs.max(1)
          total += labels.size(0)
          correct += predicted.eq(labels).sum().item()
          
          if (j+1) % 5 == 0:
              print ('Task [{}] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(i+1, epoch+1, num_epochs, j+1, total_step, loss.item()))
      train_loss=running_loss/len(data.train_loaders[i])
      accu=correct/total
   
      train_accu.append(accu)
      train_losses.append(train_loss)
      print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

                    
      # Test the model
      model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
      with torch.no_grad():
          running_loss = 0
          correct = 0
          total = 0
          for images, labels in data.test_loaders[i]:
              images = images.to(device)
              labels = labels.to(device)
              outputs = model(images)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
      
          print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
          
      test_loss=running_loss/len(data.test_loaders[i])
      accu=correct/total
 
      eval_losses.append(test_loss)
      eval_accu.append(accu)
       
      print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 
  
plt.plot(train_accu,'-o')
plt.plot(eval_accu,'-o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Accuracy')
 
plt.savefig(datadir + 'acc.png')

plt.plot(train_losses,'-o')
plt.plot(eval_losses,'-o')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Losses')
 
plt.savefig(datadir + 'loss.png')

    

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')
