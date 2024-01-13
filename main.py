import torch
import torchvision
import torchvision.transforms as transforms

# zmiana obrazow PILI range([0,1]) na Tensor range([-1, 1])
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# pokazanie paru obrazow treningowych
import matplotlib.pyplot as plt
import numpy as np

# funkcja img_show
def imshow(img):
    img = img / 2 + 0.5     # odnormalizowanie
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# pobranie losowego obrazka trenningowego
dataiter = iter(trainloader)
images, labels = next(dataiter)

# pokaz obrazy
print('\nShowing an example set of images')
imshow(torchvision.utils.make_grid(images))

# wygeneruj
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F

# definicja sieci neuronowej
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 50, 5)
        self.fc1 = nn.Linear(50 * 5 * 5, 120)
        # self.dropout1 = nn.Dropout(0)  # dropout probability of 0.5
        # self.bn1 = nn.BatchNorm1d(120)  # Batch normalization layer after the first layer
        self.fc2 = nn.Linear(120, 84)
        # self.dropout2 = nn.Dropout(0)  # dropout probability of 0.5
        # self.bn2 = nn.BatchNorm1d(84)  # Batch normalization layer after the first layer
        self.fc3 = nn.Linear(84, 10)

        # torch.nn.functional.relu(input, inplace=False)
        # torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)
        # torch.nn.functional.sigmoid(input)
        # torch.nn.functional.tanh(input)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        # x = self.bn1(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        # x = self.bn2(x)
        x = self.fc3(x)
        return x

net = Net()

# definicja funkcji strat
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# trenowanie sieci
print('\nTraining in progress...')
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()

    train_losses.append(running_loss / (i + 1))
    accuracy = 100 * correct / total
    train_accuracies.append(accuracy)
    print(f'[{epoch + 1}] loss: {running_loss / (i + 1):.3f} acc: {accuracy:.0f}%')

    # Test the model on the test set
    net.eval()  # Set the model to evaluation mode
    test_correct = 0.0
    test_total = 0.0

    with torch.no_grad():
        for data in testloader:
            test_inputs, test_labels = data
            test_outputs = net(test_inputs)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    test_accuracies.append(test_accuracy)
    print(f'Finished testing on the test set. Test accuracy: {test_accuracy:.2f}%')

# Plot accuracy chart
epoch_range = range(1, len(train_accuracies) + 1)
plt.plot(epoch_range, train_accuracies, label='Train')
plt.plot(epoch_range, test_accuracies, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()

print('Finished Training')
print('\nSaving the training model...')

# zapisanie modelu treningowego
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# wyswietlenie obrazu z testu
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('\nGroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# wczytanie ponowne modelu
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\nAccuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
print('-----------------------------------')
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# wykres straty podczas treningu
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# wykres slupkowy dokladnosci roznawania dla kazdej z klas
class_names = list(correct_pred.keys())
class_accuracy = [100 * correct_pred[classname] / total_pred[classname] for classname in class_names]

plt.figure(figsize=(10, 5))
plt.bar(class_names, class_accuracy)
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for Each Class')
plt.ylim([0, 100])
plt.show()
