import time

import numpy as np
import znet
import znet.nn as nn

import znet.optim as optim
from znet.autograd import Tensor
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

TRAIN_SIZE = 10000
epochs = 50
learning_rate = 1e-3
batch_size = 64
data_dir = "data"

data = np.load("mnist_data.npz")
train_data = data["train_data"]
train_labels = data["train_labels"]
test_data = data["test_data"]
test_labels = data["test_labels"]



print("Train Data Shape:", train_data.shape)
print("Train Data Type:", train_data.dtype)



print("Test Data Shape:", test_data.shape)
print("Test Data Type:", test_data.dtype)



iters_per_epoch = TRAIN_SIZE // batch_size
print("Iters per epoch:", iters_per_epoch)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = x.reshape((batch_size, 28 * 28))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# model = MLP(in_features=784, hidden_features=256, num_classes=10).to("cuda")
model = MLP(in_features=784, hidden_features=256, num_classes=10)
# model = torch.compile(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# Training the model
def train(model, criterion, optimizer, epoch):
    # model.train()
    running_loss = 0.0

    for i in range(iters_per_epoch):
        
        optimizer.zero_grad()
        # data = train_data[i * batch_size : (i + 1) * batch_size].to("cuda")
        # target = train_labels[i * batch_size : (i + 1) * batch_size].to("cuda")
        data = Tensor(train_data[i * batch_size : (i + 1) * batch_size], requires_grad=True)
        target = Tensor(train_labels[i * batch_size : (i + 1) * batch_size], dtype=np.int32)

        start = time.time()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end = time.time()
        running_loss += loss.item()
        if i % 10 == 0 :
            # print(f"Epoch: {epoch+1}, Iter: {i+1}, Loss: {loss.item():.4f}")
            # print(f"Iteration Time: {(end - start) * 1e3:.4f} sec")
            running_loss = 0.0


# Evaluation function to report average batch accuracy using the loaded test data
def evaluate(model, test_data, test_labels):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.eval()

    total_batch_accuracy = np.zeros(1)
    num_batches = 0

    # with znet.no_grad():
    for i in range(len(test_data) // batch_size):
        data = Tensor(test_data[i * batch_size : (i + 1) * batch_size])
        target = Tensor(test_labels[i * batch_size : (i + 1) * batch_size], dtype=np.int32)
        outputs = model(data)
        # logits -> NumPy (works for CUDA/MPS/CPU)
        logits_np = outputs.data.detach().cpu().numpy()
        predicted = np.argmax(logits_np, axis=1)

        # targets -> NumPy (works for CUDA/MPS/CPU)
        target_np = target.data.detach().cpu().numpy()

        correct_batch = (predicted == target_np).sum().item()
        total_batch = target_np.shape[0]
        if total_batch != 0:  # Check to avoid division by zero
            batch_accuracy = correct_batch / total_batch
            total_batch_accuracy += batch_accuracy
            num_batches += 1

    avg_batch_accuracy = total_batch_accuracy / num_batches
    print(f"Average Batch Accuracy: {avg_batch_accuracy.item() * 100:.2f}%")


# Main
if __name__ == "__main__":
    for epoch in range(epochs):
        train(model, criterion, optimizer, epoch)
        evaluate(model, test_data, test_labels)

    print("Finished Training")