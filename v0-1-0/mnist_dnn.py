import time

import numpy as np
import znet
import znet.nn as nn

import znet.optim as optim
from znet.autograd import Tensor
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms




# -------------------- config --------------------
TRAIN_SIZE   = 60000
epochs       = 20
learning_rate = 1e-3
batch_size   = 4

# -------------------- data ----------------------
data = np.load("mnist_data.npz")
train_data  = data["train_data"]   # expect (N, 28, 28) uint8 or float
train_labels = data["train_labels"]  # (N,)
test_data   = data["test_data"]
test_labels = data["test_labels"]

# normalize to float32 in [0,1]
if train_data.dtype != np.float32:
    train_data = train_data.astype(np.float32) / 255.0
if test_data.dtype != np.float32:
    test_data = test_data.astype(np.float32) / 255.0

print("Train Data Shape:", train_data.shape, "dtype:", train_data.dtype)
print("Test  Data Shape:", test_data.shape,  "dtype:", test_data.dtype)

iters_per_epoch = TRAIN_SIZE // batch_size
print("Iters per epoch:", iters_per_epoch)

# -------------------- model ---------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features, bias=True)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_features, num_classes, bias=True)

    def forward(self, x: Tensor):
        # x: (B, 28, 28) -> (B, 784); use dynamic batch dim, keep engine-compatible reshape
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(in_features=784, hidden_features=256, num_classes=10)
criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

# -------------------- train ---------------------
def train(model, criterion, optimizer, epoch):
    running_loss = 0.0
    # simple shuffling each epoch
    idx = np.random.permutation(TRAIN_SIZE)

    for i in range(iters_per_epoch):
        batch_idx = idx[i * batch_size : (i + 1) * batch_size]

        # Inputs do NOT need requires_grad; params already require_grad=True
        data_batch = Tensor(train_data[batch_idx], requires_grad=False)
        target_batch = Tensor(train_labels[batch_idx].astype(np.int64), requires_grad=False)

        optimizer.zero_grad()

        start = time.time()
        outputs = model(data_batch)                 # (B, 10)
        loss     = criterion(outputs, target_batch) # scalar
        loss.backward()
        optimizer.step()
        end = time.time()

        running_loss += loss.item()
        if i % 1000 == 0:
            print(f"Epoch: {epoch+1}, Iter: {i+1}, Loss: {loss.item():.4f}, "
                  f"Iter Time: {(end - start)*1e3:.2f} ms")
            running_loss = 0.0

# -------------------- eval ----------------------
def evaluate(model, test_data, test_labels):
    total_correct = 0
    total_seen = 0
    num_batches = len(test_data) // batch_size

    for i in range(num_batches):
        data_batch = Tensor(test_data[i * batch_size : (i + 1) * batch_size], requires_grad=False)
        target_batch = Tensor(test_labels[i * batch_size : (i + 1) * batch_size].astype(np.int64), requires_grad=False)

        logits = model(data_batch)                      # (B, 10)
        preds  = np.argmax(logits.data, axis=1)         # ok to read .data for eval
        total_correct += (preds == target_batch.data).sum()
        total_seen    += target_batch.data.shape[0]

    acc = (total_correct / max(1, total_seen)) * 100.0
    print(f"Average Batch Accuracy: {acc:.2f}%")

# -------------------- main ----------------------
if __name__ == "__main__":
    for epoch in range(epochs):
        train(model, criterion, optimizer, epoch)
        evaluate(model, test_data, test_labels)
    print("Finished Training")
