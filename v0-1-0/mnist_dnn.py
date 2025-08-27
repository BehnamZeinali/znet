import time
import numpy as np
import mlx.core as mx

import znet
import znet.nn as nn
import znet.optim as optim
from znet.autograd.tensor import Tensor

# -------------------- config --------------------
TRAIN_SIZE    = 60000
epochs        = 20
learning_rate = 1e-3
batch_size    = 64

# -------------------- data ----------------------
data = np.load("mnist_data.npz")
train_data   = data["train_data"]     # (N, 28, 28)
train_labels = data["train_labels"]   # (N,)
test_data    = data["test_data"]
test_labels  = data["test_labels"]

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
        # x: (B, 28, 28) -> (B, 784)
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
    idx = np.random.permutation(TRAIN_SIZE)

    epoch_start = time.time()
    for i in range(iters_per_epoch):
        batch_idx = idx[i * batch_size : (i + 1) * batch_size]

        # Inputs do NOT need requires_grad; params already require_grad=True
        data_batch = Tensor(train_data[batch_idx], requires_grad=False, dtype=mx.float32)
        target_batch = Tensor(train_labels[batch_idx].astype(np.int64), requires_grad=False)

        optimizer.zero_grad()

        iter_start = time.time()
        outputs = model(data_batch)                 # (B, 10)
        loss     = criterion(outputs, target_batch) # scalar
        loss.backward()
        optimizer.step()
        iter_end = time.time()

        running_loss += loss.item()
        # if i % 10 == 0:
        #     print(f"Epoch: {epoch+1}, Iter: {i+1}, Loss: {loss.item():.4f}, "
        #           f"Iter Time: {(iter_end - iter_start)*1e3:.2f} ms")
        #     running_loss = 0.0

    epoch_end = time.time()
    epoch_ms = (epoch_end - epoch_start) * 1e3
    avg_iter_ms = epoch_ms / max(1, iters_per_epoch)
    print(f"Epoch {epoch+1} time: {epoch_ms:.2f} ms  |  Avg iter: {avg_iter_ms:.2f} ms")
    return epoch_ms

# -------------------- eval ----------------------
def evaluate(model, test_data, test_labels):
    total_correct = 0
    total_seen = 0
    num_batches = len(test_data) // batch_size

    for i in range(num_batches):
        data_batch = Tensor(test_data[i * batch_size : (i + 1) * batch_size], requires_grad=False, dtype=mx.float32)
        target_np = test_labels[i * batch_size : (i + 1) * batch_size].astype(np.int64)

        logits = model(data_batch)                              # (B, 10)
        preds  = np.array(mx.argmax(logits.data, axis=1))       # to NumPy for comparison
        total_correct += (preds == target_np).sum()
        total_seen    += target_np.shape[0]

    acc = (total_correct / max(1, total_seen)) * 100.0
    print(f"Average Batch Accuracy: {acc:.2f}%")

# -------------------- main ----------------------
if __name__ == "__main__":
    epoch_times = []
    for epoch in range(epochs):
        t_ms = train(model, criterion, optimizer, epoch)
        epoch_times.append(t_ms)
        evaluate(model, test_data, test_labels)

    avg_epoch_ms = float(np.mean(epoch_times)) if epoch_times else 0.0
    print(f"Average epoch time over {len(epoch_times)} epochs: {avg_epoch_ms:.2f} ms")
    print("Finished Training")
