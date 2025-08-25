# mnist_cnn_torch.py
import time
from time import perf_counter
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------- config --------------------
TRAIN_SIZE     = 10_000         # subset of training set (cap)
epochs         = 20
learning_rate  = 1e-3
batch_size     = 64
num_workers    = 0              # bump if you want prefetch (e.g., 2-4)
print_every    = 50

def auto_device():
    if th.cuda.is_available(): return th.device("cuda")
    if th.backends.mps.is_available() and th.backends.mps.is_built(): return th.device("mps")
    return th.device("cpu")

device = auto_device()
print("Using device:", device)

# Optional: make GPU results closer to pure FP32 math (helps exactness tests)
th.backends.cuda.matmul.allow_tf32 = False
th.backends.cudnn.allow_tf32 = False

# -------------------- data ----------------------
class NpzMNIST(Dataset):
    def __init__(self, images, labels):
        # images: (N, 28, 28) np.float32 [0,1] or uint8
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0
        self.x = th.from_numpy(images)               # (N, 28, 28) float32
        self.y = th.from_numpy(labels.astype(np.int64))  # (N,) int64

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Return CHW for conv: (1, 28, 28)
        return self.x[idx], self.y[idx]

data = np.load("mnist_data.npz")
train_images, train_labels = data["train_data"], data["train_labels"]
test_images,  test_labels  = data["test_data"],  data["test_labels"]

# Optional: limit train set size for faster runs
train_images = train_images[:TRAIN_SIZE]
train_labels = train_labels[:TRAIN_SIZE]

train_ds = NpzMNIST(train_images, train_labels)
test_ds  = NpzMNIST(test_images,  test_labels)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=(device.type == "cuda"),
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=(device.type == "cuda"),
)

# -------------------- model ---------------------
class CNN(nn.Module):
    """
    (N,1,28,28) -> logits (N,10)
    28x28 -> 14x14 -> 7x7 via stride-2 convs
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1,   8,  kernel_size=3, stride=1, padding=1)  # 28x28
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8,  16,  kernel_size=3, stride=2, padding=1)  # 14x14
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 32,  kernel_size=3, stride=2, padding=1)  # 7x7
        self.relu3 = nn.ReLU(inplace=True)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (N,1,28,28)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.flatten(1)              # (N, 32*7*7)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # or Adam

# -------------------- train / eval ---------------------
def train_one_epoch(epoch: int) -> float:
    model.train()
    epoch_t0 = perf_counter()
    running_loss = 0.0

    for it, (xb, yb) in enumerate(train_loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        it_t0 = perf_counter()
        logits = model(xb)                # (B,10)
        loss = criterion(logits, yb)      # scalar
        loss.backward()
        optimizer.step()
        it_ms = (perf_counter() - it_t0) * 1e3

        running_loss += loss.item()
        if it % print_every == 0 or it == 1:
            avg = running_loss / (print_every if it >= print_every else it)
            print(f"Epoch {epoch+1:02d} | Iter {it:04d}/{len(train_loader)} | "
                  f"Loss {avg:.4f} | {it_ms:.1f} ms")
            running_loss = 0.0

    return perf_counter() - epoch_t0  # seconds

@th.no_grad()
def evaluate() -> float:
    model.eval()
    total_correct, total_seen = 0, 0
    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        total_correct += (pred == yb).sum().item()
        total_seen += yb.numel()
    acc = total_correct / max(1, total_seen)
    print(f"Test Accuracy: {acc*100:.2f}%")
    return acc

# -------------------- main ----------------------
if __name__ == "__main__":
    epoch_times = []
    for epoch in range(epochs):
        t = train_one_epoch(epoch)
        epoch_times.append(t)
        avg_t = sum(epoch_times) / len(epoch_times)
        print(f"Epoch {epoch+1} training time: {t:.2f}s | Average: {avg_t:.2f}s")
        evaluate()
    print("Finished Training")
