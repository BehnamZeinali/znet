# mnist_cnn_torch_fixed.py
import time
from time import perf_counter
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------- config --------------------
TRAIN_SIZE     = 60_000
epochs         = 20
learning_rate  = 1e-3
batch_size     = 64
num_workers    = 0
print_every    = 50

def auto_device():
    if th.cuda.is_available(): return th.device("cuda")
    if th.backends.mps.is_available() and th.backends.mps.is_built(): return th.device("mps")
    return th.device("cpu")

device = auto_device()
print("Using device:", device)

# Optional: deterministic-ish matmul (closer to pure FP32)
th.backends.cuda.matmul.allow_tf32 = False
th.backends.cudnn.allow_tf32 = False

# -------------------- data ----------------------
class NpzMNIST(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # images: (N, 28, 28) [float32 in 0..1] or uint8 in 0..255
        # if images.dtype != np.float32:
        #     images = images.astype(np.float32) / 255.0
        self.x = th.from_numpy(images)                      # (N, 28, 28) float32
        self.y = th.from_numpy(labels.astype(np.int64))     # (N,) int64

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # >>> FIX: return CHW for convs (1, 28, 28)
        x = self.x[idx].unsqueeze(0)                        # (1, 28, 28)
        y = self.y[idx]
        return x, y

data = np.load("mnist_data.npz")
train_images, train_labels = data["train_data"], data["train_labels"]
test_images,  test_labels  = data["test_data"],  data["test_labels"]

# Optional: limit train set size
train_images = train_images[:TRAIN_SIZE]
train_labels = train_labels[:TRAIN_SIZE]

train_ds = NpzMNIST(train_images, train_labels)
test_ds  = NpzMNIST(test_images,  test_labels)

pin = (device.type == "cuda")
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)

# -------------------- model ---------------------
class CNN(nn.Module):
    """(N,1,28,28) -> logits (N,10); 28->14->7 via stride-2 convs"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1,   8, kernel_size=3, stride=1, padding=1)  # 28x28
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8,  16, kernel_size=3, stride=2, padding=1)  # 14x14
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 7x7
        self.relu3 = nn.ReLU(inplace=True)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.flatten(1)          # (N, 32*7*7)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # try Adam(1e-3) to sanity-check

# -------------------- train / eval ---------------------
def train_one_epoch(epoch: int) -> float:
    model.train()
    epoch_t0 = perf_counter()
    running_loss = 0.0

    for it, (xb, yb) in enumerate(train_loader, 1):
        xb = xb.squeeze(1).to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)                # (B,10)
        loss = criterion(logits, yb)      # scalar
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if it % print_every == 0 or it == 1:
            avg = running_loss / (print_every if it >= print_every else it)
            print(f"Epoch {epoch+1:02d} | Iter {it:04d}/{len(train_loader)} | Loss {avg:.4f}")
            running_loss = 0.0

    return perf_counter() - epoch_t0

@th.no_grad()
def evaluate() -> float:
    model.eval()
    total_correct, total_seen = 0, 0
    for xb, yb in test_loader:
        xb = xb.squeeze(1).to(device, non_blocking=True)
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
