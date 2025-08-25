import time
import numpy as np
import torch as th
from contextlib import nullcontext
from time import perf_counter
import znet
import torch.nn as nn
import torch.optim as optim


# -------------------- config --------------------
TRAIN_SIZE    = 10000
epochs        = 20
learning_rate = 1e-3
batch_size    = 64  # feel free to set back to 4; 64 is just faster

# -------------------- device helper --------------------
def _auto_device():
    if th.cuda.is_available():
        return "cuda"
    if th.backends.mps.is_available() and th.backends.mps.is_built():
        return "mps"
    return "cpu"

device = _auto_device()
print("Using device:", device)

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

# cap TRAIN_SIZE to dataset size
TRAIN_SIZE = min(TRAIN_SIZE, len(train_data))
iters_per_epoch = TRAIN_SIZE // batch_size

print("Train Data Shape:", train_data.shape, "dtype:", train_data.dtype)
print("Test  Data Shape:", test_data.shape,  "dtype:", test_data.dtype)
print("Iters per epoch:", iters_per_epoch)

# -------------------- model ---------------------
class CNN(nn.Module):
    """
    Simple MNIST ConvNet (N,1,28,28) -> logits (N,10)
    Downsampling via stride-2 convs: 28x28 -> 14x14 -> 7x7
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=8,  kernel_size=3, stride=1, padding=1)  # 28x28
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8,  out_channels=16, kernel_size=3, stride=2, padding=1)  # 14x14
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)  # 7x7
        self.relu3 = nn.ReLU()
        # 32 * 7 * 7 = 1568
        self.fc1   = nn.Linear(32 * 7 * 7, 64, bias=True)
        self.relu4 = nn.ReLU()
        self.fc2   = nn.Linear(64, num_classes, bias=True)

    def forward(self, x):
        # Accept (B,28,28) or (B,1,28,28); convert to NCHW
        if len(x.shape) == 3:
            x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))  # (B,1,28,28)
        x = self.conv1(x); x = self.relu1(x)
        x = self.conv2(x); x = self.relu2(x)
        x = self.conv3(x); x = self.relu3(x)
        x = x.reshape((x.shape[0], -1))         # (B, 1568)
        x = self.fc1(x);  x = self.relu4(x)
        x = self.fc2(x)
        return x

model = CNN(num_classes=10)
# move model params to device (guarded in case your Module doesn't implement .to)
if hasattr(model, "to"): model.to(device)

criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# -------------------- train ---------------------
def train(model, criterion, optimizer, epoch):
    # set train mode if available
    if hasattr(model, "train"): 
        model.train()

    running_loss = 0.0
    idx = np.random.permutation(TRAIN_SIZE)
    epoch_t0 = perf_counter()

    for i in range(iters_per_epoch):
        batch_idx = idx[i * batch_size : (i + 1) * batch_size]

        # --- Torch tensors from NumPy batches (on the right device) ---
        xb = th.as_tensor(train_data[batch_idx]).to(device=device, dtype=th.float32, non_blocking=True)
        # If images are (B, H, W), make them NCHW for convs: (B, 1, H, W)
        if xb.ndim == 3:
            xb = xb.unsqueeze(1)
        xb = xb.contiguous()

        yb = th.as_tensor(train_labels[batch_idx]).to(device=device, dtype=th.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        

        outputs = model(xb)                 # (B, 10)
        loss    = criterion(outputs, yb)    # scalar
        loss.backward()
        optimizer.step()

        end = time.time()
        running_loss += loss.item()
        if i % 50 == 0:
            # print(f"Epoch {epoch+1:02d} | Iter {i+1:04d}/{iters_per_epoch} | "
            #       f"Loss {loss.item():.4f} | {((end-start)*1e3):.1f} ms")
            running_loss = 0.0

    return perf_counter() - epoch_t0

# -------------------- eval ----------------------
def evaluate(model, test_data, test_labels):
    # eval mode if available
    if hasattr(model, "eval"): 
        model.eval()

    total_batch_accuracy = 0.0
    counted_batches = 0
    num_batches = len(test_data) // batch_size

    with th.no_grad():
        for i in range(num_batches):
            # NumPy -> Torch on the right device/dtypes
            xb = th.as_tensor(test_data[i * batch_size : (i + 1) * batch_size])\
                   .to(device=device, dtype=th.float32, non_blocking=True)
            if xb.ndim == 3:  # (B, H, W) -> (B, 1, H, W)
                xb = xb.unsqueeze(1)
            xb = xb.contiguous()

            yb = th.as_tensor(test_labels[i * batch_size : (i + 1) * batch_size])\
                   .to(device=device, dtype=th.long, non_blocking=True)

            # Forward & accuracy
            logits = model(xb)                # (B, 10)
            pred = logits.argmax(dim=1)      # (B,)
            correct_batch = (pred == yb).sum().item()
            total_batch = yb.numel()

            if total_batch > 0:
                total_batch_accuracy += correct_batch / total_batch
                counted_batches += 1

    avg_batch_accuracy = (total_batch_accuracy / counted_batches) if counted_batches > 0 else 0.0
    print(f"Average Batch Accuracy: {avg_batch_accuracy * 100:.2f}%")
    return avg_batch_accuracy

# -------------------- main ----------------------
if __name__ == "__main__":
    epoch_times = []
    for epoch in range(epochs):
        t = train(model, criterion, optimizer, epoch)   # seconds (training only)
        epoch_times.append(t)
        avg_t = sum(epoch_times) / len(epoch_times)
        print(f"Epoch {epoch+1} training time: {t:.2f}s | Average so far: {avg_t:.2f}s")

        evaluate(model, test_data, test_labels)  # not included in training time
    print("Finished Training")
