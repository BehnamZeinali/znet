import numpy as np
from znet.optim import SGD
from znet.autograd import Tensor

import znet.nn as nn
# from .nn import CrossEntropyLoss  # assume you put it in nn/loss.py

# Set seed for reproducibility
np.random.seed(42)

# Step 1: Define the Model
class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Step 2: Generate Dummy Classification Data
def generate_data(num_samples=100, in_features=2, num_classes=3):
    X = np.random.randn(num_samples, in_features)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

# Step 3: Train the model
def train():
    # Parameters
    in_features = 2
    hidden = 16
    num_classes = 3
    num_epochs = 50
    batch_size = 16
    lr = 0.1

    # Data
    X_data, y_data = generate_data(num_samples=200, in_features=in_features, num_classes=num_classes)

    # Convert to Tensor
    X = Tensor(X_data, requires_grad=True)

    # Model, Loss, Optimizer
    model = SimpleMLP(in_features, hidden, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(X_data), batch_size):
            x_batch = X_data[i:i+batch_size]
            y_batch = y_data[i:i+batch_size]

            x_tensor = Tensor(x_batch, requires_grad=True)

            # Forward
            out = model(x_tensor)
            loss = criterion(out, y_batch)
            total_loss += loss.data

            # Backward
            loss.backward()

            # Update
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / (len(X_data) // batch_size)
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()



