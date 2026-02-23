import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Callable


data = np.load("/Users/sonn/Sonn/Workspace/Projects/Oranges/MLOrangeSweetness/data.npz")

X = data["X"]
y = data["y"]


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Encode label
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, stratify=y_encoded)


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# =============================
# Check device (MPS nếu có)
# =============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =============================
# Convert data sang Tensor
# =============================
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# =============================
# Định nghĩa ANN model
# =============================
class ANN(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, num_classes=3):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)  # không softmax vì dùng CrossEntropyLoss
        return x

num_classes = len(np.unique(y_train))
model = ANN(X_train.shape[1], num_classes=num_classes).to(device)

# =============================
# Loss & Optimizer
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================
# Training loop
# =============================
epochs = 30
batch_size = 32

for epoch in range(epochs):
    permutation = torch.randperm(X_train_t.size()[0])
    for i in range(0, X_train_t.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# =============================
# Evaluation
# =============================
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)

y_pred = predicted.cpu().numpy()
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)


