# backend/train_and_save_model.py
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# Define model architecture
class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 9),
            nn.ReLU(),
            nn.Linear(9, 3)
        )

    def forward(self, x):
        return self.net(x)

# Load data
url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
df = pd.read_csv(url)
df['variety'] = df['variety'].replace({'Setosa': 0.0, 'Versicolor': 1.0, 'Virginica': 2.0})

x = df.drop('variety', axis=1).values
y = df['variety'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# Train model
model = IrisModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("✅ Model saved as model.pth")
