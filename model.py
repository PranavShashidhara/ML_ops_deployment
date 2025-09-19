import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing

titanic = sns.load_dataset("titanic")

features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
df = titanic[features + ["survived"]].dropna()

df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)
X = df.drop("survived", axis=1).values
y = df["survived"].values

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


model = TitanicModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

with torch.no_grad():
    preds = (model(X_test) > 0.5).float()
    acc = (preds.eq(y_test).sum() / len(y_test)).item()
    print(f"Accuracy: {acc:.2f}")

dummy_input = torch.randn(1, X_train.shape[1])

torch.onnx.export(
    model,  # trained model
    dummy_input,  # dummy input
    "app/titanic_model.onnx",  # output file
    input_names=["input"],  # input tensor name
    output_names=["output"],  # output tensor name
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)

print("ONNX model exported successfully!")
