import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取Excel文件
file_path = r'C:\CodeSandbox\Shandong2023\extracted_data\牛死亡数量统计.xlsx'
df = pd.read_excel(file_path)

dates = df.iloc[:, 0]
data = df.iloc[:, 3].values

# 找到数据为0的索引
zero_indices = np.where(data == 0)[0]

# 对数据为0的地方进行三次样条插值
non_zero_indices = np.where(data != 0)[0]
non_zero_data = data[non_zero_indices]

cs = CubicSpline(non_zero_indices, non_zero_data)
data_interpolated = data.copy()
data_interpolated[zero_indices] = cs(zero_indices)

# 数据归一化
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_interpolated.reshape(-1, 1)).flatten()

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=200, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h_0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c_0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        lstm_out_last = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        predictions = self.linear(lstm_out_last)
        return predictions

# 准备数据
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 参数设置
train_window = 5  # 时间窗口大小
train_data = create_inout_sequences(data_normalized, train_window)

# 创建TensorDataset
train_tensor = TensorDataset(
    torch.tensor([seq for seq, _ in train_data], dtype=torch.float32).unsqueeze(-1),
    torch.tensor([label for _, label in train_data], dtype=torch.float32).unsqueeze(-1)
)

# 按顺序分割数据集
train_size = int(len(train_tensor) * 0.7)
val_size = int(len(train_tensor) * 0.2)
test_size = len(train_tensor) - train_size - val_size

# 进行切片分割
train_dataset = torch.utils.data.Subset(train_tensor, list(range(train_size)))
val_dataset = torch.utils.data.Subset(train_tensor, list(range(train_size, train_size + val_size)))
test_dataset = torch.utils.data.Subset(train_tensor, list(range(train_size + val_size+16, len(train_tensor))))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 模型实例化和训练
model = LSTMModel(input_size=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 400
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for seq, labels in train_loader:
        optimizer.zero_grad()

        y_pred = model(seq).squeeze()  # 确保预测的形状和标签一致
        labels = labels.squeeze()

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

        train_loss += single_loss.item()
    
    train_loss /= len(train_loader)

    if epoch % 10 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_seq, val_labels in val_loader:
                val_y_pred = model(val_seq).squeeze()  # 确保预测的形状和标签一致
                val_labels = val_labels.squeeze()

                val_loss += loss_function(val_y_pred, val_labels).item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch} train loss: {train_loss} val loss: {val_loss}')

# 测试模型并打印真实值和预测值
model.eval()
test_loss = 0.0
true_values = []
predicted_values = []
with torch.no_grad():
    for test_seq, test_labels in test_loader:
        test_y_pred = model(test_seq).squeeze()  # 确保预测的形状和标签一致
        test_labels = test_labels.squeeze()

        test_loss += loss_function(test_y_pred, test_labels).item()

        # 反归一化
        test_labels_inv = scaler.inverse_transform(test_labels.cpu().numpy().reshape(-1, 1))
        test_y_pred_inv = scaler.inverse_transform(test_y_pred.cpu().numpy().reshape(-1, 1))

        true_values.extend(test_labels_inv.flatten())
        predicted_values.extend(test_y_pred_inv.flatten())

        # 打印非插值部分
        if test_labels_inv[0][0] not in data[zero_indices]:
            print(f'True value: {test_labels_inv[0][0]}, Predicted value: {test_y_pred_inv[0][0]}')

print(f'Test loss: {test_loss / len(test_loader)}')

# 绘制真实值和预测值的折线图
plt.figure(figsize=(14, 7))
plt.plot(range(len(true_values)), true_values, label='真实值', color='blue', marker='o')
plt.plot(range(len(predicted_values)), predicted_values, label='预测值', color='red', linestyle='--', marker='x')
plt.xlabel('时间步')
plt.ylabel('数值')
plt.title('真实值与预测值对比')
plt.legend()
plt.grid(True)
plt.show()
