import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
import os

# ---------------------------------------------------------
# 日志记录配置 (添加的部分)
# ---------------------------------------------------------
class Logger(object):
    def __init__(self, filename='training_log.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 立即写入，防止程序崩溃时丢失日志

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 将控制台输出重定向到 Logger，同时显示在屏幕和保存到文件
sys.stdout = Logger('training_log.txt')
print("日志系统已启动，所有输出将保存至 training_log.txt")

# ---------------------------------------------------------
# 1. 数据准备
# ---------------------------------------------------------
print("正在读取数据...")

# 【实际使用时，请修改文件名为你真实的Excel文件名】
excel_file = 'shuizhi.xlsx' 

try:
    # 尝试读取Excel文件
    # 假设表头为: 254, 550, tem, cod, uv254
    df = pd.read_excel(excel_file)
    print(f"成功读取文件: {excel_file}")
except FileNotFoundError:
    print(f"提示: 未找到文件 '{excel_file}'，将生成模拟数据用于演示代码运行。")
    # 生成模拟数据 (60组)
    data = {
        '254': np.random.rand(60) * 10,
        '550': np.random.rand(60) * 5,
        'tem': np.random.rand(60) * 30 + 10, # 10-40度
    }
    # 模拟输出关系 (y = x * w + b + noise)
    data['cod'] = data['254'] * 2.5 + data['550'] * 1.2 + data['tem'] * 0.5 + np.random.normal(0, 1, 60)
    data['uv254'] = data['254'] * 0.8 + data['550'] * 0.1 + np.random.normal(0, 0.1, 60)
    df = pd.DataFrame(data)

# ---------------------------------------------------------
# 数据列名处理 (防止KeyError)
# ---------------------------------------------------------
# 1. 强制将所有列名转换为字符串 (防止Excel读入数字表头导致KeyError)
# 2. 去除列名可能存在的空格
df.columns = df.columns.astype(str).str.strip()
print("当前数据的列名:", df.columns.tolist())

# 提取特征 X 和 目标 Y
# 输入特征: 254nm, 550nm, 温度
X_raw = df[['254', '550', 'tem']].values
# 输出目标: cod, uv254
y_raw = df[['cod', 'uv254']].values

# 划分训练集和测试集 (80% 训练, 20% 测试)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 2. 数据标准化 (非常重要)
# ---------------------------------------------------------
# 对输入数据进行标准化
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train_raw)
X_test = scaler_x.transform(X_test_raw)

# 对输出数据进行标准化 (有助于模型更快收敛)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train_raw)
y_test = scaler_y.transform(y_test_raw)

# 转换为 PyTorch 的 Tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# ---------------------------------------------------------
# 3. 搭建神经网络模型
# ---------------------------------------------------------
class WaterQualityPredictor(nn.Module):
    def __init__(self):
        super(WaterQualityPredictor, self).__init__()
        # 输入层 3 -> 隐藏层 32
        self.fc1 = nn.Linear(3, 32)
        self.relu1 = nn.ReLU()
        # 隐藏层 32 -> 隐藏层 16
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        # 隐藏层 16 -> 输出层 2 (cod, uv254)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

model = WaterQualityPredictor()

# ---------------------------------------------------------
# 4. 定义损失函数和优化器
# ---------------------------------------------------------
criterion = nn.MSELoss() # 均方误差损失 -> 适合回归问题
optimizer = optim.Adam(model.parameters(), lr=0.01) # 学习率 0.01

# ---------------------------------------------------------
# 5. 开始训练
# ---------------------------------------------------------
print("开始训练模型...")
epochs = 1000 # 训练轮数
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad() # 清空梯度
    loss.backward()       # 计算梯度
    optimizer.step()      # 更新参数
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ---------------------------------------------------------
# 6. 模型评估与预测
# ---------------------------------------------------------
model.eval() # 切换到评估模式
with torch.no_grad():
    # 在测试集上预测
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    
    # 将标准化的预测结果反变换回真实值
    predicted_values = scaler_y.inverse_transform(test_outputs.numpy())
    real_values = scaler_y.inverse_transform(y_test_tensor.numpy())
    
    print("\n---------------- 测试集结果对比 ----------------")
    print(f"Test Loss (Normalized): {test_loss.item():.4f}")
    print("真实值 (COD, UV254)  vs  预测值 (COD, UV254)")
    for i in range(len(predicted_values)):
        real = real_values[i]
        pred = predicted_values[i]
        print(f"真实: [{real[0]:.2f}, {real[1]:.4f}] | 预测: [{pred[0]:.2f}, {pred[1]:.4f}]")

# ---------------------------------------------------------
# 7. 如何预测新数据
# ---------------------------------------------------------
print("\n---------------- 单组数据预测示例 ----------------")
# 假设有一组新数据: 254nm=1.5, 550nm=0.2, tem=25
new_data = np.array([[0.35, 0.55, 25]]) 
    
# 1. 预处理
new_data_scaled = scaler_x.transform(new_data)
new_data_tensor = torch.FloatTensor(new_data_scaled)

# 2. 预测
model.eval()
with torch.no_grad():
    prediction_scaled = model(new_data_tensor)
    # 3. 反归一化得到真实结果
    prediction_real = scaler_y.inverse_transform(prediction_scaled.numpy())

print(f"输入: {new_data[0]}")
print(f"预测结果: COD={prediction_real[0][0]:.2f}, UV254={prediction_real[0][1]:.4f}")

# ---------------------------------------------------------
# 8. 保存模型和标准化参数 (以便后续使用)
# ---------------------------------------------------------
import joblib 

print("\n---------------- 保存模型 ----------------")
# 保存 PyTorch 模型参数
torch.save(model.state_dict(), 'water_quality_model.pth')
print("模型参数已保存至: water_quality_model.pth")

# 保存标准化器 (非常重要！否则预测时无法进行同样的数据处理)
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("标准化参数已保存至: scaler_x.pkl, scaler_y.pkl")
print("训练完成。请运行 use_model.py 进行独立预测。")
