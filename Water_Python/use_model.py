import torch
import torch.nn as nn
import numpy as np
import joblib 
import sys

# ---------------------------------------------------------
# 1. 重新定义模型结构 
# (必须与训练时的结构完全一致)
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

# ---------------------------------------------------------
# 2. 加载模型和标准化器
# ---------------------------------------------------------
def load_model_system():
    # 检查文件是否存在
    files = ['water_quality_model.pth', 'scaler_x.pkl', 'scaler_y.pkl']
    for f in files:
        try:
            open(f, 'r')
        except FileNotFoundError:
            print(f"错误: 未找到文件 {f}。请先运行 new.py 完成训练。")
            return None, None, None

    print("正在加载模型和参数...")
    
    # 加载标准化器
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    
    # 初始化模型并加载权重0.
    model = WaterQualityPredictor()
    model.load_state_dict(torch.load('water_quality_model.pth'))
    model.eval() # 设置为评估模式
    
    print("加载成功！")
    return model, scaler_x, scaler_y

# ---------------------------------------------------------
# 3. 预测函数
# ---------------------------------------------------------
def predict_water_quality(absorb_254, absorb_550, temperature):
    """
    输入: 254nm吸光度, 550nm吸光度, 温度
    输出: COD, UV254
    """
    model, scaler_x, scaler_y = load_model_system()
    if model is None:
        return

    # 1. 准备输入数据
    input_data = np.array([[absorb_254, absorb_550, temperature]])
    
    # 2. 数据标准化
    input_scaled = scaler_x.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)
    
    # 3. 模型推理
    with torch.no_grad():
        output_scaled = model(input_tensor)
    
    # 4. 结果反归一化
    output_real = scaler_y.inverse_transform(output_scaled.numpy())
    
    cod = output_real[0][0]
    uv254 = output_real[0][1]
    
    return cod, uv254

# ---------------------------------------------------------
# 4. 主程序入口
# ---------------------------------------------------------
if __name__ == "__main__":
    print("--- 水质预测系统 ---")
    while True:
        try:
            print("\n请输入以下数据 (输入 'q' 退出):")
            s_254 = input("254nm吸光度: ")
            if s_254.lower() == 'q': break
            
            s_550 = input("550nm吸光度: ")
            s_tem = input("温度 (tem): ")
            
            # 转换为浮点数
            val_254 = float(s_254)
            val_550 = float(s_550)
            val_tem = float(s_tem)
            
            # 进行预测
            cod, uv = predict_water_quality(val_254, val_550, val_tem)
            
            print(f"\n>>> 预测结果:")
            print(f"    COD   : {cod:.2f}")
            print(f"    UV254 : {uv:.4f}")
            
        except ValueError:
            print("输入错误，请输入数字。")
        except Exception as e:
            print(f"发生错误: {e}")
