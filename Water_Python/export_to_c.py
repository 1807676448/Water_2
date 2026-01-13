import torch
import torch.nn as nn
import joblib
import numpy as np

# ---------------------------------------------------------
# 1. 重新定义模型结构 (用于加载权重)
# ---------------------------------------------------------
class WaterQualityPredictor(nn.Module):
    def __init__(self):
        super(WaterQualityPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        pass

# ---------------------------------------------------------
# 2. 加载模型和标准化参数
# ---------------------------------------------------------
try:
    model = WaterQualityPredictor()
    model.load_state_dict(torch.load('water_quality_model.pth'))
    model.eval()
    
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    print("模型和参数加载成功。")
except FileNotFoundError:
    print("错误：找不到模型文件 (.pth) 或 参数文件 (.pkl)。请先运行 new.py。")
    exit()

# ---------------------------------------------------------
# 3. 辅助函数：将张量/数组转换为C语言数组字符串
# ---------------------------------------------------------
def float_array_to_string(arr, name):
    # arr: numpy array (flattened)
    data_str = ", ".join([f"{x:.6f}f" for x in arr.flatten()])
    return f"const float {name}[] = {{{data_str}}};\n"

# ---------------------------------------------------------
# 4. 提取权重并生成 .h 头文件
# ---------------------------------------------------------
header_content = "#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n"

# --- 1. 标准化参数 (Input) ---
# StandardScaler: x_new = (x - mean) / scale
# 对应 sklearn 的 .mean_ 和 .scale_
header_content += "// Input Scaler (StandardScaler)\n"
header_content += float_array_to_string(scaler_x.mean_, "INPUT_MEAN")
header_content += float_array_to_string(scaler_x.scale_, "INPUT_SCALE")

# --- 2. 标准化参数 (Output) ---
header_content += "\n// Output Scaler (StandardScaler)\n"
header_content += float_array_to_string(scaler_y.mean_, "OUTPUT_MEAN")
header_content += float_array_to_string(scaler_y.scale_, "OUTPUT_SCALE")

# --- 3. 神经网络权重 ---
# PyTorch 的 Linear 层权重形状是 [out_features, in_features]
# 导出时我们通常展平，C语言推理时按顺序读取
state_dict = model.state_dict()

# Layer 1: 3 -> 32
w1 = state_dict['fc1.weight'].numpy() # Shape [32, 3]
b1 = state_dict['fc1.bias'].numpy()   # Shape [32]
header_content += "\n// Layer 1 (3 -> 32)\n"
header_content += f"#define W1_ROWS 32\n#define W1_COLS 3\n"
header_content += float_array_to_string(w1, "W1")
header_content += float_array_to_string(b1, "B1")

# Layer 2: 32 -> 16
w2 = state_dict['fc2.weight'].numpy() # Shape [16, 32]
b2 = state_dict['fc2.bias'].numpy()   # Shape [16]
header_content += "\n// Layer 2 (32 -> 16)\n"
header_content += f"#define W2_ROWS 16\n#define W2_COLS 32\n"
header_content += float_array_to_string(w2, "W2")
header_content += float_array_to_string(b2, "B2")

# Layer 3: 16 -> 2
w3 = state_dict['fc3.weight'].numpy() # Shape [2, 16]
b3 = state_dict['fc3.bias'].numpy()   # Shape [2]
header_content += "\n// Layer 3 (16 -> 2)\n"
header_content += f"#define W3_ROWS 2\n#define W3_COLS 16\n"
header_content += float_array_to_string(w3, "W3")
header_content += float_array_to_string(b3, "B3")

header_content += "\n#endif // MODEL_DATA_H\n"

# 写入文件
with open('model_data.h', 'w', encoding='utf-8') as f:
    f.write(header_content)

print("生成成功！已生成 'model_data.h'，请将其复制到你的C工程中。")
