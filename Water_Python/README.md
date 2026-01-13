# 水质预测神经网络与单片机部署系统

本项目包含了一套完整的水质预测解决方案，从神经网络模型的训练、在PC端的验证，到最终生成可用于STM32单片机的C语言代码。

通过输入 **254nm吸光度**、**550nm吸光度** 和 **温度(tem)**，系统可以预测 **COD** 和 **UV254** 两个水质参数。

## 📁 文件说明与运行流程

整个项目的运行逻辑分为三个阶段：**训练 -> 验证 -> 部署**。请按照以下顺序操作：

### 第一阶段：模型训练

**核心文件**: `new.py`

*   **功能**: 
    1.  读取 `shuizhi.xlsx` 中的水质数据。
    2.  进行数据预处理（StandardScaler 标准化），解决不同特征量级差异过大的问题。
    3.  搭建并训练一个 3 层全连接神经网络 (3输入 -> 32 -> 16 -> 2输出)。
    4.  训练结束后，保存以下三个关键文件：
        *   `water_quality_model.pth` (神经网络权重)
        *   `scaler_x.pkl` (输入数据的标准化参数)
        *   `scaler_y.pkl` (输出数据的标准化参数)
    5.  生成 `training_log.txt` 记录训练过程。
*   **运行**: 
    ```bash
    python new.py
    ```

### 第二阶段：PC端验证与使用

**核心文件**: `use_model.py`

*   **功能**: 
    *   加载训练好的模型和参数文件。
    *   提供一个简单的命令行交互界面，允许用户输入测试数据并查看预测结果，验证模型效果。
*   **运行**: 
    ```bash
    python use_model.py
    ```

### 第三阶段：单片机部署 (STM32/C语言)

**核心文件**: `export_to_c.py`, `water_quality_ai.c`, `water_quality_ai.h`, `model_data.h`

1.  **导出参数**:
    *   运行 `export_to_c.py`。
    *   它会读取 `.pth` 和 `.pkl` 文件，将所有的神经网络权重和标准化参数提取出来，生成一个 C 语言头文件 **`model_data.h`**。
    *   ```bash
        python export_to_c.py
        ```

2.  **移植到工程**:
    *   将 **`water_quality_ai.c`**, **`water_quality_ai.h`**, **`model_data.h`** 三个文件复制到你的 STM32 HAL 工程中。
    *   `water_quality_ai.c` 包含了一个纯C语言实现的轻量级推理引擎（包括标准化、矩阵乘法、ReLU激活）。

3.  **调用方法**:
    *   参见 **`stm32_usage_example.c`** 中的示例代码。在你的主循环中调用 `WaterQuality_Predict` 函数即可。

---

## 🧠 技术原理

### 1. 神经网络架构
采用全连接前馈神经网络 (Feedforward Neural Network / BP Network)。
*   **输入层 (3节点)**: 254nm, 550nm, Temperature (经过 (x - mean)/scale 标准化)
*   **隐藏层1 (32节点)**: 激活函数 ReLU
*   **隐藏层2 (16节点)**: 激活函数 ReLU
*   **输出层 (2节点)**: COD, UV254 (输出后需反标准化还原)

### 2. 标准化 (Normalization)
由于输入数据中，温度约为20-30，而吸光度通常小于2；输出数据中，COD可能高达几十，UV254只有零点几。如果不进行标准化，梯度下降很难收敛。
本项目使用了 `StandardScaler` (Z-Score标准化)，公式为：$z = (x - \mu) / \sigma$。
在单片机部署时，我们也手动实现了这一步，参数由 `scaler_x.pkl` 和 `scaler_y.pkl` 导出。

### 3. 微型推理引擎
为了适应单片机资源，我们在 `water_quality_ai.c` 中手写了最基础的线性代数运算。不依赖任何重型库（如 TensorFlow Lite），仅需标准 math.h 库，Flash占用极低（约3KB）。

---

## 🛠️ 依赖环境

**PC端 (Python)**:
*   Python 3.x
*   pandas, openpyxl (数据处理)
*   numpy, scikit-learn (数值计算与标准化)
*   torch (PyTorch 深度学习框架)
*   joblib (参数保存)

**单片机端 (C语言)**:
*   标准 C99 编译器 (Keil MDK, IAR, GCC等)
*   STM32 HAL 库 (可选，核心算法与硬件无关)
