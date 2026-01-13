#include "water_quality_ai.h"
#include <math.h>
#include <stdio.h>

// -------------------------------------------------------------------------
// 包含权重文件
// 注意：请确保运行 export_to_c.py 后生成的 model_data.h 文件在项目的 Include Path 中
// -------------------------------------------------------------------------
#include "model_data.h"

// ------------------------------------------------------------
// 内部辅助函数 (static 修饰，限制在当前文件内可见，避免命名冲突)
// ------------------------------------------------------------

/**
 * @brief ReLU 激活函数: f(x) = max(0, x)
 * @param data 数据数组指针
 * @param size 数组长度
 */
static void float_relu(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (data[i] < 0.0f)
        {
            data[i] = 0.0f;
        }
    }
}

/**
 * @brief 全连接层 (Dense/Linear) 计算: y = W * x + b
 * @param input   输入向量 [cols]
 * @param weights 权重矩阵 [rows * cols] (展平的一维数组)
 * @param bias    偏置向量 [rows]
 * @param output  输出向量 [rows]
 * @param rows    输出维度
 * @param cols    输入维度
 */
static void layer_dense(const float *input, const float *weights, const float *bias, float *output, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        float sum = 0.0f;
        // 矩阵乘法: 累加 input[j] * weights[i, j]
        for (int j = 0; j < cols; j++)
        {
            // 注意: 这里的索引映射与 export_to_c.py 的导出顺序一致
            // PyTorch Linear权重默认是 [Out, In] (Row-Major flattening)
            sum += input[j] * weights[i * cols + j];
        }
        output[i] = sum + bias[i];
    }
}

// ------------------------------------------------------------
// 核心接口实现
// ------------------------------------------------------------

void WaterQuality_Predict(float in_254, float in_550, float in_tem, float *out_cod, float *out_uv254)
{
    // 0. 定义临时缓冲区 (Layer buffer)
    // 这些数组定义在栈上。
    // 如果是 RAM 极小的单片机(如8051)，建议改为全局 static 变量复用内存
    // 对于 STM32，栈空间通常足够 (这里总共约 32+16+2 个float，不到200字节)
    float layer1_out[W1_ROWS]; // 隐藏层 1 输出 (32)
    float layer2_out[W2_ROWS]; // 隐藏层 2 输出 (16)
    float layer3_out[W3_ROWS]; // 输出层输出 (2)

    // 1. 输入数据准备与标准化 (StandardScaler)
    // 公式: x_new = (x - mean) / scale
    float input_vec[3];

    // INPUT_MEAN 和 INPUT_SCALE 定义在 model_data.h 中
    input_vec[0] = (in_254 - INPUT_MEAN[0]) / INPUT_SCALE[0];
    input_vec[1] = (in_550 - INPUT_MEAN[1]) / INPUT_SCALE[1];
    input_vec[2] = (in_tem - INPUT_MEAN[2]) / INPUT_SCALE[2];

    // 2. 第一层 FC (3 -> 32) + ReLU
    layer_dense(input_vec, W1, B1, layer1_out, W1_ROWS, W1_COLS);
    float_relu(layer1_out, W1_ROWS);

    // 3. 第二层 FC (32 -> 16) + ReLU
    layer_dense(layer1_out, W2, B2, layer2_out, W2_ROWS, W2_COLS);
    float_relu(layer2_out, W2_ROWS);

    // 4. 第三层 FC (16 -> 2)
    // 根据模型设计，输出层不使用激活函数（线性输出）
    layer_dense(layer2_out, W3, B3, layer3_out, W3_ROWS, W3_COLS);

    // 5. 输出反标准化 (StandardScaler 反向)
    // 公式: y_real = y_pred * scale + mean
    // OUTPUT_SCALE 和 OUTPUT_MEAN 定义在 model_data.h 中
    float pred_cod = layer3_out[0] * OUTPUT_SCALE[0] + OUTPUT_MEAN[0];
    float pred_uv254 = layer3_out[1] * OUTPUT_SCALE[1] + OUTPUT_MEAN[1];

    if (out_cod != 0)
    {
        *out_cod = pred_cod;
    }

    if (out_uv254 != 0)
    {
        *out_uv254 = pred_uv254;
    }

    printf("--- AI Analysis Result ---\r\n");
    printf("Input: 254nm=%.2f, 550nm=%.2f, Tem=%.1f\r\n", in_254, in_550, in_tem);
    printf("Predict COD:   %.2f mg/L\r\n", pred_cod);
    printf("Predict UV254: %.4f\r\n", pred_uv254);
}
