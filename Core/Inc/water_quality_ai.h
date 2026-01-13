#ifndef WATER_QUALITY_AI_H
#define WATER_QUALITY_AI_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief 水质预测 AI 模型接口
 * 
 * 该模块实现了基于 3层全连接神经网络 的推理逻辑。
 * 权重数据在 "model_data.h" 中定义。
 */

/**
 * @brief 核心预测函数
 * 此函数执行完整的神经网络前向传播过程：
 * 1. 输入归一化 (StandardScaler)
 * 2. 神经网络推理 (FC -> ReLU -> FC -> ReLU -> FC)
 * 3. 输出反归一化 (StandardScaler)
 * 
 * @param in_254   输入: 254nm 吸光度值
 * @param in_550   输入: 550nm 吸光度值
 * @param in_tem   输入: 温度 (摄氏度)
 * @param out_cod  输出: 指针，用于返回预测的 COD (化学需氧量)
 * @param out_uv254 输出: 指针，用于返回预测的 UV254 值
 */
void WaterQuality_Predict(float in_254, float in_550, float in_tem, float* out_cod, float* out_uv254);

#ifdef __cplusplus
}
#endif

#endif // WATER_QUALITY_AI_H
