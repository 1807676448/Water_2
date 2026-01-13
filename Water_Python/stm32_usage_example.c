/*
 * stm32_usage_example.c
 * 这是一个 STM32 调用 AI 模型的示例文件。
 * 
 * 集成步骤:
 * 1. 将 water_quality_ai.c, water_quality_ai.h, model_data.h 加入工程。
 * 2. 在需要调用的地方 include "water_quality_ai.h"
 */

#include "main.h"             // STM32 HAL 标准头文件 (根据实际项目可能是 "stm32f1xx_hal.h" 等)
#include "water_quality_ai.h" // 引入 AI 推理库头文件
#include <stdio.h>            // 用于 printf (如果重定向了串口)

// 这里模拟一个 STM32 任务函数
void User_App_WaterQuality_Task(void) {
    
    // ----------------------------------------------------
    // 第一步：准备传感器数据变量
    // ----------------------------------------------------
    float sensor_254_nm = 0.0f;
    float sensor_550_nm = 0.0f;
    float sensor_temperature = 0.0f;
    
    // ----------------------------------------------------
    // 第二步：获取传感器数据
    // 在实际代码中，这里调用 ADC 读取或 I2C 读取函数
    // ----------------------------------------------------
    // 示例：模拟读取的数据
    sensor_254_nm = 1.65f;     // 假设 ADC 转换后的吸光度
    sensor_550_nm = 0.12f;     // 假设 ADC 转换后的吸光度
    sensor_temperature = 23.5f;// 假设 温度传感器DS18B20读数

    // ----------------------------------------------------
    // 第三步：调用 AI 进行预测
    // ----------------------------------------------------
    float result_cod = 0.0f;
    float result_uv254 = 0.0f;

    // 调用我们在 water_quality_ai.c 中定义的函数
    // 该函数计算量极小，耗时通常在微秒(us)级，不会阻塞系统
    WaterQuality_Predict(
        sensor_254_nm, 
        sensor_550_nm, 
        sensor_temperature, 
        &result_cod,   // 结果将存入这里
        &result_uv254  // 结果将存入这里
    );

    // ----------------------------------------------------
    // 第四步：使用预测结果
    // ----------------------------------------------------
    
    // 例如：通过串口打印 (需要重写 fputc)
    printf("--- AI Analysis Result ---\r\n");
    printf("Input: 254nm=%.2f, 550nm=%.2f, Tem=%.1f\r\n", sensor_254_nm, sensor_550_nm, sensor_temperature);
    printf("Predict COD:   %.2f mg/L\r\n", result_cod);
    printf("Predict UV254: %.4f\r\n", result_uv254);
    
    // 例如：简单的阈值逻辑控制
    /*
    if (result_cod > 50.0f) {
        // COD 偏高，开启报警灯
        HAL_GPIO_WritePin(ALARM_GPIO_Port, ALARM_Pin, GPIO_PIN_SET);
    } else {
        HAL_GPIO_WritePin(ALARM_GPIO_Port, ALARM_Pin, GPIO_PIN_RESET);
    }
    */
}

/**
 * 为了让上面的代码在 STM32CubeIDE 环境中运行，
 * 您可以将其放入 main.c 的 while(1) 循环中，或者定时器回调中。
 */
/* 
// 伪代码示例：main.c

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART1_UART_Init();

    while (1) {
        // 每隔 1 秒执行一次分析
        User_App_WaterQuality_Task();
        HAL_Delay(1000);
    }
}
*/
