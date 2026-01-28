#include "AllSet.h"
#include "water_quality_ai.h"
extern uint8_t TestStart;

void LedTest(void)
{
    HAL_GPIO_WritePin(OC7140_1_GPIO_Port, OC7140_1_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(OC7140_2_GPIO_Port, OC7140_2_Pin, GPIO_PIN_SET);

    HAL_Delay_Us(LedBlinkTime);

    HAL_GPIO_WritePin(OC7140_1_GPIO_Port, OC7140_1_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(OC7140_2_GPIO_Port, OC7140_2_Pin, GPIO_PIN_RESET);

    HAL_Delay_Us(LedBlinkTime);
}

void BoomTest(float* out_cod, float* out_uv254)
{
    float Led_550 = 0.0f;
    float Led_254 = 0.0f;
    int Led_550_int = 0;
    int Led_254_int = 0;

    float Tem;

    HAL_GPIO_WritePin(Boom_GPIO_Port, Boom_Pin, GPIO_PIN_SET);
    HAL_Delay(4000);
    for (int i = 0; i < 10; i++)
    {
        Led_550_int += led550_blink(250);
        HAL_Delay(100);
        Led_254_int -= led254_blink(1000);
        Tem += DS18B20_Get_Temp();
    }
    Led_550 = ((float)Led_550_int / 10 / 8388608.0f) * 3.3f;
    Led_254 = ((float)Led_254_int / 10 / 8388608.0f) * 3.3f;
    Tem = Tem / 10;
    HAL_Delay(300);

    HAL_GPIO_WritePin(Boom_GPIO_Port, Boom_Pin, GPIO_PIN_RESET);

    TestStart = 0;

    // AI 预测
    float pred_cod = 0.0f;
    float pred_uv254 = 0.0f;
    WaterQuality_Predict(Led_254, Led_550, Tem, &pred_cod, &pred_uv254);

    // 如果外部指针有效，赋值回去
    if (out_cod != 0) *out_cod = pred_cod;
    if (out_uv254 != 0) *out_uv254 = pred_uv254;

    // 构建并发送 JSON 格式结果到 USART3
    char tx_buffer[256];
    int len = snprintf(tx_buffer, sizeof(tx_buffer), 
        "{\"Led_550\": %.6f, \"Led_254\": %.6f, \"Temp\": %.4f, \"COD\": %.4f, \"UV254\": %.4f,\"device_id\":%d,\"status\":\"%s\"}\r\n", 
        Led_550, Led_254, Tem, pred_cod, pred_uv254,COMM_DEVICE_ID, "Active");
    
    HAL_UART_Transmit(&huart3, (uint8_t *)tx_buffer, len, 1000);

    // 调试打印
    printf("JSON Sent: %s", tx_buffer);
}

void BoardLedTest(void)
{
    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET);
    HAL_Delay(BoardLedTime);
    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);
    HAL_Delay(BoardLedTime);
}
