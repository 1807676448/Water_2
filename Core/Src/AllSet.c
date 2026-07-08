#include "AllSet.h"
#include "water_quality_ai.h"
extern uint8_t TestStart;
extern uint64_t tick_last_rx;
extern uint8_t tx_muted;  // 遥控系统静默标志

#define BUS_IDLE_MS 50  // 总线空闲阈值：连续50ms无数据则认为总线空闲

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
    // ========== 抽水阶段: 先开泵3秒抽水，然后关泵使水流平稳 ==========
    HAL_GPIO_WritePin(Boom_GPIO_Port, Boom_Pin, GPIO_PIN_SET);
    HAL_Delay(3000);
    HAL_GPIO_WritePin(Boom_GPIO_Port, Boom_Pin, GPIO_PIN_RESET);

    // ========== Phase 0: 暗光基线采集 (LED全关，泵关闭) ==========
    int base_550_int = 0;
    int base_254_int = 0;

    for (int i = 0; i < 10; i++)
    {
        int32_t dark_550 = ADS1220_ReadConvertOnce(&hads1220, 0);  // 通道0暗光值
        base_550_int += dark_550;

        // 暗光基线串口发送已注释，避免占用信道
        // char buf[64];
        // int n = snprintf(buf, sizeof(buf), "{\"ch\":550,\"dark\":%ld}\r\n", dark_550);
        // HAL_UART_Transmit(&huart3, (uint8_t *)buf, n, 100);

        HAL_Delay(50);

        int32_t dark_254 = ADS1220_ReadConvertOnce(&hads1220, 1);  // 通道1暗光值
        base_254_int += dark_254;

        // n = snprintf(buf, sizeof(buf), "{\"ch\":254,\"dark\":%ld}\r\n", dark_254);
        // HAL_UART_Transmit(&huart3, (uint8_t *)buf, n, 100);
    }
    float Base_550 = ((float)base_550_int / 10 / 8388608.0f) * 3.3f;
    float Base_254 = ((float)base_254_int / 10 / 8388608.0f) * 3.3f;

    // ========== Phase 1: LED点亮检测 ==========
    float Led_550 = 0.0f;
    float Led_254 = 0.0f;
    int Led_550_int = 0;
    int Led_254_int = 0;

    float Tem = 0.0f;

    // 泵已关闭，水流平稳后开始LED检测
    HAL_Delay(500);  // 500ms后开始检测

    // ===== 254nm: 持续点亮LED，稳定后连续采样10次 =====
    // 开启254nm LED并等待足够稳定时间（该LED启动慢，需~2s）
    HAL_TIM_PWM_ConfigChannel(&htim1, &(TIM_OC_InitTypeDef){.OCMode = TIM_OCMODE_PWM1, .Pulse = 1000}, TIM_CHANNEL_3);
    HAL_Delay(2500);  // 等待LED完全稳定（从之前测试数据推算需~1.7s以上）

    // 丢弃前2次转换，让ADC通道切换后充分建立
    ADS1220_ReadConvertOnce(&hads1220, 1);
    ADS1220_ReadConvertOnce(&hads1220, 1);

    for (int i = 0; i < 10; i++)
    {
        int32_t raw = ADS1220_ReadConvertOnce(&hads1220, 1);
        Led_254_int += raw;
        Tem += DS18B20_Get_Temp();

        // 单次检测值串口发送已注释，避免占用信道
        // char buf[64];
        // int n = snprintf(buf, sizeof(buf), "{\"ch\":254,\"idx\":%d,\"raw\":%ld}\r\n", i, raw);
        // HAL_UART_Transmit(&huart3, (uint8_t *)buf, n, 100);

        HAL_Delay(10);
    }

    // 关闭254nm LED
    HAL_TIM_PWM_ConfigChannel(&htim1, &(TIM_OC_InitTypeDef){.OCMode = TIM_OCMODE_PWM1, .Pulse = 0}, TIM_CHANNEL_3);

    HAL_Delay(200);

    // ===== 550nm: 持续点亮LED，稳定后连续采样10次 =====
    HAL_TIM_PWM_ConfigChannel(&htim1, &(TIM_OC_InitTypeDef){.OCMode = TIM_OCMODE_PWM1, .Pulse = 250}, TIM_CHANNEL_2);
    HAL_Delay(200);  // 等待LED完全点亮并稳定

    for (int i = 0; i < 10; i++)
    {
        int32_t raw = ADS1220_ReadConvertOnce(&hads1220, 0);
        Led_550_int += raw;

        // 单次检测值串口发送已注释，避免占用信道
        // char buf[64];
        // int n = snprintf(buf, sizeof(buf), "{\"ch\":550,\"idx\":%d,\"raw\":%ld}\r\n", i, raw);
        // HAL_UART_Transmit(&huart3, (uint8_t *)buf, n, 100);

        HAL_Delay(20);
    }

    // 关闭550nm LED
    HAL_TIM_PWM_ConfigChannel(&htim1, &(TIM_OC_InitTypeDef){.OCMode = TIM_OCMODE_PWM1, .Pulse = 0}, TIM_CHANNEL_2);
    Led_550 = ((float)Led_550_int / 10 / 8388608.0f) * 3.3f;
    Led_254 = ((float)Led_254_int / 10 / 8388608.0f) * 3.3f;
    Tem = Tem / 10;
    HAL_Delay(100);

    TestStart = 0;

    // ========== Phase 2: 补偿计算 (检测值 - 基线值) ==========
    float Comp_550 = Led_550 - Base_550;
    float Comp_254 = Led_254 - Base_254;

    // 模拟输出：COD 随机 2.0~4.0，UV254 = COD * 0.25--用于调试和验证数据传输
    float pred_cod = 2.0f + ((float)(HAL_GetTick() % 2001) / 1000.0f);  // 2.0~4.0
    float pred_uv254 = pred_cod * 0.25f;

    // 如果外部指针有效，赋值回去
    if (out_cod != 0) *out_cod = pred_cod;
    if (out_uv254 != 0) *out_uv254 = pred_uv254;

    // 构建并发送精简 JSON 到 USART3（正常模式仅输出检测结果）
    // 遥控系统静默期间(tx_muted=1)跳过发送，避免占用信道
    if (!tx_muted)
    {
      char tx_buffer[128];
      int len = snprintf(tx_buffer, sizeof(tx_buffer),
          "{\"device_id\":%d,\"COD\":%.2f,\"UV254\":%.2f}\r\n",
          COMM_DEVICE_ID, pred_cod, pred_uv254);

      // 等待总线空闲（共享总线场景下，另一设备可能正在发送）
      while (HAL_GetTick() - tick_last_rx < BUS_IDLE_MS)
      {
        HAL_Delay(1);
      }

      HAL_UART_Transmit(&huart3, (uint8_t *)tx_buffer, len, 1000);
    }
}

void BoardLedTest(void)
{
    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET);
    HAL_Delay(BoardLedTime);
    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);
    HAL_Delay(BoardLedTime);
}
