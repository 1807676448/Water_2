/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "spi.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "Led.h"
#include "Ds18B20.h"
#include "AllSet.h"
#include "ADS1220.h"
#include "water_quality_ai.h"
#include "cJSON.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define RX_BUFFER_SIZE 64
// 新调试指令格式: 0xAA 0xAA 0xBB 0xBB (4字节序列)
#define DEBUG_CMD_BYTE0 0xAA
#define DEBUG_CMD_BYTE1 0xAA
#define DEBUG_CMD_BYTE2 0xBB
#define DEBUG_CMD_BYTE3 0xBB

ADS1220_HandleTypeDef hads1220;
uint32_t adc_data;
// 配置寄存器 (使用宏定义方便调整)
// 范围 = +/- Vref / Gain
// 注意：单端测量(AINx-AVSS)且信号接近0V时，必须禁用PGA (PGA_BYPASS)
uint8_t config_reg[4] = {
    MUX_P_AIN0_N_AVSS | GAIN_1 | PGA_BYPASS,               // Config 0: 增益1, 禁用PGA (关键修改)
    DR_20SPS | MODE_NORMAL | CM_SINGLE | TS_OFF | BCS_OFF, // Config 1: 20SPS, 关闭温度传感器
    VREF_AVDD | FIR_NONE | PSW_OPEN | IDAC_OFF,            // Config 2: 使用AVDD(3.3V)作为参考
    I1MUX_DISABLED | I2MUX_DISABLED | DRDY_ON_DRDY_ONLY    // Config 3: IDAC禁用
};

float sensor_254_nm = 0.0f;
float sensor_550_nm = 0.0f;
float sensor_temperature = 0.0f;
float result_cod = 0.0f;
float result_uv254 = 0.0f;

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
// USART3接收相关变量
uint8_t usart3_rx_buffer[RX_BUFFER_SIZE];
uint16_t usart3_rx_index = 0;
uint8_t usart3_rx_complete = 0;
uint8_t cmd_rx_state = 0;  // 新指令状态: 0=等AA1, 1=等AA2, 2=等BB1, 3=等BB2
uint8_t cmd_toggle_state = 0;  // 调试切换指令: 0=等CC1, 1=等CC2, 2=等CC3
uint8_t cmd_mute_state = 0;    // 静默指令: 0=等DD1, 1=等DD2, 2=等DD3
uint8_t cmd_unmute_state = 0;  // 恢复指令: 0=等EE1, 1=等EE2, 2=等EE3

uint8_t tx_muted = 0;  // 0=允许USART3发送, 1=静默(遥控系统占用信道时禁止发送)

// USART1接收相关变量 (新增)
uint8_t usart1_rx_buffer[1];

uint8_t TestStart = 0;

uint8_t debug_mode = 0;  // 0=正常模式(周期检测), 1=调试模式(指令触发检测)

uint64_t tick_counter = 0;
uint64_t tick_heart = 0;

// 初始水泵控制（非阻塞10秒）
uint8_t initial_pump_done = 0;
uint64_t tick_initial_pump = 0;

// 总线空闲检测：记录最后一次收到字节的时刻
uint64_t tick_last_rx = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int fputc(int ch, FILE *f)
{
  HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, 0xffff); ///< 普通串口发送数据
  while (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_TC) == RESET)
  {
  } ///< 等待发送完成
  return ch;
}

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_SPI1_Init();
  MX_USART1_UART_Init();
  MX_TIM6_Init();
  MX_TIM7_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();
  MX_TIM1_Init();
  MX_USART3_UART_Init();
  /* USER CODE BEGIN 2 */
  HAL_TIM_Base_Start_IT(&htim6);
  HAL_TIM_Base_Start_IT(&htim7);
  HAL_TIM_Base_Start_IT(&htim1);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);

  // 初始化完成后开启水泵（非阻塞），10秒后自动关闭
  HAL_GPIO_WritePin(Boom_GPIO_Port, Boom_Pin, GPIO_PIN_SET);
  tick_initial_pump = HAL_GetTick();
  initial_pump_done = 0;

  ADS1220_InitStruct(&hads1220, &hspi1); // 初始化ADS1220
  ADS1220_Reset(&hads1220);              // 复位ADS1220
  HAL_Delay(10);
  ADS1220_WriteRegisters(&hads1220, 0, 4, config_reg); // 配置寄存器
  HAL_Delay(1);
  // ADS1220_DebugPrintRegisters(&hads1220); // 打印初始配置 (已注释，避免占用信道)

  // 启动USART3中断接收
  usart3_rx_index = 0;
  usart3_rx_complete = 0;
  HAL_UART_Receive_IT(&huart3, &usart3_rx_buffer[0], 1);

  // 启动USART1中断接收 (新增)
  HAL_UART_Receive_IT(&huart1, &usart1_rx_buffer[0], 1);

  HAL_GPIO_WritePin(GPIOA, LED_Pin, 0);

  tick_counter = HAL_GetTick();
  tick_heart = HAL_GetTick();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    // 非阻塞检查：初始水泵运行10秒后关闭
    if (!initial_pump_done && (HAL_GetTick() - tick_initial_pump >= 5000))
    {
      HAL_GPIO_WritePin(Boom_GPIO_Port, Boom_Pin, GPIO_PIN_RESET);
      initial_pump_done = 1;
    }

    if (debug_mode)
    {
      // 调试模式：由串口指令触发单次检测
      if (TestStart)
      {
        BoomTest(0, 0);
        TestStart = 0;
      }
    }
    else
    {
      // 正常模式：每10秒周期检测
      if (HAL_GetTick() - tick_counter > 10000)
      {
        BoomTest(0, 0);
        tick_counter = HAL_GetTick();
        HAL_Delay(200);
      }
    }

    // 心跳包已注释，仅保留数据上传，避免占用信道
    // if (!debug_mode)
    // {
    //   if (HAL_GetTick() - tick_heart > 1000)
    //   {
    //     tick_heart = HAL_GetTick();
    //     char tx_buffer_1[80];
    //
    //     int len = snprintf(tx_buffer_1, sizeof(tx_buffer_1),
    //                        "{\"device_id\":%d,\"status\":\"%s\"}\r\n",
    //                        COMM_DEVICE_ID, "online");
    //
    //     HAL_UART_Transmit(&huart3, (uint8_t *)tx_buffer_1, len, 1000);
    //   }
    // }
    // BoardLedTest();
    HAL_Delay(200);
  }
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
   */
  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1_BOOST);

  /** Initializes the RCC Oscillators according to the specified parameters
   * in the RCC_OscInitTypeDef structure.
   */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV4;
  RCC_OscInitStruct.PLL.PLLN = 85;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
   */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
uint16_t i = 0;
uint16_t k = 0;

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  // 判断中断是否来自 TIM6
  if (htim->Instance == TIM6)
  {
    // printf("%f\n\r", DS18B20_Get_Temp()); // 已注释，避免占用信道
  }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART3)
  {
    tick_last_rx = HAL_GetTick();  // 更新最后收到字节的时刻
    uint8_t received_byte = usart3_rx_buffer[usart3_rx_index];

    // 新指令格式: 0xAA 0xAA 0xBB 0xBB (4字节序列状态机, 全内联)
    switch (cmd_rx_state)
    {
    case 0:  // 等待第一个 0xAA
      if (received_byte == DEBUG_CMD_BYTE0)
      {
        cmd_rx_state = 1;
      }
      break;

    case 1:  // 等待第二个 0xAA
      if (received_byte == DEBUG_CMD_BYTE1)
      {
        cmd_rx_state = 2;
      }
      else
      {
        cmd_rx_state = 0;
      }
      break;

    case 2:  // 等待第一个 0xBB
      if (received_byte == DEBUG_CMD_BYTE2)
      {
        cmd_rx_state = 3;
      }
      else
      {
        cmd_rx_state = 0;
      }
      break;

    case 3:  // 等待第二个 0xBB → 指令完成
      if (received_byte == DEBUG_CMD_BYTE3)
      {
        TestStart = 1;   // 触发单次检测
      }
      cmd_rx_state = 0;  // 无论匹配与否, 均重置状态
      break;
    }

    // 调试模式切换指令: 0xCC 0xCC 0xCC (3字节序列)
    switch (cmd_toggle_state)
    {
    case 0:  // 等待第一个 0xCC
      if (received_byte == 0xCC)
      {
        cmd_toggle_state = 1;
      }
      break;

    case 1:  // 等待第二个 0xCC
      if (received_byte == 0xCC)
      {
        cmd_toggle_state = 2;
      }
      else
      {
        cmd_toggle_state = 0;
      }
      break;

    case 2:  // 等待第三个 0xCC → 指令完成
      if (received_byte == 0xCC)
      {
        debug_mode = !debug_mode;  // 切换调试模式
        // 调试模式切换回传已注释，仅保留数据上传，避免占用信道
        // char toggle_msg[64];
        // int toggle_len = snprintf(toggle_msg, sizeof(toggle_msg),
        //     "{\"device_id\":%d,\"debug_mode\":%d}\r\n",
        //     COMM_DEVICE_ID, debug_mode);
        // HAL_UART_Transmit(&huart3, (uint8_t *)toggle_msg, toggle_len, 1000);
      }
      cmd_toggle_state = 0;
      break;
    }

    // 静默开启指令: 0xDD 0xDD 0xDD (遥控系统启动时发出)
    switch (cmd_mute_state)
    {
    case 0:
      if (received_byte == 0xDD) { cmd_mute_state = 1; }
      break;
    case 1:
      if (received_byte == 0xDD) { cmd_mute_state = 2; }
      else { cmd_mute_state = 0; }
      break;
    case 2:
      if (received_byte == 0xDD) { tx_muted = 1; }
      cmd_mute_state = 0;
      break;
    }

    // 静默关闭指令: 0xEE 0xEE 0xEE (遥控系统退出时发出)
    switch (cmd_unmute_state)
    {
    case 0:
      if (received_byte == 0xEE) { cmd_unmute_state = 1; }
      break;
    case 1:
      if (received_byte == 0xEE) { cmd_unmute_state = 2; }
      else { cmd_unmute_state = 0; }
      break;
    case 2:
      if (received_byte == 0xEE) { tx_muted = 0; }
      cmd_unmute_state = 0;
      break;
    }

    // 继续接收下一个字节
    if (usart3_rx_index < RX_BUFFER_SIZE - 1)
    {
      usart3_rx_index++;
      HAL_UART_Receive_IT(&huart3, &usart3_rx_buffer[usart3_rx_index], 1);
    }
    else
    {
      // 缓冲区满，重置索引
      usart3_rx_index = 0;
      HAL_UART_Receive_IT(&huart3, &usart3_rx_buffer[usart3_rx_index], 1);
    }
  }
}

// 错误回调
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART3)
  {
    // 清除错误标志（通过读取寄存器）
    __HAL_UART_CLEAR_OREFLAG(huart);
    __HAL_UART_CLEAR_NEFLAG(huart);
    __HAL_UART_CLEAR_FEFLAG(huart);
    __HAL_UART_CLEAR_PEFLAG(huart);

    // 重新启动中断接收
    HAL_UART_Receive_IT(&huart3, &usart3_rx_buffer[0], 1);
  }
}
/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
