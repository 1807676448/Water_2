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
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define RX_BUFFER_SIZE 64
#define CMD_HEADER 0xAA                             // 包头
#define CMD_TAIL 0x55                               // 包尾
#define CMD_INSTR_SIZE 2                            // 指令长度
#define CMD_TOTAL_SIZE (1 + CMD_INSTR_SIZE + 1 + 1) // 包头+指令+校验+包尾

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

typedef enum
{
  CMD_STATE_WAIT_HEADER,
  CMD_STATE_RECEIVE_INSTR,
  CMD_STATE_RECEIVE_CHECKSUM,
  CMD_STATE_RECEIVE_TAIL
} cmd_state_t;

static cmd_state_t usart3_cmd_state = CMD_STATE_WAIT_HEADER;
static uint8_t usart3_cmd_index = 0;
static uint8_t usart3_cmd_buffer[CMD_TOTAL_SIZE];

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

// USART1接收相关变量 (新增)
uint8_t usart1_rx_buffer[1];

uint8_t TestStart = 0;
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

  HAL_GPIO_WritePin(Boom_GPIO_Port, Boom_Pin, GPIO_PIN_RESET);

  ADS1220_InitStruct(&hads1220, &hspi1); // 初始化ADS1220
  ADS1220_Reset(&hads1220);              // 复位ADS1220
  HAL_Delay(10);
  ADS1220_WriteRegisters(&hads1220, 0, 4, config_reg); // 配置寄存器
  HAL_Delay(1);
  ADS1220_DebugPrintRegisters(&hads1220); // 打印初始配置

  // 启动USART3中断接收
  usart3_rx_index = 0;
  usart3_rx_complete = 0;
  HAL_UART_Receive_IT(&huart3, &usart3_rx_buffer[0], 1);

  // 启动USART1中断接收 (新增)
  HAL_UART_Receive_IT(&huart1, &usart1_rx_buffer[0], 1);

  HAL_GPIO_WritePin(GPIOA, LED_Pin, 0);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    // HAL_GPIO_TogglePin(LED_GPIO_Port, LED_Pin);
    // HAL_Delay(400);
    if (TestStart == 1)
    {
      BoomTest(0, 0);
      TestStart = 0;
      HAL_Delay(200);
    }
    HAL_Delay(10);
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
    printf("%f\n\r", DS18B20_Get_Temp());
  }
}

static uint8_t calculate_checksum(uint8_t *data, uint8_t len)
{
  uint8_t checksum = 0;
  for (uint8_t i = 0; i < len; i++)
  {
    checksum ^= data[i];
  }
  return checksum;
}
// 命令处理函数
static void process_command(uint8_t *cmd_buffer)
{
  // 提取指令部分（跳过包头）
  uint8_t *instruction = &cmd_buffer[1];

  // 计算校验和
  uint8_t received_checksum = cmd_buffer[CMD_TOTAL_SIZE - 2]; // 校验位位置
  uint8_t calculated_checksum = calculate_checksum(instruction, CMD_INSTR_SIZE);

  // 验证校验和
  if (received_checksum == calculated_checksum)
  {
    TestStart = 1; // 设置标志位
    // 可选：通过串口回传确认信息
    // uint8_t ack_msg[] = "Command executed: BoomTest\r\n";
    // HAL_UART_Transmit_IT(&huart3, ack_msg, sizeof(ack_msg) - 1);
  }
  else
  {
    // 校验失败
    uint8_t error_msg[] = "Checksum error\r\n";
    HAL_UART_Transmit_IT(&huart3, error_msg, sizeof(error_msg) - 1);
  }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART3)
  {
    uint8_t received_byte = usart3_rx_buffer[usart3_rx_index];
    switch (usart3_cmd_state)
    {
    case CMD_STATE_WAIT_HEADER:
      if (received_byte == CMD_HEADER)
      {
        usart3_cmd_buffer[0] = received_byte; // 存储包头
        usart3_cmd_index = 1;
        usart3_cmd_state = CMD_STATE_RECEIVE_INSTR;
      }
      break;

    case CMD_STATE_RECEIVE_INSTR:
      if (usart3_cmd_index < (1 + CMD_INSTR_SIZE))
      {
        usart3_cmd_buffer[usart3_cmd_index] = received_byte;
        usart3_cmd_index++;

        // 指令接收完成
        if (usart3_cmd_index == (1 + CMD_INSTR_SIZE))
        {
          usart3_cmd_state = CMD_STATE_RECEIVE_CHECKSUM;
        }
      }
      break;

    case CMD_STATE_RECEIVE_CHECKSUM:
      usart3_cmd_buffer[usart3_cmd_index] = received_byte;
      usart3_cmd_index++;
      usart3_cmd_state = CMD_STATE_RECEIVE_TAIL;
      break;

    case CMD_STATE_RECEIVE_TAIL:
      if (received_byte == CMD_TAIL)
      {
        usart3_cmd_buffer[usart3_cmd_index] = received_byte;

        // 完整命令接收完成，处理命令
        process_command(usart3_cmd_buffer);

        // 重置状态机
        usart3_cmd_state = CMD_STATE_WAIT_HEADER;
        usart3_cmd_index = 0;
      }
      else
      {
        // 包尾错误，重置状态机
        usart3_cmd_state = CMD_STATE_WAIT_HEADER;
        usart3_cmd_index = 0;
      }
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
