/**
  ******************************************************************************
  * @file    ads1220.h
  * @brief   ADS1220 24位ADC驱动库
  * @author  嵌入式工程师
  ******************************************************************************
  * @attention
  *
  * 版权所有 (c) 2025 STMicroelectronics。
  * 保留所有权利。
  *
  * 本软件根据许可条款使用，条款可在LICENSE文件中找到。
  *
  ******************************************************************************
  */

#ifndef __ADS1220_H
#define __ADS1220_H

#ifdef __cplusplus
extern "C" {
#endif

/* 包含头文件 --------------------------------------------------------------*/
#include "main.h"
#include "spi.h"
#include "gpio.h"
#include "tim.h"
/* 导出类型 ----------------------------------------------------------------*/
typedef enum
{
    ADS1220_OK   = 0x00U,
    ADS1220_FAIL = 0x01U,
} ADS1220_StatusTypedef;

typedef struct {
    SPI_HandleTypeDef *hspi;        // SPI句柄
    GPIO_TypeDef *cs_port;          // CS引脚端口
    uint16_t cs_pin;                // CS引脚编号
    GPIO_TypeDef *drdy_port;        // DRDY引脚端口  
    uint16_t drdy_pin;              // DRDY引脚编号
} ADS1220_HandleTypeDef;

/* 导出常量 ----------------------------------------------------------------*/
/* ADS1220命令定义 */
#define ADS1220_CMD_RESET          0x06    // 复位命令
#define ADS1220_CMD_START_SYNC     0x08    // 启动同步转换
#define ADS1220_CMD_POWERDOWN      0x02    // 掉电命令
#define ADS1220_CMD_RDATA          0x10    // 读取数据命令
#define ADS1220_CMD_READ_REG       0x20    // 读取寄存器命令
#define ADS1220_CMD_WRITE_REG      0x40    // 写入寄存器命令

/* 通道选择 */
#define ADS1220_CH0                0
#define ADS1220_CH1                1
#define ADS1220_CH2                2
#define ADS1220_CH3                3

/* 寄存器配置宏定义 --------------------------------------------------------*/
/* Reg0 [7:4]MUX - 多路复用器配置 */
#define MUX_P_AIN0_N_AIN1          (0X00U)
#define MUX_P_AIN0_N_AIN2          (0X10U)
#define MUX_P_AIN0_N_AIN3          (0X20U)
#define MUX_P_AIN1_N_AIN2          (0X30U)
#define MUX_P_AIN1_N_AIN3          (0X40U)
#define MUX_P_AIN2_N_AIN3          (0X50U)
#define MUX_P_AIN1_N_AIN0          (0X60U)
#define MUX_P_AIN3_N_AIN2          (0X70U)
#define MUX_P_AIN0_N_AVSS          (0X80U)
#define MUX_P_AIN1_N_AVSS          (0X90U)
#define MUX_P_AIN2_N_AVSS          (0XA0U)
#define MUX_P_AIN3_N_AVSS          (0XB0U)
#define MUX_P_REFP_N_REFN          (0XC0U)
#define MUX_P_AVDD_N_AVSS          (0XD0U)
#define MUX_PN_SHORT_HALFVDD       (0XE0U)

/* Reg0 [3:1]GAIN - 增益设置 */
#define GAIN_1                     (0X00U)
#define GAIN_2                     (0X02U)
#define GAIN_4                     (0X04U)
#define GAIN_8                     (0X06U)
#define GAIN_16                    (0X08U)
#define GAIN_32                    (0X0AU)
#define GAIN_64                    (0X0CU)
#define GAIN_128                   (0X0EU)

/* Reg0 [0]PGA_BYPASS - PGA旁路 */
#define PGA_BYPASS                 (0X01U)
#define PGA_AMP                    (0X00U)

/* Reg1 [7:5]DR - 数据速率 */
#define DR_20SPS                   (0X00U)
#define DR_45SPS                   (0X20U)
#define DR_90SPS                   (0X40U)
#define DR_175SPS                  (0X60U)
#define DR_330SPS                  (0X80U)
#define DR_600SPS                  (0XA0U)
#define DR_1000SPS                 (0XC0U)

/* Reg1 [4:3]MODE - 工作模式 */
#define MODE_NORMAL                (0X00U)
#define MODE_DUTY                  (0X08U)
#define MODE_TURBO                 (0X10U)

/* Reg1 [2]CM - 转换模式 */
#define CM_SINGLE                  (0X00U)
#define CM_CONTINUE                (0X04U)

/* Reg1 [1]TS - 温度传感器模式 */
#define TS_ON                      (0X02U)
#define TS_OFF                     (0X00U)

/* Reg1 [0]BCS - 烧断电流源 */
#define BCS_ON                     (0X01U)
#define BCS_OFF                    (0X00U)

/* Reg2 [7:6]VREF - 参考电压选择 */
#define VREF_INTERNAL              (0X00U)
#define VREF_EXT_REF0_PINS         (0X40U)
#define VREF_EXT_REF1_PINS         (0X80U)
#define VREF_AVDD                  (0XC0U)

/* Reg2 [5:4]FIR - 滤波器设置 */
#define FIR_NONE                   (0X00U)
#define FIR_50_60                  (0X10U)
#define FIR_50                     (0X20U)
#define FIR_60                     (0X30U)

/* Reg2 [3]PSW - 电源开关 */
#define PSW_OPEN                   (0X00U)
#define PSW_CLOSES                 (0X08U)

/* Reg2 [2:0]IDAC - IDAC电流设置 */
#define IDAC_OFF                   (0X00U)
#define IDAC_10uA                  (0X01U)
#define IDAC_50uA                  (0X02U)
#define IDAC_100uA                 (0X03U)
#define IDAC_250uA                 (0X04U)
#define IDAC_500uA                 (0X05U)
#define IDAC_1000uA                (0X06U)
#define IDAC_1500uA                (0X07U)

/* Reg3 [7:5]I1MUX - IDAC1输出选择 */
#define I1MUX_DISABLED             (0X00U)
#define I1MUX_AIN0                 (0X20U)
#define I1MUX_AIN1                 (0X40U)
#define I1MUX_AIN2                 (0X60U)
#define I1MUX_AIN3                 (0X80U)
#define I1MUX_REFP0                (0XA0U)
#define I1MUX_REFN0                (0XC0U)

/* Reg3 [4:2]I2MUX - IDAC2输出选择 */
#define I2MUX_DISABLED             (0X00U)
#define I2MUX_AIN0                 (0X04U)
#define I2MUX_AIN1                 (0X08U)
#define I2MUX_AIN2                 (0X0CU)
#define I2MUX_AIN3                 (0X10U)
#define I2MUX_REFP0                (0X14U)
#define I2MUX_REFN0                (0X18U)

/* Reg3 [1]DRDYM - DRDY模式 */
#define DRDY_ON_DOUT_DRDY          (0X02U)
#define DRDY_ON_DRDY_ONLY          (0X00U)

/* 导出宏 ------------------------------------------------------------------*/
// 检查DRDY引脚是否就绪（低电平有效）
#define ADS1220_DRDY_IS_READY(handle) (!HAL_GPIO_ReadPin((handle)->drdy_port, (handle)->drdy_pin))

// 控制CS引脚电平
#define ADS1220_CS_LOW(handle)      HAL_GPIO_WritePin((handle)->cs_port, (handle)->cs_pin, GPIO_PIN_RESET)
#define ADS1220_CS_HIGH(handle)     HAL_GPIO_WritePin((handle)->cs_port, (handle)->cs_pin, GPIO_PIN_SET)

/* 导出函数 ----------------------------------------------------------------*/

/**
  * @brief  初始化ADS1220结构体
  * @param  handle: ADS1220句柄指针
  * @param  hspi: SPI句柄指针
  * @retval 无
  */
void ADS1220_InitStruct(ADS1220_HandleTypeDef *handle, SPI_HandleTypeDef *hspi);

/**
  * @brief  复位ADS1220
  * @param  handle: ADS1220句柄指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_Reset(ADS1220_HandleTypeDef *handle);

/**
  * @brief  让ADS1220进入掉电模式
  * @param  handle: ADS1220句柄指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_PowerDown(ADS1220_HandleTypeDef *handle);

/**
  * @brief  启动转换
  * @param  handle: ADS1220句柄指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_Start(ADS1220_HandleTypeDef *handle);

/**
  * @brief  等待DRDY信号
  * @param  handle: ADS1220句柄指针
  * @param  timeout: 超时时间（毫秒）
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_WaitDRDY(ADS1220_HandleTypeDef *handle, uint32_t timeout);

/**
  * @brief  读取ADS1220转换数据
  * @param  handle: ADS1220句柄指针
  * @retval 32位有符号整数（实际为24位数据）
  */
int32_t ADS1220_ReadData(ADS1220_HandleTypeDef *handle);

/**
  * @brief  直接读取数据（不等待DRDY）
  * @param  handle: ADS1220句柄指针
  * @retval 32位有符号整数（实际为24位数据）
  */
int32_t ADS1220_ReadDataDirect(ADS1220_HandleTypeDef *handle);

/**
  * @brief  写入配置寄存器
  * @param  handle: ADS1220句柄指针
  * @param  regStartAddr: 起始寄存器地址（0-3）
  * @param  regNum: 寄存器数量（1-4）
  * @param  pData: 指向配置数据的指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_WriteRegisters(ADS1220_HandleTypeDef *handle, uint8_t regStartAddr, uint8_t regNum, uint8_t *pData);

/**
  * @brief  读取配置寄存器
  * @param  handle: ADS1220句柄指针
  * @param  regStartAddr: 起始寄存器地址（0-3）
  * @param  regNum: 寄存器数量（1-4）
  * @param  pData: 指向接收缓冲区的指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_ReadRegisters(ADS1220_HandleTypeDef *handle, uint8_t regStartAddr, uint8_t regNum, uint8_t *pData);

/**
  * @brief  选择ADS1220通道
  * @param  handle: ADS1220句柄指针
  * @param  chl: 通道号（0-3）
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_ChannelSelect(ADS1220_HandleTypeDef *handle, uint8_t chl);

/**
  * @brief  单次转换并读取数据
  * @param  handle: ADS1220句柄指针
  * @param  chl: 通道号（0-3）
  * @retval 32位有符号整数（实际为24位数据）
  */
int32_t ADS1220_ReadConvertOnce(ADS1220_HandleTypeDef *handle, uint8_t chl);

/* 调试函数 ----------------------------------------------------------------*/
/**
  * @brief  打印ADS1220调试信息
  * @param  data: 24位ADC数据
  * @retval 无
  */
void ADS1220_DebugPrint(int32_t data);

/**
  * @brief  打印配置寄存器值
  * @param  handle: ADS1220句柄指针
  * @retval 无
  */
void ADS1220_DebugPrintRegisters(ADS1220_HandleTypeDef *handle);

#ifdef __cplusplus
}
#endif

#endif /* __ADS1220_H */
