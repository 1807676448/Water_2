/**
  ******************************************************************************
  * @file    ads1220.c
  * @brief   ADS1220 24位ADC驱动实现
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

/* 包含头文件 --------------------------------------------------------------*/
#include "ads1220.h"
#include <stdio.h>

/* 私有常量 ----------------------------------------------------------------*/
#define ADS1220_SPI_TIMEOUT         100    // SPI超时时间（毫秒）
#define ADS1220_DRDY_TIMEOUT        500    // DRDY超时时间（毫秒）

/* 私有函数原型 ------------------------------------------------------------*/
static ADS1220_StatusTypedef ADS1220_SPI_Write(ADS1220_HandleTypeDef *handle, uint8_t *data, uint8_t len);
static ADS1220_StatusTypedef ADS1220_SPI_Receive(ADS1220_HandleTypeDef *handle, uint8_t *data, uint8_t len);

/* 私有函数 ----------------------------------------------------------------*/

/**
  * @brief  SPI写函数
  * @param  handle: ADS1220句柄指针
  * @param  data: 发送数据指针
  * @param  len: 数据长度
  * @retval ADS1220状态
  */
static ADS1220_StatusTypedef ADS1220_SPI_Write(ADS1220_HandleTypeDef *handle, uint8_t *data, uint8_t len)
{
    ADS1220_CS_LOW(handle);
    HAL_Delay_Us(10);  // CS建立时间延迟
    
    for (uint8_t i = 0; i < len; i++)
    {
        if (HAL_OK != HAL_SPI_Transmit(handle->hspi, &data[i], 1, ADS1220_SPI_TIMEOUT))
        {
            ADS1220_CS_HIGH(handle);
            return ADS1220_FAIL;
        }
    }
    
    ADS1220_CS_HIGH(handle);
    return ADS1220_OK;
}

/**
  * @brief  SPI读函数
  * @param  handle: ADS1220句柄指针
  * @param  data: 接收数据指针
  * @param  len: 数据长度
  * @retval ADS1220状态
  */
static ADS1220_StatusTypedef ADS1220_SPI_Receive(ADS1220_HandleTypeDef *handle, uint8_t *data, uint8_t len)
{
    uint8_t temp = 0xFF;
    
    ADS1220_CS_LOW(handle);
    HAL_Delay_Us(10);  // CS建立时间延迟
    
    for (uint8_t i = 0; i < len; i++)
    {
        if (HAL_OK != HAL_SPI_TransmitReceive(handle->hspi, &temp, &data[i], 1, ADS1220_SPI_TIMEOUT))
        {
            ADS1220_CS_HIGH(handle);
            return ADS1220_FAIL;
        }
    }
    
    ADS1220_CS_HIGH(handle);
    return ADS1220_OK;
}

/* 导出函数 ----------------------------------------------------------------*/

/**
  * @brief  初始化ADS1220结构体（使用默认引脚定义）
  * @param  handle: ADS1220句柄指针
  * @param  hspi: SPI句柄指针
  * @retval 无
  */
void ADS1220_InitStruct(ADS1220_HandleTypeDef *handle, SPI_HandleTypeDef *hspi)
{
    // 设置SPI句柄
    handle->hspi = hspi;
    
    // 设置CS引脚（PA4）
    handle->cs_port = AD_CS_GPIO_Port;
    handle->cs_pin = AD_CS_Pin;
    
    // 设置DRDY引脚（PA3）
    handle->drdy_port = AD_DRDY_GPIO_Port;
    handle->drdy_pin = AD_DRDY_Pin;
}

/**
  * @brief  复位ADS1220
  * @param  handle: ADS1220句柄指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_Reset(ADS1220_HandleTypeDef *handle)
{
    uint8_t cmd = ADS1220_CMD_RESET;
    return ADS1220_SPI_Write(handle, &cmd, 1);
}

/**
  * @brief  让ADS1220进入掉电模式
  * @param  handle: ADS1220句柄指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_PowerDown(ADS1220_HandleTypeDef *handle)
{
    uint8_t cmd = ADS1220_CMD_POWERDOWN;
    return ADS1220_SPI_Write(handle, &cmd, 1);
}

/**
  * @brief  启动转换
  * @param  handle: ADS1220句柄指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_Start(ADS1220_HandleTypeDef *handle)
{
    uint8_t cmd = ADS1220_CMD_START_SYNC;
    return ADS1220_SPI_Write(handle, &cmd, 1);
}

/**
  * @brief  等待DRDY信号
  * @param  handle: ADS1220句柄指针
  * @param  timeout: 超时时间（毫秒）
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_WaitDRDY(ADS1220_HandleTypeDef *handle, uint32_t timeout)
{
    uint32_t tickstart = HAL_GetTick();
    
    // 等待DRDY引脚变为低电平（数据就绪）
    while (!ADS1220_DRDY_IS_READY(handle))
    {
        // 检查是否超时
        if ((HAL_GetTick() - tickstart) > timeout)
        {
            return ADS1220_FAIL;
        }
    }
    
    return ADS1220_OK;
}

/**
  * @brief  读取ADS1220转换数据
  * @param  handle: ADS1220句柄指针
  * @retval 32位有符号整数（实际为24位数据）
  */
int32_t ADS1220_ReadData(ADS1220_HandleTypeDef *handle)
{
    uint8_t temp[3] = {0};
    uint32_t returnVal = 0;
    uint8_t cmd = ADS1220_CMD_RDATA;
    
    ADS1220_CS_LOW(handle);
    HAL_Delay_Us(10);  // CS建立时间延迟
    
    // 发送读取数据命令
    if (HAL_OK != HAL_SPI_Transmit(handle->hspi, &cmd, 1, ADS1220_SPI_TIMEOUT))
    {
        ADS1220_CS_HIGH(handle);
        return 0;
    }
    
    // 读取3个字节数据
    for (uint8_t i = 0; i < 3; i++)
    {
        if (HAL_OK != HAL_SPI_Receive(handle->hspi, &temp[i], 1, ADS1220_SPI_TIMEOUT))
        {
            ADS1220_CS_HIGH(handle);
            return 0;
        }
    }
    
    ADS1220_CS_HIGH(handle);
    
    // 组合24位数据
    returnVal = (temp[0] << 16) | (temp[1] << 8) | temp[2];
    
    // 处理符号扩展（24位转32位）
    if (returnVal & 0x00800000)
    {
        returnVal |= 0xFF000000;
    }
    
    return (int32_t)returnVal;
}

/**
  * @brief  直接读取数据（不等待DRDY）
  * @param  handle: ADS1220句柄指针
  * @retval 32位有符号整数（实际为24位数据）
  */
int32_t ADS1220_ReadDataDirect(ADS1220_HandleTypeDef *handle)
{
    // 直接调用ReadData函数，不需要额外等待DRDY
    return ADS1220_ReadData(handle);
}

/**
  * @brief  写入配置寄存器
  * @param  handle: ADS1220句柄指针
  * @param  regStartAddr: 起始寄存器地址（0-3）
  * @param  regNum: 寄存器数量（1-4）
  * @param  pData: 指向配置数据的指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_WriteRegisters(ADS1220_HandleTypeDef *handle, uint8_t regStartAddr, uint8_t regNum, uint8_t *pData)
{
    uint8_t temp = ((regStartAddr << 2) & 0x0C);
    temp |= (regNum - 1) & 0x03;
    temp |= ADS1220_CMD_WRITE_REG;
    
    ADS1220_CS_LOW(handle);
    HAL_Delay_Us(10);  // CS建立时间延迟
    
    // 发送写寄存器命令
    if (HAL_OK != HAL_SPI_Transmit(handle->hspi, &temp, 1, ADS1220_SPI_TIMEOUT))
    {
        ADS1220_CS_HIGH(handle);
        return ADS1220_FAIL;
    }
    
    // 发送寄存器数据
    for (uint8_t i = 0; i < regNum; i++)
    {
        if (HAL_OK != HAL_SPI_Transmit(handle->hspi, &pData[i], 1, ADS1220_SPI_TIMEOUT))
        {
            ADS1220_CS_HIGH(handle);
            return ADS1220_FAIL;
        }
    }
    
    ADS1220_CS_HIGH(handle);
    return ADS1220_OK;
}

/**
  * @brief  读取配置寄存器
  * @param  handle: ADS1220句柄指针
  * @param  regStartAddr: 起始寄存器地址（0-3）
  * @param  regNum: 寄存器数量（1-4）
  * @param  pData: 指向接收缓冲区的指针
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_ReadRegisters(ADS1220_HandleTypeDef *handle, uint8_t regStartAddr, uint8_t regNum, uint8_t *pData)
{
    uint8_t temp = ((regStartAddr << 2) & 0x0C);
    temp |= (regNum - 1) & 0x03;
    temp |= ADS1220_CMD_READ_REG;
    
    ADS1220_CS_LOW(handle);
    HAL_Delay_Us(10);  // CS建立时间延迟
    
    // 发送读寄存器命令
    if (HAL_OK != HAL_SPI_Transmit(handle->hspi, &temp, 1, ADS1220_SPI_TIMEOUT))
    {
        ADS1220_CS_HIGH(handle);
        return ADS1220_FAIL;
    }
    
    // 读取寄存器数据
    for (uint8_t i = 0; i < regNum; i++)
    {
        if (HAL_OK != HAL_SPI_Receive(handle->hspi, &pData[i], 1, ADS1220_SPI_TIMEOUT))
        {
            ADS1220_CS_HIGH(handle);
            return ADS1220_FAIL;
        }
    }
    
    ADS1220_CS_HIGH(handle);
    return ADS1220_OK;
}

/**
  * @brief  选择ADS1220通道（只修改MUX位，保留其他配置）
  * @param  handle: ADS1220句柄指针
  * @param  chl: 通道号（0-3）
  * @retval ADS1220状态
  */
ADS1220_StatusTypedef ADS1220_ChannelSelect(ADS1220_HandleTypeDef *handle, uint8_t chl)
{
    uint8_t reg0 = 0;
    uint8_t new_mux = 0;
    
    // 1. 读取当前的 Config 0 寄存器
    if (ADS1220_OK != ADS1220_ReadRegisters(handle, 0, 1, &reg0))
    {
        return ADS1220_FAIL;
    }

    // 2. 确定新的 MUX 值
    switch (chl)
    {
        case ADS1220_CH0:
            new_mux = MUX_P_AIN0_N_AVSS;
            break;
        case ADS1220_CH1:
            new_mux = MUX_P_AIN1_N_AVSS;
            break;
        case ADS1220_CH2:
            new_mux = MUX_P_AIN2_N_AVSS;
            break;
        case ADS1220_CH3:
            new_mux = MUX_P_AIN3_N_AVSS;
            break;
        default:
            return ADS1220_FAIL;
    }
    
    // 3. 清除旧的 MUX 位 (Bit 7:4) 并设置新的 MUX
    reg0 &= 0x0F;       // 清除高4位
    reg0 |= new_mux;    // 设置新MUX
    
    // 4. 写回 Config 0 寄存器
    if (ADS1220_OK != ADS1220_WriteRegisters(handle, 0, 1, &reg0))
    {
        return ADS1220_FAIL;
    }
    
    return ADS1220_OK;
}

/**
  * @brief  单次转换并读取数据
  * @param  handle: ADS1220句柄指针
  * @param  chl: 通道号（0-3）
  * @retval 32位有符号整数（实际为24位数据）
  */
int32_t ADS1220_ReadConvertOnce(ADS1220_HandleTypeDef *handle, uint8_t chl)
{
    // 选择通道
    if (ADS1220_OK != ADS1220_ChannelSelect(handle, chl))
    {
        return 0;
    }
    
    // 启动转换
    if (ADS1220_OK != ADS1220_Start(handle))
    {
        return 0;
    }
    
    // 检查DRDY信号，如果超时则尝试直接读取
    if (ADS1220_OK != ADS1220_WaitDRDY(handle, 100))
    {
        // DRDY超时，可能是引脚未连接或配置错误
        // 尝试直接读取数据作为调试
        return 0; // 原来的错误处理
    }
    
    // 读取数据
    return ADS1220_ReadData(handle);
}

/* 调试函数 ----------------------------------------------------------------*/

/**
  * @brief  打印ADS1220调试信息
  * @param  data: 24位ADC数据
  * @retval 无
  */
void ADS1220_DebugPrint(int32_t data)
{
    // 计算电压 (假设 VREF=2.048V, Gain=1)
    float voltage = ((float)data / 8388608.0f) * 3.3f;
    
    printf("ADS1220 Raw: 0x%06X (%d), Voltage: %.6f V\r\n", 
           (unsigned int)(data & 0x00FFFFFF), (int)data, voltage);
}

/**
  * @brief  打印配置寄存器值
  * @param  handle: ADS1220句柄指针
  * @retval 无
  */
void ADS1220_DebugPrintRegisters(ADS1220_HandleTypeDef *handle)
{
    uint8_t reg_values[4] = {0};
    
    // 读取所有配置寄存器
    if (ADS1220_OK == ADS1220_ReadRegisters(handle, 0, 4, reg_values))
    {
        printf("ADS1220:\r\n");
        printf("  CONFIG0: 0x%02X\r\n", reg_values[0]);
        printf("  CONFIG1: 0x%02X\r\n", reg_values[1]);
        printf("  CONFIG2: 0x%02X\r\n", reg_values[2]);
        printf("  CONFIG3: 0x%02X\r\n", reg_values[3]);
    }
    else
    {
        printf("ads1220 error\r\n");
    }
}
