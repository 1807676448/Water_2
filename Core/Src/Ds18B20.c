#include "Ds18B20.h"
#include "stm32g4xx_hal.h"
#include "tim.h"
#include <stdio.h>

/**************************************************************************************
 * 描  述 : 配置使DS18B20-DATA引脚变为输入模式
 * 入  参 : 无
 * 返回值 : 无
 **************************************************************************************/
static void DS18B20_Mode_IPU(void)
{
	GPIO_InitTypeDef GPIO_InitStruct = {0};

	GPIO_InitStruct.Pin = DS18B20_PIN;
	GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
	GPIO_InitStruct.Pull = GPIO_PULLUP;
	HAL_GPIO_Init(DS18B20_PORT, &GPIO_InitStruct);
}

/**************************************************************************************
 * 描  述 : 配置使DS18B20-DATA引脚变为输出模式
 * 入  参 : 无
 * 返回值 : 无
 **************************************************************************************/
static void DS18B20_Mode_Out_PP(void)
{
	GPIO_InitTypeDef GPIO_InitStruct = {0};

	GPIO_InitStruct.Pin = DS18B20_PIN;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP; // 推挽输出
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
	HAL_GPIO_Init(DS18B20_PORT, &GPIO_InitStruct);
}

/**************************************************************************************
 * 描  述 : 主机给从机发送复位脉冲
 * 入  参 : 无
 * 返回值 : 无
 **************************************************************************************/
static void DS18B20_Rst(void)
{
	DS18B20_Mode_Out_PP(); // 主机设置为推挽输出

	DS18B20_DATA_OUT(GPIO_PIN_RESET); // 主机至少产生480us的低电平复位信号
	HAL_Delay_Us(750);
	DS18B20_DATA_OUT(GPIO_PIN_SET); // 主机在产生复位信号后，需将总线拉高
	HAL_Delay_Us(15);				// 从机接收到主机的复位信号后，会在15~60us后给主机发一个存在脉冲
}

/**************************************************************************************
 * 描  述 : 检测从机给主机返回的存在脉冲
 * 入  参 : 无
 * 返回值 : 0：成功   1：失败
 **************************************************************************************/
static uint8_t DS18B20_Presence(void)
{
	uint8_t pulse_time = 0;

	DS18B20_Mode_IPU(); // 主机设置为上拉输入

	/* 等待存在脉冲的到来，存在脉冲为一个60~240us的低电平信号
	 * 如果存在脉冲没有来则做超时处理，从机接收到主机的复位信号后，会在15~60us后给主机发一个存在脉冲
	 */
	while (DS18B20_DATA_IN() && pulse_time < 100)
	{
		pulse_time++;
		HAL_Delay_Us(1);
	}

	if (pulse_time >= 100) // 经过100us后，存在脉冲都还没有到来
		return 1;		   // 读取失败
	else				   // 经过100us后，存在脉冲到来
		pulse_time = 0;	   // 清零计时变量

	while (!DS18B20_DATA_IN() && pulse_time < 240) // 存在脉冲到来，且存在的时间不能超过240us
	{
		pulse_time++;
		HAL_Delay_Us(1);
	}
	if (pulse_time >= 240) // 存在脉冲到来，且存在的时间超过了240us
		return 1;		   // 读取失败
	else
		return 0;
}

/**************************************************************************************
 * 描  述 : 从DS18B20读取一个bit
 * 入  参 : 无
 * 返回值 : u8
 **************************************************************************************/
static uint8_t DS18B20_Read_Bit(void)
{
	uint8_t dat;

	/* 读0和读1的时间至少要大于60us */
	DS18B20_Mode_Out_PP();
	/* 读时间的起始：必须由主机产生 >1us <15us 的低电平信号 */
	DS18B20_DATA_OUT(GPIO_PIN_RESET);
	HAL_Delay_Us(10);

	/* 设置成输入，释放总线，由外部上拉电阻将总线拉高 */
	DS18B20_Mode_IPU();

	if (DS18B20_DATA_IN() == GPIO_PIN_SET)
		dat = 1;
	else
		dat = 0;

	/* 这个延时参数请参考时序图 */
	HAL_Delay_Us(45);

	return dat;
}

/**************************************************************************************
 * 描  述 : 从DS18B20读一个字节，低位先行
 * 入  参 : 无
 * 返回值 : u8
 **************************************************************************************/
uint8_t DS18B20_Read_Byte(void)
{
	uint8_t i, j, dat = 0;

	for (i = 0; i < 8; i++)
	{
		j = DS18B20_Read_Bit(); // 从DS18B20读取一个bit
		dat = (dat) | (j << i);
	}

	return dat;
}

/**************************************************************************************
 * 描  述 : 写一个字节到DS18B20，低位先行
 * 入  参 : u8
 * 返回值 : 无
 **************************************************************************************/
void DS18B20_Write_Byte(uint8_t dat)
{
	uint8_t i, testb;
	DS18B20_Mode_Out_PP();

	for (i = 0; i < 8; i++)
	{
		testb = dat & 0x01;
		dat = dat >> 1;
		/* 写0和写1的时间至少要大于60us */
		if (testb)
		{
			DS18B20_DATA_OUT(GPIO_PIN_RESET);
			HAL_Delay_Us(8); // 1us < 这个延时 < 15us

			DS18B20_DATA_OUT(GPIO_PIN_SET);
			HAL_Delay_Us(58); // 58us+8us>60us
		}
		else
		{
			DS18B20_DATA_OUT(GPIO_PIN_RESET);
			/* 60us < Tx 0 < 120us */
			HAL_Delay_Us(70);

			DS18B20_DATA_OUT(GPIO_PIN_SET);
			/* 1us < Trec(恢复时间) < 无穷大*/
			HAL_Delay_Us(2);
		}
	}
}

/**************************************************************************************
 * 描  述 : 起始DS18B20
 * 入  参 : 无
 * 返回值 : 无
 **************************************************************************************/
void DS18B20_Start(void)
{
	DS18B20_Rst();			  // 主机给从机发送复位脉冲
	DS18B20_Presence();		  // 检测从机给主机返回的存在脉冲
	DS18B20_Write_Byte(0XCC); // 跳过 ROM
	DS18B20_Write_Byte(0X44); // 开始转换
}

/**************************************************************************************
 * 描  述 : DS18B20初始化函数
 * 入  参 : 无
 * 返回值 : u8
 **************************************************************************************/
uint8_t DS18B20_Init(void)
{
	// DS18B20_GPIO_Config();
	DS18B20_Rst();

	return DS18B20_Presence();
}

/**************************************************************************************
 * 描  述 : 从DS18B20读取温度值
 * 入  参 : 无
 * 返回值 : float
 **************************************************************************************/
float DS18B20_Get_Temp(void)
{
	uint8_t tpmsb, tplsb;
	short s_tem;
	float f_tem;

	DS18B20_Rst();
	DS18B20_Presence();
	DS18B20_Write_Byte(0XCC); /* 跳过 ROM */
	DS18B20_Write_Byte(0X44); /* 开始转换 */

	DS18B20_Rst();
	DS18B20_Presence();
	DS18B20_Write_Byte(0XCC); /* 跳过 ROM */
	DS18B20_Write_Byte(0XBE); /* 读温度值 */

	tplsb = DS18B20_Read_Byte();
	tpmsb = DS18B20_Read_Byte();

	s_tem = tpmsb << 8;
	s_tem = s_tem | tplsb;

	if (s_tem < 0) /* 负温度 */
		f_tem = (~s_tem + 1) * 0.0625;
	else
		f_tem = (s_tem * 0.0625);

	// 这样做的目的将小数点后第一位也转换为可显示数字
	// 同时进行一个四舍五入操作。

	// printf("DS18B20 Tem is %.6f\n\r", f_tem);
	return f_tem;
}

/*************************************END OF FILE******************************/
