#ifndef __ALLSET_H
#define __ALLSET_H

#include "main.h"
#include "adc.h"
#include "spi.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"
#include <stdio.h>
#include <string.h>
#include "Led.h"
#include "Ds18B20.h"

#define LedBlinkTime 50
#define BoomTime 1000
#define BoardLedTime 300

// ================== 配置宏定义 ==================
#define COMM_DEVICE_ID      2           // 设备ID
#define COMM_UART_HANDLE    &huart2     // 通信使用的串口句柄
#define COMM_RX_BUFFER_SIZE 256         // 接收缓冲区大小


void LedTest(void);
void BoomTest(float* out_cod, float* out_uv254);
void BoardLedTest(void);


#endif