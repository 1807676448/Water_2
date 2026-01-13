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

void LedTest(void);
void BoomTest(float* out_cod, float* out_uv254);
void BoardLedTest(void);


#endif