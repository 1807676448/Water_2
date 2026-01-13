#ifndef __LED_H
#define __LED_H

#include "stm32g4xx_hal.h"
#include "main.h"
#include "adc.h"
#include "gpio.h"
#include <stdio.h>
#include <string.h>
#include "Led.h"
#include "Ds18B20.h"
#include "AllSet.h"
#include "ADS1220.h"

extern ADS1220_HandleTypeDef hads1220;

#define LedDelayTime 150

uint32_t led254_blink(uint16_t time);
uint32_t led550_blink(uint16_t time);

#endif