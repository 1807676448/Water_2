#include "Led.h"

uint32_t led254_blink(uint16_t time)
{
    uint32_t adc_data;

    HAL_TIM_PWM_ConfigChannel(&htim1, &(TIM_OC_InitTypeDef){.OCMode = TIM_OCMODE_PWM1, .Pulse = 1000}, TIM_CHANNEL_3);
    HAL_Delay(LedDelayTime);
    adc_data = ADS1220_ReadConvertOnce(&hads1220, 1);
    // ADS1220_DebugPrint(adc_data);
    HAL_Delay(LedDelayTime);
    HAL_TIM_PWM_ConfigChannel(&htim1, &(TIM_OC_InitTypeDef){.OCMode = TIM_OCMODE_PWM1, .Pulse = 0}, TIM_CHANNEL_3);

    return adc_data;
}

uint32_t led550_blink(uint16_t Pulse)
{
    uint32_t adc_data;

    HAL_TIM_PWM_ConfigChannel(&htim1, &(TIM_OC_InitTypeDef){.OCMode = TIM_OCMODE_PWM1, .Pulse = Pulse}, TIM_CHANNEL_2);
    HAL_Delay(LedDelayTime);
    adc_data = ADS1220_ReadConvertOnce(&hads1220, 0);
    // ADS1220_DebugPrint(adc_data);
    HAL_Delay(LedDelayTime);
    HAL_TIM_PWM_ConfigChannel(&htim1, &(TIM_OC_InitTypeDef){.OCMode = TIM_OCMODE_PWM1, .Pulse = 0}, TIM_CHANNEL_2);

    return adc_data;
}
