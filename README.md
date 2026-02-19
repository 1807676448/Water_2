# Water_2 水质检测与 AI 预测固件工程

基于 `STM32G431RBTx` 的水质检测固件，集成以下能力：

- 双光路采样（254nm / 550nm）
- 温度采集（DS18B20）
- ADS1220 高精度 ADC 采样
- 串口通信（指令接收 + JSON 状态上报）
- 端侧神经网络推理（输出 COD 与 UV254）

---

## 1. 工程定位

本仓库是**设备端固件工程**，主路径在：

- `Core/`：业务代码与 HAL 回调
- `Drivers/`：STM32 HAL / CMSIS
- `MDK-ARM/`：Keil 与 EIDE 构建工程文件

同时，本地存在一个并列目录 `d:\STM32\Water_Python`（不在本仓库内），用于离线训练与模型导出。

---

## 2. 当前功能状态（按代码实现）

### 2.1 主循环行为

主循环中：

- 每 10 秒执行一次 `BoomTest()`，进行一次完整采样 + AI 推理 + JSON 上报
- 每 1 秒发送一次心跳 JSON（`status=Active`）
- 板载 LED 周期闪烁

### 2.2 指令接收状态机

USART3 采用单字节中断 + 状态机收包，包格式：

```text
AA [INS1] [INS2] [CHK] 55
```

- 包头：`0xAA`
- 指令长度：2 字节
- 校验：`INS1 ^ INS2`
- 包尾：`0x55`

示例：`AA 01 02 03 55`（`0x01 ^ 0x02 = 0x03`）。

> 注：当前实现中，命令校验通过后仅置位 `TestStart=1`；实际测量仍由“10 秒定时路径”触发。

### 2.3 串口输出 JSON

心跳（约 1Hz）：

```json
{"device_id":2,"status":"Active"}
```

测量结果（约 10s 一次）：

```json
{
  "Led_550": 0.123456,
  "Led_254": 0.123456,
  "Temp": 25.1234,
  "COD": 12.3456,
  "UV254": 0.1234,
  "device_id": 2,
  "status": "Active"
}
```

---

## 3. 目录结构（建议理解顺序）

```text
Water1/
├─ Core/
│  ├─ Inc/
│  │  ├─ main.h
│  │  ├─ ADS1220.h
│  │  ├─ Ds18B20.h
│  │  ├─ Led.h
│  │  ├─ AllSet.h
│  │  ├─ water_quality_ai.h
│  │  └─ model_data.h
│  └─ Src/
│     ├─ main.c
│     ├─ ADS1220.c
│     ├─ Ds18B20.c
│     ├─ Led.c
│     ├─ AllSet.c
│     └─ water_quality_ai.c
├─ Drivers/
├─ MDK-ARM/
│  ├─ Water_2.uvprojx
│  ├─ Water_2.uvoptx
│  ├─ build/
│  └─ Water_2/            # Keil中间文件目录
├─ model_data.h           # 根目录存在历史副本（见“模型文件说明”）
├─ Water_2.ioc
└─ training_log.txt
```

---

## 4. 核心模块说明

### 4.1 `main.c`

- 完成系统时钟、GPIO、SPI、TIM、USART 初始化
- 启动 `TIM6/TIM7/TIM1`、PWM 输出、串口中断接收
- 实现 UART 命令状态机与校验
- 周期触发采样与心跳

### 4.2 `AllSet.c`

- 封装 `BoomTest()`：
  1. 控制外设使能
  2. 多次采样 `led550_blink()/led254_blink()` 与温度
  3. 电压换算与平均
  4. 调用 `WaterQuality_Predict()`
  5. 组包 JSON 并通过 USART3 发送

### 4.3 `ADS1220.c`

- 完成 ADS1220 的 SPI 命令、寄存器配置、单次转换读取
- 数据为 24bit，内部做符号扩展到 32bit

### 4.4 `water_quality_ai.c`

- 三层全连接网络（3→32→16→2）
- ReLU 激活
- 使用 `model_data.h` 中权重与标准化参数
- 输出 `COD` 与 `UV254`

---

## 5. 硬件与引脚映射（依据当前源码）

### 5.1 串口

- `USART1`：PC4(TX), PC5(RX), 115200（调试打印）
- `USART3`：PB10(TX), PB11(RX), 115200（通信与数据上报）

### 5.2 SPI

- `SPI1`：PA5(SCK), PA6(MISO), PA7(MOSI)
- ADS1220 控制脚：
  - `AD_CS`：PA4
  - `AD_DRDY`：PA3

### 5.3 PWM / 控制

- `TIM1_CH2`：PC1（OC7140_1）
- `TIM1_CH3`：PC2（OC7140_2）
- `Boom` 控制脚：PC12
- 板载 LED：PA12

### 5.4 温度

- DS18B20 数据脚：PA15（单总线）

---

## 6. 构建、下载与调试

### 6.1 VS Code + EIDE（推荐）

当前工作区已定义任务：

- `build`
- `flash`
- `build and flash`
- `rebuild`
- `clean`

可直接通过 VS Code 任务面板执行。

### 6.2 Keil MDK

- 打开 `MDK-ARM/Water_2.uvprojx`
- 选择目标配置后编译、下载

---

## 7. 模型文件说明（非常重要）

当前仓库中存在两份 `model_data.h`：

- `Core/Inc/model_data.h`（编译包含路径中的实际使用版本）
- 根目录 `model_data.h`（历史副本）

建议仅维护 `Core/Inc/model_data.h` 为唯一真源，避免训练后拷贝错文件导致“模型已更新但固件结果未变化”。

---

## 8. 与 Python 训练链路的衔接

离线训练目录（本地并列目录）建议流程：

1. 训练模型（`new.py`）
2. 导出参数头文件（`export_to_c.py`）
3. 将导出的 `model_data.h` 覆盖到本仓库 `Core/Inc/model_data.h`
4. 重新编译并下载固件验证

---

## 9. 常见问题

### Q1：串口收到命令，但设备没有立刻测量？

当前版本测量主路径由 10 秒定时分支触发；指令置位仅是预留逻辑。若需“命令即测量”，请在主循环恢复 `TestStart` 分支并与 10 秒分支协调。

### Q2：模型替换后预测没变化？

优先检查是否更新了 `Core/Inc/model_data.h`，而不是根目录副本。

### Q3：串口只见心跳不见结果？

确认 `BoomTest()` 没有被外设阻塞（如温度/ADC读数异常、串口发送超时）。

---

## 10. 建议的提交与维护规范

- 不提交 `MDK-ARM/build/` 与 `MDK-ARM/Water_2/` 下中间产物
- 不提交 Python 虚拟环境、缓存、训练中间文件
- 版本发布时记录：固件版本、模型版本、数据集版本

仓库已提供 `.gitignore` 作为默认过滤规则。

---

## 11. 参考文档

- 工程整理与归档建议：`docs/工程整理说明.md`
