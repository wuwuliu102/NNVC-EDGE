# NNVC-EDGE
videocoding_emberged
# NNVC — Neural Network-based Video Coding for Edge Devices

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)

**NNVC** 是一个轻量化的神经网络视频编码框架，专为资源受限的边缘设备（RK3588、树莓派、高通 QCS 等）设计。项目提供了从服务器训练、端侧部署到真实场景实测的完整工具链，通过多 Agent 协作自动完成模型压缩、硬件适配与动态码率调节。

## 核心特性

- **端到端全流程**：覆盖训练、剪枝量化、边缘部署和在线实测闭环。
- **多 Agent 协同**：服务器训练 Agent、端侧部署 Agent、场景实测 Agent 共同工作，自动迭代优化。
- **硬件友好**：支持结构化剪枝、8-bit 量化、算子融合与硬件感知重参数化（RepVCN）。
- **动态编码调节**：实测 Agent 根据视频内容与信道状态实时调整 QP、GOP 等参数。
- **轻量高效**：在 RK3588 上实现 ≤65ms/帧 的解码延迟，模型体积 ≤500KB。


