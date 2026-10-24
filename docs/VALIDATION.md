# 软件验证

**简体中文** | [English](VALIDATION.en.md) · [README](../README.md)

验证结果：3 项模型约束测试全部通过，包括有界残差、模态丢弃、掩膜损失与有限梯度。随机缓存生成、一轮 CPU 训练、检查点保存及加载推理流程均已通过。

环境：Windows、Python 3.12.14、NumPy 2.3.5、PyTorch 2.8.0+cpu。使用已有本地依赖；尚未验证全新虚拟环境及 GPU。

合成规模：4 个训练样本、2 个验证样本，32×32 网格。推理输出为 `(1,3,32,32)` 雨量和 `(1,3,5,32,32)` 分类概率。

GitHub Actions 自动检查已在远端通过，见[运行记录](https://github.com/gentle-kai/raincast/actions/runs/40076069460)。随机数据只验证软件能运行，不说明实际预测能力。
