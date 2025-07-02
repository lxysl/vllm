# vLLM Hidden States 提取功能

## 概述

本次修改为 vLLM 添加了在 prefill 阶段获取指定层 hidden states 的功能。该功能通过在 `SamplingParams` 中添加 `prefill_hidden_layer` 参数来实现，支持从任意指定层提取 hidden states，而无需修改具体的模型实现。

## 主要特性

- ✅ **指定层提取**: 通过 `prefill_hidden_layer` 参数指定要提取的层号
- ✅ **模型无关**: 使用 PyTorch forward hook 机制，兼容所有 HuggingFace 风格的模型
- ✅ **仅 prefill 阶段**: 只在 prefill（prompt）阶段提取，不影响 decode 性能
- ✅ **完全兼容**: 不影响现有代码的正常运行
- ✅ **自动清理**: hook 在使用后自动清理，避免内存泄漏
- ✅ **内存优化**: 使用 `detach().clone()` 避免计算图保留，支持 CPU 存储减少显存
- ✅ **参数验证**: 自动验证层号合法性，避免运行时错误
- ✅ **性能优化**: 避免重复注册相同层的 hook

## 修改的文件

### 1. `vllm/sampling_params.py`
- 添加 `prefill_hidden_layer: Optional[int] = None` 字段
- 更新 `from_optional` 方法支持新参数

### 2. `vllm/worker/model_runner.py` (GPU)
- 在 `GPUModelRunnerBase.__init__` 中添加 hook 相关变量
- 添加 `_register_hidden_hook` 方法自动注册 forward hook
- 在 `execute_model` 中添加 hook 逻辑和结果处理

### 3. `vllm/worker/cpu_model_runner.py` (CPU)
- 在 `CPUModelRunnerBase.__init__` 中添加 hook 相关变量
- 添加 `_register_hidden_hook` 方法（与 GPU 版本类似）
- 在 `execute_model` 中添加相应的 hook 逻辑

## 使用方法

### 环境变量

- `VLLM_HIDDEN_STATES_TO_CPU=true`: 将 hidden states 存储到 CPU 内存以减少 GPU 显存使用（默认：false）

### 基础用法

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="your-model-path", trust_remote_code=True)

# 创建包含 prefill_hidden_layer 的 SamplingParams
sampling_params = SamplingParams(
    max_tokens=10,
    temperature=0.7,
    prefill_hidden_layer=5,  # 获取第5层的 hidden states
)

# 执行推理
outputs = llm.generate(["你的 prompt"], sampling_params)

# 获取 hidden states
for output in outputs:
    for completion in output.outputs:
        if hasattr(completion, 'prefill_hidden_states') and completion.prefill_hidden_states is not None:
            hidden_states = completion.prefill_hidden_states
            print(f"Hidden states shape: {hidden_states.shape}")
            print(f"Hidden states dtype: {hidden_states.dtype}")
```

## 技术实现细节

### Hook 机制
- 使用 PyTorch 的 `register_forward_hook` 在指定层注册回调函数
- Hook 函数在模型前向传播时自动调用，捕获该层的输出
- 支持处理 tensor 或 tuple 类型的输出

### 层查找逻辑
为了兼容不同的模型结构，实现了自动层查找机制：
```python
candidates = ["model.layers", "layers", "transformer.h", "transformer.layers"]
```
这覆盖了大多数 HuggingFace 模型的层结构命名。

### 内存管理
- Hook 在每次使用后自动移除，避免累积
- Hidden states 在获取后立即清理，减少内存占用
- 只在需要时才注册 hook，不影响常规推理性能

## 兼容性

### 现有代码兼容性
- 不设置 `prefill_hidden_layer` 时，行为与原来完全一致
- 不会影响现有的 `return_hidden_states` 功能
- 对不需要此功能的用户透明

### 模型兼容性
支持大多数基于 Transformer 架构的模型，包括但不限于：
- LLaMA/LLaMA2
- Qwen/Qwen2  
- ChatGLM
- Baichuan
- 其他 HuggingFace 兼容模型

## 测试

提供了完整的测试脚本 `test_hidden_states_vita.py`，专门针对 `my_vita_server/web_demo/server_vla.py` 中的 VITAQwen2ForCausalLM 模型进行测试。

运行测试：
```bash
python test_hidden_states_vita.py
```

测试内容包括：
- 不同层的 hidden states 提取
- 兼容性测试（确保不影响正常功能）
- 边界情况处理

## 注意事项

1. **层索引**: 层号从0开始，确保指定的层号在模型层数范围内
2. **性能影响**: 仅在 prefill 阶段有轻微性能影响，decode 阶段无影响
3. **内存使用**: hidden states 会增加内存使用，建议及时处理和释放
4. **CUDA Graph**: 建议设置 `enforce_eager=True` 以避免与 hook 机制的潜在冲突

## 故障排除

### 常见问题

1. **找不到层**: 如果出现 "Could not find layer list" 警告，可能是模型结构不在预期范围内
2. **Hook 不生效**: 确保使用 `enforce_eager=True` 禁用 CUDA Graph
3. **内存泄漏**: Hook 应该自动清理，如果发现内存问题，请检查是否有异常中断

### 调试建议

在测试新模型时，建议：
1. 先用小层号（如0, 1, 2）测试
2. 检查模型的具体层结构：`print(model.model.layers)` 或类似
3. 使用小的 `max_tokens` 值进行初步测试

## 未来扩展

该实现为后续功能扩展提供了基础，可能的扩展包括：
- 支持多层同时提取
- 支持 decode 阶段的 hidden states（如果需要）
- 更精细的内存管理选项
- 与现有 `aux_hidden_states` 功能的集成 