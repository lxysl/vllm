#!/usr/bin/env python3
"""
测试脚本：为 my_vita_server/web_demo/server_vla.py 中的 VITAQwen2ForCausalLM 模型
提供获取指定层 hidden states 的测试

该脚本演示如何使用新添加的 prefill_hidden_layer 参数来获取模型指定层的 hidden states。
注意：这里只针对 prefill 阶段，不包含 decode/generate 阶段的 hidden states。
"""

import torch
from vllm import LLM, SamplingParams

def test_hidden_states_extraction():
    """测试在 prefill 阶段获取指定层的 hidden states"""
    
    # 从 server_vla.py 复制的模型路径
    model_path = "/home/vita/.cache/huggingface/hub/models--VITA-MLLM--VITA-1.5-2B/snapshots/6df7b71c5e1c5e3b0aed89b3de26e9ef43de4d24"
    
    print("正在初始化 vLLM 引擎...")
    try:
        # 使用与 server_vla.py 相同的配置初始化 LLM
        llm = LLM(
            model=model_path,
            # 注意：从 server_vla.py 复制的设置
            trust_remote_code=True,
            max_num_seqs=256,
            max_model_len=8192,
            enforce_eager=True,  # 确保不使用 CUDA Graph，避免与 hook 冲突
        )
        print("✓ vLLM 引擎初始化成功")
    except Exception as e:
        print(f"✗ vLLM 引擎初始化失败: {e}")
        return

    # 测试用的 prompt
    test_prompt = "请介绍一下人工智能的发展历史。"
    
    # 测试不同层的 hidden states 获取
    test_layers = [0, 5, 10, 15, 20]  # 假设模型至少有21层
    
    for layer_idx in test_layers:
        print(f"\n测试获取第 {layer_idx} 层的 hidden states...")
        
        try:
            # 创建 SamplingParams，指定要获取的层
            sampling_params = SamplingParams(
                max_tokens=1,  # 只生成一个 token，专注于 prefill 阶段
                temperature=0.0,  # 确定性输出
                prefill_hidden_layer=layer_idx,  # 指定要获取的层
            )
            
            # 执行推理
            outputs = llm.generate([test_prompt], sampling_params)
            
            # 检查是否成功获取到 hidden states
            for output in outputs:
                for completion in output.outputs:
                    if hasattr(completion, 'prefill_hidden_states') and completion.prefill_hidden_states is not None:
                        hidden_states = completion.prefill_hidden_states
                        print(f"  ✓ 成功获取第 {layer_idx} 层 hidden states")
                        print(f"    形状: {hidden_states.shape}")
                        print(f"    数据类型: {hidden_states.dtype}")
                        print(f"    设备: {hidden_states.device}")
                        print(f"    数值范围: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
                        
                        # 验证 hidden states 不全为零
                        if torch.all(hidden_states == 0):
                            print("  ⚠ 警告: hidden states 全为零，可能有问题")
                        else:
                            print("  ✓ hidden states 包含非零值")
                    else:
                        print(f"  ✗ 未能获取第 {layer_idx} 层的 hidden states")
                        
        except Exception as e:
            print(f"  ✗ 第 {layer_idx} 层测试失败: {e}")
    
    # 测试不指定层的情况（应该使用默认行为）
    print(f"\n测试不指定 prefill_hidden_layer 的默认行为...")
    try:
        sampling_params_default = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            # 不设置 prefill_hidden_layer
        )
        
        outputs = llm.generate([test_prompt], sampling_params_default)
        for output in outputs:
            for completion in output.outputs:
                if hasattr(completion, 'prefill_hidden_states'):
                    if completion.prefill_hidden_states is not None:
                        print("  ✓ 默认情况下获取到了 hidden states（可能是最后一层）")
                    else:
                        print("  ✓ 默认情况下未获取 hidden states（符合预期）")
                else:
                    print("  ✓ 默认情况下没有 prefill_hidden_states 属性（符合预期）")
    except Exception as e:
        print(f"  ✗ 默认行为测试失败: {e}")
    
    print("\n测试完成！")
    
def test_compatibility():
    """测试兼容性：确保不使用新功能时模型仍能正常工作"""
    
    model_path = "/home/vita/.cache/huggingface/hub/models--VITA-MLLM--VITA-1.5-2B/snapshots/6df7b71c5e1c5e3b0aed89b3de26e9ef43de4d24"
    
    print("\n=== 兼容性测试 ===")
    print("测试不使用 prefill_hidden_layer 时的正常推理...")
    
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_num_seqs=256,
            max_model_len=8192,
            enforce_eager=True,
        )
        
        # 标准的推理测试
        sampling_params = SamplingParams(
            max_tokens=50,
            temperature=0.7,
            top_k=50,
        )
        
        test_prompt = "人工智能的未来发展趋势包括"
        outputs = llm.generate([test_prompt], sampling_params)
        
        for output in outputs:
            prompt = output.prompt
            for completion in output.outputs:
                generated_text = completion.text
                print(f"输入: {prompt}")
                print(f"输出: {generated_text}")
                print("✓ 兼容性测试通过：标准推理功能正常")
                
    except Exception as e:
        print(f"✗ 兼容性测试失败: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("VITAQwen2ForCausalLM Hidden States 提取测试")
    print("=" * 60)
    
    # 运行 hidden states 提取测试
    test_hidden_states_extraction()
    
    # 运行兼容性测试
    test_compatibility()
    
    print("\n" + "=" * 60)
    print("测试脚本运行完成")
    print("=" * 60) 