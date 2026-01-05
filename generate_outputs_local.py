#!/usr/bin/env python
"""
使用本地模型生成输出并保存为 FActScore 格式的 .jsonl 文件

支持：
1. vLLM 本地部署
2. OpenAI 兼容的本地服务（如 llama.cpp server, FastChat）

使用方法:
python generate_outputs_local.py --input example_topics.txt --model_type vllm \
--model_path ~/CKM/ckpts/1203/test_mix_wikidata_3+9_shuffle-quality-append-lr2e-05-rr0.9-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-3562

python generate_outputs_local.py --input topics.txt --output outputs.jsonl --model_type vllm --model_path /path/to/model
"""

import json
import argparse
from tqdm import tqdm
import os
import logging
import contextlib

# 抑制库的日志输出
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("chardet").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def load_topics(input_file):
    """从 .txt 文件加载 topic 列表（每行一个实体名）"""
    topics = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            topic = line.strip()
            if topic:
                topics.append(topic)
    return topics

class LocalModelWrapper:    
    def __init__(self, model_type, model_path, tokenizer_path=None, **kwargs):
        self.model_type = model_type
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        if model_type == "vllm":
            self._init_vllm(**kwargs)
        elif model_type == "openai_compatible":
            self._init_openai_compatible(**kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def _init_vllm(self, tensor_parallel_size=1, gpu_memory_utilization=0.9, **kwargs):
        """初始化 vLLM 模型"""
        print("Importing vllm module...")
        try:
            from vllm import LLM, SamplingParams
            print("Imported vllm module successfully")
        except Exception as e:
            print(f"Failed to import vllm: {e}")
            raise
        
        # 展开 ~ 路径，转换为绝对路径，并解析任何符号链接
        model_path = os.path.realpath(os.path.expanduser(self.model_path))
        print(f"Loading vLLM model from {model_path}...")
        
        # 确保路径存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # 构建 vLLM 参数
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        }
        
        # 如果提供了单独的 tokenizer 路径，也展开并验证
        if self.tokenizer_path:
            tokenizer_path = os.path.realpath(os.path.expanduser(self.tokenizer_path))
            print(f"Using custom tokenizer from {tokenizer_path}...")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")
            llm_kwargs["tokenizer"] = tokenizer_path
        
        print("Constructing LLM object (this may take a while)...")
        try:
            self.llm = LLM(**llm_kwargs)
            print("LLM constructed successfully")
        except Exception as e:
            print(f"Failed to construct LLM: {e}")
            raise
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=200
        )
        print("vLLM model loaded successfully!")
    
    def _init_openai_compatible(self, base_url="http://localhost:8000/v1", api_key="EMPTY", **kwargs):
        """初始化 OpenAI 兼容的本地服务"""
        from openai import OpenAI
        
        print(f"Connecting to OpenAI-compatible server at {base_url}...")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = kwargs.get("model_name", "default")
        print("Connected successfully!")
    
    def generate(self, prompt, max_tokens=200):
        if self.model_type == "vllm":
            return self._generate_vllm(prompt, max_tokens)
        elif self.model_type == "openai_compatible":
            return self._generate_openai_compatible(prompt, max_tokens)
    
    def _generate_vllm(self, prompt, max_tokens=200):
        """vLLM 生成"""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.4,
            top_p=0.9,
            max_tokens=max_tokens
        )

        # Temporarily suppress stdout/stderr from vLLM internals (e.g. "Adding requests / Processed prompts")
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                outputs = self.llm.generate([prompt], sampling_params=sampling_params)

        return outputs[0].outputs[0].text
    
    def _generate_openai_compatible(self, prompt, max_tokens=200):
        """OpenAI 兼容服务生成"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content

def generate_biography(model, topic):
    # prompt = f"Please provide a detailed and comprehensive biography of {topic}, including their early life, education, career, major achievements, personal life, and any significant events that shaped their journey. You need to answer it in 200 tokens."
    # prompt = f"""
    # Write a concise biographical paragraph about {topic}. 
    # The biography should mention their early life, education, career, major achievements, and important life events. 
    # The length should be around 200 tokens.

    # Biography of {topic}:
    # """
    prompt = f"""
    The following is a biography about {topic}.
    The biography mentions their early life, education, career, major achievements, and important life events. 
    The length is around 200 tokens.

    Biography:
    {topic}
    """
    output = model.generate(prompt)
    return output

def main():
    parser = argparse.ArgumentParser(description='Generate model outputs using local models')
    parser.add_argument('--input', type=str, default=".cache/factscore/data/labeled/prompt_entities.txt")

    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['vllm', 'openai_compatible'],
                        help='Type of local model')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to model (for vllm/transformers) or model name (for openai_compatible)')
    parser.add_argument('--tokenizer_path', type=str, default="~/CKM/model/Meta-Llama-3-8B",
                        help='(Optional) Separate path to tokenizer for vLLM')
    
    # vLLM 
    parser.add_argument('--tensor_parallel_size', type=int, default=8, help='Tensor parallel size for vLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.7, help='GPU memory utilization for vLLM')
    
    # OpenAI 兼容服务参数
    parser.add_argument('--base_url', type=str, default='http://localhost:8000/v1', 
                        help='Base URL for OpenAI-compatible server')
    parser.add_argument('--api_key', type=str, default='EMPTY', help='API key for OpenAI-compatible server')
    
    # 生成参数
    parser.add_argument('--prompt_template', type=str, default=None,
                        help='Custom prompt template (use {topic} as placeholder)')
    parser.add_argument('--max_tokens', type=int, default=200, help='Max tokens to generate')
    
    args = parser.parse_args()
    
    # ~/CKM/ckpts/1115/test_mix.../checkpoint-396 -> test_mix...
    path_parts = os.path.expanduser(args.model_path).strip("/").split("/")
    experiment_name = path_parts[-2]
    output_path = f'out-less/Bio-{experiment_name}.jsonl'
    # output_path = 'example_o.jsonl'
    
    print(f"Output will be saved to: {output_path}")
    
    # 加载 topics
    print(f"Loading topics from {args.input}...")
    topics = load_topics(args.input)
    print(f"Found {len(topics)} topics")
    
    # 初始化模型
    model_kwargs = {
        'tensor_parallel_size': args.tensor_parallel_size,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'base_url': args.base_url,
        'api_key': args.api_key,
        'model_name': args.model_path  # for openai_compatible
    }
    
    model = LocalModelWrapper(args.model_type, args.model_path, tokenizer_path=args.tokenizer_path, **model_kwargs)
    
    results = []
    print(f"Generating biographies...")
    for topic in tqdm(topics, desc="Generating", unit="bio"):
        try:
            output = generate_biography(model, topic)
            output_filtered = output
            
            # to avoid the base model simply continue writing the prompt at the beginning
            # if '\n' in output:
            #     output_filtered = output.split('\n', 1)[1].strip()
                
            #     if not output_filtered.startswith(topic):
            #         topic_idx = output.find(topic)
            #         if topic_idx != -1:
            #             output_filtered = output[topic_idx:].strip()
            
            results.append({
                "topic": topic,
                "output": output_filtered
            })
        except Exception as e:
            print(f"\nError generating bio for {topic}: {e}")
            # store the finished part
            if results:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
            raise
    
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Done! Generated {len(results)} biographies")
    print(f"\nNext step: Run FActScore evaluation:")
    print(f"python -m factscore.factscorer --input_path {output_path} --model_name retrieval+ChatGPT --verbose")

if __name__ == "__main__":
    main()
