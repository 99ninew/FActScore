# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LlamaTokenizer

from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM

class CLM(LM):
    def __init__(self, model_name, model_dir, cache_file=None):
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        # 多卡并行加载：使用 HF 的 device_map + bitsandbytes 8bit，避免单卡爆显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        offload_folder = os.path.join(self.model_dir, ".offload")
        os.makedirs(offload_folder, exist_ok=True)

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

        max_memory = None
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpus > 1:
            max_memory = {}
            for i in range(n_gpus):
                total = torch.cuda.get_device_properties(i).total_memory
                # 预留约 10% 余量
                gb = int((total * 0.90) / (1024**3))
                max_memory[i] = f"{max(1, gb)}GiB"
            max_memory["cpu"] = "128GiB"

        # 默认尽量避开 GPU0（检索 encoder 默认会 .cuda() 到 0），但仍然是多卡分片。
        # 可用环境变量覆盖：FACTSCORE_DEVICE_MAP=balanced|balanced_low_0|auto|sequential
        device_map = os.environ.get("FACTSCORE_DEVICE_MAP")
        if not device_map:
            device_map = "balanced_low_0" if n_gpus > 1 else "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=torch.float16,
            offload_folder=offload_folder,
            max_memory=max_memory,
        )
        self.model.eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_dir)

    def _input_device(self):
        # 分片模型没有单一 device；把输入放到“第一个执行设备”即可
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for dev in device_map.values():
                # accelerate may store GPU ids as ints
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                if isinstance(dev, torch.device):
                    return dev
                if isinstance(dev, str) and dev not in ("cpu", "disk", "meta"):
                    # Some versions store GPU ids as digit strings like "0", "1"...
                    if dev.isdigit():
                        return torch.device(f"cuda:{dev}")
                    return torch.device(dev)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]

        input_ids = self.tokenizer(prompts).input_ids
        if verbose:
            input_ids = tqdm(input_ids)

        generations = []
        scores = []
        device = self._input_device()
        for curr_input_ids in input_ids:
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([curr_input_ids]).to(device)
            gen_outputs = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            gen_tokens = gen_outputs["sequences"]
            # saving the logits for the very first token
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            if self.model_name.startswith("llama-sni"):
                gen = gen.split("</s>")[0]
                
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores

