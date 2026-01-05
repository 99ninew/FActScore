from factscore.lm import LM
import openai
from openai import OpenAI
import sys
import time
import os
import numpy as np
import logging

DEFAULT_BASE_URL="https://cloud.infini-ai.com/maas/v1"

# Use a separate logger for OpenAI API calls to avoid conflicts
logger = logging.getLogger("OpenAIModel")

class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        # Build client (OpenAI SDK >=1.x) and keep model name
        self.client = OpenAI(api_key=api_key.strip(), base_url=DEFAULT_BASE_URL)
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        # if self.add_n % self.save_interval == 0:
        #     self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "gpt-oss-120b":
            message = [{"role": "user", "content": prompt}]
            # message = self.tokenizer.apply_chat_template(
            #             message,
            #             add_generation_prompt=True,
            #             tokenize=False,
            #             model_identity="You are OpenAI GPT OSS.",
            #             reasoning_effort="high" # long system prompt eg. "step-by-step"
            #             )
            response = call_gptoss(self.client, message, model_name="gpt-oss-120b", temp=self.temp, max_len=max_sequence_length)
            
            if response is None:
                raise RuntimeError(f"API returned None for model {self.model_name}")
            output = response["choices"][0]["message"]["content"]
            
            return output, response
        elif self.model_name == "deepseek-v3.2":
            message = [{"role": "user", "content": prompt}]
            try:
                response = call_deepseek(self.client, message, model_name="deepseek-v3.2", temp=self.temp, max_len=max_sequence_length)
                
                if response is None:
                    logger.error(f"API returned None response for model {self.model_name}")
                    raise RuntimeError(f"API returned None for model {self.model_name}")
                
                output = response["choices"][0]["message"]["content"]
                return output, response
            except Exception as e:
                logger.error(f"Error in deepseek API call: {type(e).__name__}: {str(e)}")
                raise
        elif self.model_name == "ChatGPT":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(self.client, message, temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            output = response["choices"][0]["message"]["content"]
            return output, response
        elif self.model_name == "InstructGPT":
            # Call API
            response = call_GPT3(self.client, prompt, temp=self.temp)
            # Get the output from the response
            output = response["choices"][0]["text"]
            return output, response
        else:
            raise NotImplementedError()
        
def call_gptoss(client, message, model_name="gpt-oss-120b", max_len=1024, temp=0.7, verbose=False, max_retries=2):
    # call deepseek API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    invalid_req_err = getattr(openai, "BadRequestError", None) or getattr(openai, "InvalidRequestError", None) or getattr(getattr(openai, "error", None), "InvalidRequestError", None)
    while not received and num_rate_errors < max_retries:
        try:
            response = client.chat.completions.create(model=model_name,
                                                      messages=message,
                                                      max_tokens=max_len,
                                                      temperature=temp,
                                                      timeout=120) 
            response = response.model_dump()
            received = True
        except Exception as exc:
            num_rate_errors += 1
            if invalid_req_err is not None and isinstance(exc, invalid_req_err):
                # something is wrong: e.g. prompt too long
                logger.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                raise
            
            logger.error("API error: %s (%d). Waiting %dsec" % (exc, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    
    if not received:
        logger.error(f"Failed to get response for model {model_name} after {max_retries} retries")
        raise RuntimeError(f"Failed to get response after {max_retries} retries for model {model_name}")
    
    if response is None:
        logger.error(f"Response is None for model {model_name} despite received=True")
        raise RuntimeError(f"Got None response for model {model_name}")
    
    return response
        
def call_deepseek(client, message, model_name="deepseek-v3.2", max_len=1024, temp=0.7, verbose=False, max_retries=2):
    # call deepseek API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    invalid_req_err = getattr(openai, "BadRequestError", None) or getattr(openai, "InvalidRequestError", None) or getattr(getattr(openai, "error", None), "InvalidRequestError", None)
    logger.debug(f"Calling API with model={model_name}, max_tokens={max_len}, temp={temp}")
    
    while not received and num_rate_errors < max_retries:
        try:
            logger.info(f"[API] Sending request to {model_name}...")
            raw_response = client.chat.completions.create(model=model_name,
                                                      messages=message,
                                                      max_tokens=max_len,
                                                      temperature=temp,
                                                      timeout=120)  # 设置 120 秒超时
            logger.info(f"[API] Received raw response type: {type(raw_response)}")
            
            # 检查原始响应
            if raw_response is None:
                logger.error(f"[API] Raw response is None! Retrying...")
                num_rate_errors += 1
                time.sleep(2 ** num_rate_errors)
                continue
            
            # 尝试转换为字典
            try:
                response = raw_response.model_dump()
                logger.info(f"[API] Successfully converted to dict, keys: {list(response.keys()) if response else 'empty'}")
            except Exception as dump_err:
                logger.error(f"[API] model_dump() failed: {type(dump_err).__name__}: {str(dump_err)}")
                num_rate_errors += 1
                time.sleep(2 ** num_rate_errors)
                continue
            
            # 最后检查转换后的响应
            if response is None:
                logger.error(f"[API] Response dict is None after model_dump()! Retrying...")
                num_rate_errors += 1
                time.sleep(2 ** num_rate_errors)
                continue
            
            logger.info(f"[API] Successfully received valid response from {model_name}")
            received = True
            
        except Exception as exc:
            num_rate_errors += 1
            exc_type = type(exc).__name__
            exc_msg = str(exc)
            logger.warning(f"[API] Exception {exc_type}: {exc_msg}")
            
            if invalid_req_err is not None and isinstance(exc, invalid_req_err):
                # something is wrong: e.g. prompt too long
                logger.critical(f"[API] InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                raise
            
            # Cap wait time at 60 seconds to avoid excessive delays
            wait_time = min(np.power(2, num_rate_errors), 10)
            logger.error(f"[API] Error: {exc_type} ({num_rate_errors}/{max_retries}). Waiting {wait_time}sec...")
            time.sleep(wait_time)
    
    if not received:
        logger.error(f"[API] Failed after {max_retries} retries. Last model: {model_name}")
        raise RuntimeError(f"Failed to get response after {max_retries} retries")
    
    if response is None:
        logger.error(f"[API] FATAL: Response is None for model {model_name} despite received=True!")
        logger.error(f"[API] This should not happen - check the logs above for details")
        raise RuntimeError(f"Got None response for model {model_name}")
    
    logger.info(f"[API] Returning valid response from {model_name}")
    return response

def call_ChatGPT(client, message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    invalid_req_err = getattr(openai, "BadRequestError", None) or getattr(openai, "InvalidRequestError", None) or getattr(getattr(openai, "error", None), "InvalidRequestError", None)
    while not received:
        try:
            response = client.chat.completions.create(model=model_name,
                                                      messages=message,
                                                      max_tokens=max_len,
                                                      temperature=temp)
            response = response.model_dump()
            received = True
        except Exception as exc:
            num_rate_errors += 1
            if invalid_req_err is not None and isinstance(exc, invalid_req_err):
                # something is wrong: e.g. prompt too long
                logger.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                raise
            
            logger.error("API error: %s (%d). Waiting %dsec" % (exc, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(prompt, model_name="text-davinci-003", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.Completion.create(model=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo)
            received = True
        except:
            error = sys.exc_info()[0]
            num_rate_errors += 1
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logger.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logger.error("API error: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response
