from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch
import os
import json
import requests
import re
import argparse
import time
import threading
import itertools
from tqdm import tqdm
import logging

DEVICE = "cuda"  
DEVICE_ID = "0"  
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE   

def torch_gc():
    if torch.cuda.is_available(): 
        with torch.cuda.device(CUDA_DEVICE): 
            torch.cuda.empty_cache()  
            torch.cuda.ipc_collect()  

def parse_arguments():
    parser = argparse.ArgumentParser(description='LLaMA3 dataset rewriting script')
    
    parser.add_argument('--model_path', type=str, 
                       default='/share/share/tangkexian/model/Meta-Llama-3-8B-Instruct',
                       help='model path')
    parser.add_argument('--model_name', type=str, 
                       default='llama-3-8B-Instruct',
                       help='model name')

    parser.add_argument('--port', type=int, default=8000,
                       help='port for vLLM API, default is 8000')
    
    parser.add_argument('--input_file', type=str, default='/root/FActscore/.cache/factscore/data/labeled/prompt_entities.txt',
                       help='input file path for input text')
    parser.add_argument('--output_file', type=str, default='output.txt',
                       help='output file path for processed text') 
    
    # hyperparameters for text generation
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                       help='maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='repetition penalty for text generation')
    
    parser.add_argument('--system_prompt', type=str, 
                       default="You are an assistant trained to generate the biography of definite figure.",
                       help='system prompt for the model')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output for debugging')
    parser.add_argument('--debug-http', action='store_true',
                       help='Enable HTTP request debugging (very verbose)')
    
    return parser.parse_args()

def setup_logging(output_dir, verbose=False, debug_http=False):
    if output_dir is None:
        output_dir = "."  
    
    log_file = Path(output_dir).parent / "processing.log"
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  
        ]
    )
    
    if not debug_http:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING) 
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def split_text_by_structure(text):
    logger = logging.getLogger(__name__)
    
    # Validate input
    if text is None:
        logger.error("None text provided to split_text_by_structure")
        return [], []
    
    if not isinstance(text, str):
        logger.error(f"Non-string text provided to split_text_by_structure: {type(text)}")
        return [], []
    
    if not text.strip():
        logger.warning("Empty or whitespace-only text provided to split_text_by_structure")
        return [""], [""]
    
    logger.debug(f"Splitting text by structure, length: {len(text)}")
    
    text_parts = []
    titles = []
    Onlytitle = ""
    title = ""
    
    try:
        if '</s>' in text:
            main_parts = text.split('</s>')

            for i, part in enumerate(main_parts):
                part = part.strip()
                if not part:
                    continue
                if i==0 and part.startswith('<s>'):
                    text_parts.append(part)
                    titles.append("")
                    continue
                
                if len(part) < 10:
                    Onlytitle += part.strip()
                    if Onlytitle and i == len(main_parts) - 1:
                        titles.append(Onlytitle.strip())
                        text_parts.append("")
                    continue
                    
                if part.startswith('<s>'):
                    # Combine subtitle with the title in the next part
                    if Onlytitle:
                        title += Onlytitle + '</s><s>'
                        Onlytitle = ""

                    part = part.replace('<s>', '').strip()
                    title_pattern = r'\.\:'
                    text = part
                    # logger.debug(f"Processing part with title pattern: {part[:100]}...")
                    while re.search(title_pattern, text, re.IGNORECASE):
                        snippets = re.split(title_pattern, text, 1, re.IGNORECASE)
                        title += snippets[0].strip() + '.:'
                        text = snippets[1].strip() if len(snippets) > 1 else ""
                        # logger.debug(f"remained text a: {snippets[1].strip() if len(snippets) > 1 else ''}")
                    snippets = re.split(r'\.', text, 1, re.IGNORECASE)
                    title += snippets[0].strip() + '.'

                    # logger.debug(f"Extracted title: {title.strip()}")
                    # logger.debug(f"Remaining text: {snippets[1].strip() if len(snippets) > 1 else ''}")
                    titles.append(title.strip())
                    text_parts.append(snippets[1].strip() if len(snippets) > 1 else "")
                    
                    title = ""
        
        # If no structured text found, treat entire text as one part
        if not text_parts and text.strip():
            text_parts = [text.strip()]
            titles = [""]

        # text_parts = [part for part in text_parts if part.strip()]

        for title in titles:
            print (f"Title: {title}")
        
        logger.debug(f"Text split completed: {len(text_parts)} parts, {len(titles)} titles")
        return text_parts, titles
        
    except Exception as e:
        logger.error(f"Error in split_text_by_structure: {type(e).__name__}: {e}")
        logger.error(f"Input text preview: {text[:200] if text else 'None'}...")
        # Return the entire text as a single part in case of error
        return [text] if text and text.strip() else [""], [""]

def process_txt_folder(input_dir, output_dir, model, tokenizer, system_prompt, mode, api_base):
    logger = logging.getLogger(__name__)
    
    # Validate input directory
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    if not os.path.isdir(input_dir):
        logger.error(f"Input path is not a directory: {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    try:
        files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    except Exception as e:
        logger.error(f"Failed to list files in input directory {input_dir}: {e}")
        return

    output_parent = Path(output_dir).parent  
    group_dir_1 = output_parent / "group_1"
    group_dir_2 = output_parent / "group_2"
    group_dir_3 = output_parent / "group_3"
    group_dir_4 = output_parent / "group_4"
    
    logger.info(f"Found {len(files)} text files to process in {input_dir}")

    if not files:
        logger.warning(f"No .txt files found in input directory: {input_dir}")
        return

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for file in tqdm(files, desc="Processing txt files", unit="file"):
        # Check if the file already exists in the output directory
        output_file_path = os.path.join(output_dir, file)
        if os.path.exists(output_file_path) or os.path.exists(group_dir_1 / file) or \
           os.path.exists(group_dir_2 / file) or os.path.exists(group_dir_3 / file) or \
           os.path.exists(group_dir_4 / file):
            logger.info(f"Skipping {file} as it already exists in one of the output directories")
            skipped_count += 1
            continue
            
        input_file_path = os.path.join(input_dir, file)
        
        try:
            logger.debug(f"Reading file: {input_file_path}")
            with open(input_file_path, "r", encoding="utf-8") as f:
                original_text = f.read()
            
            if not original_text or not original_text.strip():
                logger.warning(f"File {file} is empty or contains only whitespace")
                # Still create an empty output file to mark it as processed
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write("")
                processed_count += 1
                continue
            
            # Split text by structure
            try:
                text_parts, titles = split_text_by_structure(original_text)
                logger.debug(f"File {file} split into {len(text_parts)} text parts")
            except Exception as e:
                logger.error(f"Error splitting text structure for file {file}: {e}")
                error_count += 1
                continue
            
            try:
                rewritten_parts = multithread_processing(
                    text_parts,
                    titles,
                    api_base,
                    args
                )
                torch_gc()
                
                if not rewritten_parts:
                    logger.warning(f"No rewritten parts returned for file {file}")
                    rewritten_text = ""
                else:
                    rewritten_text = '\n'.join(rewritten_parts)
                    
                # logger.debug(f"File {file} processed successfully, output length: {len(rewritten_text)}")
                
            except Exception as e:
                logger.error(f"Error processing text parts for file {file}: {e}")
                error_count += 1
                continue
            
            try:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(rewritten_text)
                logger.debug(f"Output written to: {output_file_path}")
                processed_count += 1
            except Exception as e:
                logger.error(f"Error writing output file {output_file_path}: {e}")
                error_count += 1
                continue

        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file_path}")
            error_count += 1
            continue
        except PermissionError:
            logger.error(f"Permission denied accessing file: {input_file_path}")
            error_count += 1
            continue
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error reading file {input_file_path}: {e}")
            error_count += 1
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing file {file}: {type(e).__name__}: {e}")
            error_count += 1
            continue
    
    logger.info(f"Folder processing completed:")
    logger.info(f"  Total files found: {len(files)}")
    logger.info(f"  Files processed: {processed_count}")
    logger.info(f"  Files skipped (already exist): {skipped_count}")
    logger.info(f"  Files with errors: {error_count}")
    
    if error_count > 0:
        logger.warning(f"Processing completed with {error_count} file errors")

def get_prompt_from_text(input_text, mode):
    logger = logging.getLogger(__name__)
    
    if not input_text or not input_text.strip():
        logger.warning("Empty or whitespace-only input_text provided to get_prompt_from_text")
        return ""
    
    if not mode or mode not in ['extract', 'svo', 'compact']:
        logger.error(f"Invalid mode provided to get_prompt_from_text: {mode}")
        return ""
    
    # logger.debug(f"Generating prompt for mode '{mode}', input length: {len(input_text)}")
    
    if mode == 'extract':
        prompt = f"""Your task is to rewrite knowledge from the provided text, which is a biography, following these instructions:
        - Rewrite the text as one passage using easy-to-understand and high-quality English like sentences in textbooks and Wikipedia.
        - Focus on content related to the person in the biography.
        - Retain all details from the original text (including dates, numbers, names, and specific descriptions). Do not omit information.
        - Do not add or alter details. Only restate what is already in the text.
        - Write in plain text.
        - Avoid using any pronouns and use specific nouns instead (e.g., use "John" instead of "he").
        - Do not insert real line breaks.

        Text:
        {input_text}

        Task:
        Rewrite facts and knowledge from the above text as a paragraph following the instructions.
        """
    elif mode == 'svo':
        prompt = f"""Your task is to rewrite the given biography text into a list of brief factual sentences, following these instructions:
        1.Rewrite the following paragraph into a list of brief facts. Each fact should:
        - Use only a simple subject-verb-object (SVO) structure.
        - Avoid complex clauses or conjunctions.
        - Retain all details from the original text (including dates, numbers, names, and specific descriptions). Do not omit information.
        - Avoid using any pronouns as subjects (e.g., use "John" instead of "he").
        - Use simple and straightforward vocabulary.
        2.After rewriting, concatenate all sentences from the list into one continuous paragraph.

        Example:
        According to Plotz, dentists have successfully adapted to remain relevant and profitable by offering treatments that address patients' desire for self-improvement and cosmetic enhancements, while also prioritizing a more comfortable patient experience. 
        Rewritten:
        Plotz states dentists adapted successfully. Dentists remain relevant. Dentists remain profitable. Dentists offer treatments. Treatments address patients' desires. Patients desire self-improvement. Patients desire cosmetic enhancements. Dentists prioritize comfort. Comfort improves patient experience.
		
        Example:
		These tactics, including fear-inducing diagnostic tools and slick marketing campaigns, are reshaping patient expectations and the nature of dental practice.
		Rewritten:
		Tactics include fear-inducing tools. Tactics include slick campaigns. Tactics reshape patient expectations. Tactics reshape dental practice.
        
        Text:
        {input_text}

        Task:
        Compact the text above following the instructions and output only the final version in a paragraph without any other words like 'Here is' or explanations.
        """
    elif mode == 'compact':
        prompt = f"""Your task is to rewrite the provided text which uses a simple subject-verb-object (SVO) structure.following these instructions:
        - Combine sentences that share the same subject (in the SVO structure) into one sentence. 
        - Keep sentences with the same object separate if their subjects are different.
        - Consider a person’s full name and their shortened name (e.g., “David Plotz” and “Plotz”, or “Edward Shannon” and “Shannon”) as the same subject and merge their sentences. 
        - Do not use relative clauses (such as "who", "which", "that", "where").
        - Avoid complex sentence structures or conjunctions that create embedded clauses.
        - Use specific nouns as subjects (e.g., use “John” instead of “he”).
        - Keep the subject the same as in the original sentences.
        
        Example:
        Plotz describes the profession. The profession avoided obsolescence. The profession shifted to patient comfort. The profession shifted to cosmetic services. The profession marketed elective services. Elective services include cosmetic services.
        Rewritten:
        Plotz describes the profession. The profession avoided obsolescence, shifted to patient comfort, shifted to cosmetic services, and marketed elective services. Elective services include cosmetic services.
        
        Example:
        The profession marketed elective services. Elective services include cosmetic services.
        Rewritten:
        The profession marketed elective services. Elective services include cosmetic services.
        
        Example:
        Edward Shannon owns Shannon's Imperial Circus. Shannon portrays a society on interplanetary expansion's brink.
        Rewritten:
        Edward Shannon owns Shannon's Imperial Circus, portrays a society on interplanetary expansion's brink.

        Text:
        {input_text}

        Task:
        Compact the text above following the instructions and output only the final version in a paragraph without any other words like 'Here is' or explanations.
        """
    else:
        raise ValueError("Invalid mode. Please choose 'extract', 'svo' or 'compact'.")
    
    return prompt

def generate_response_vllm(prompt,
                           system_prompt,
                           api_base,
                           model_name,
                           max_new_tokens=2048,
                           temperature=0.5,
                           top_p=0.9):
    logger = logging.getLogger(__name__)
    
    if not prompt or not prompt.strip():
        logger.error("Empty or whitespace-only prompt provided to generate_response_vllm")
        return ""
    
    if not system_prompt or not system_prompt.strip():
        logger.warning("Empty or whitespace-only system_prompt provided to generate_response_vllm")
        system_prompt = "You are a helpful assistant."
    
    try:
        # Use requests directly to avoid OpenAI client proxy issues
        url = f"{api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY"
        } 
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens
        }
        
        # Disable proxy for localhost/127.0.0.1
        proxies = {}
        if "localhost" in api_base or "127.0.0.1" in api_base:
            proxies = {
                'http': None,
                'https': None
            }
        
        response = requests.post(url, headers=headers, json=data, proxies=proxies, timeout=600)
        
        if response.status_code == 400:
            logger.error(f"400 Bad Request - Response: {response.text}")
            logger.error(f"Request data: {json.dumps(data, indent=2)}")
            return ""
        elif response.status_code == 404:
            logger.error(f"404 Not Found - API endpoint not available at {url}")
            return ""
        elif response.status_code == 500:
            logger.error(f"500 Internal Server Error - Server error at {api_base}")
            return ""
        elif response.status_code != 200:
            logger.error(f"HTTP {response.status_code} error - Response: {response.text}")
            return ""
            
        response.raise_for_status()
        
        result = response.json()
        output_text = result['choices'][0]['message']['content']
        
        return output_text.strip()
                
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error when calling API at {api_base} (timeout=600s)")
        return ""
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error when calling API at {api_base}: {e}")
        return ""
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error when calling API at {api_base}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error in vLLM generation: {type(e).__name__}: {e}")
        logger.error(f"API base: {api_base}, Model: {model_name}")
        return ""

def multithread_process_text(i, text_part, api_base, model_name, system_prompt, mode, max_new_tokens=2048, temperature=0.5, top_p=0.9):
    # api_base = API_BASES[i % len(API_BASES)]
    prompt = get_prompt_from_text(text_part, mode=mode)
    response = generate_response_vllm(
        prompt, system_prompt, api_base,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return response

def Formatted_text(rewritten_texts, titles): # titles + ' ' + rewritten_texts + \n
    rewritten_parts = []
    for i, rewritten_text in enumerate(rewritten_texts):
        title = titles[i] if i < len(titles) else ""
        rewritten_text = rewritten_text or ""  # Handle None values
        
        if not rewritten_text:
            rewritten_part = f"<s>{title}</s>".strip() if title else ""
        else:
            rewritten_part = f"<s>{title} {rewritten_text}</s>".strip() if title else f"<s>{rewritten_text}</s>".strip()

        rewritten_parts.append(rewritten_part)
    
    return rewritten_parts

def multithread_processing(text_parts, titles, api_base, args):
    logger = logging.getLogger(__name__)
    rewritten_texts = [None] * len(text_parts)
    empty_input_count = 0

    with ThreadPoolExecutor(max_workers=256) as executor:
        future_to_index = {}
        for i, text_part in enumerate(text_parts):
            if not text_part or text_part.strip() == "":
                logger.debug(f"Skipping empty text part {i}")
                rewritten_texts[i] = ""
                empty_input_count += 1
                continue
            future = executor.submit(multithread_process_text, 
                    i, text_part, api_base,
                    args.model_name,
                    system_prompt=args.system_prompt,
                    mode=args.mode,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p)
            future_to_index[future] = i

        completed_count = 0
        
        with tqdm(total=len(text_parts), desc="Processing text parts", unit="part") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    rewritten_texts[index] = result   
                except Exception as e:
                    logger.error(f"Error processing text part {index}: {type(e).__name__}: {e}")
                    rewritten_texts[index] = ""
                
                completed_count += 1
                pbar.update(1)
                pbar.set_postfix(completed=f"{completed_count}/{len(text_parts)}")

    return Formatted_text(rewritten_texts, titles)

if __name__ == '__main__':
    args = parse_arguments()

    if args.output_dir:
        root_dir = Path(args.output_dir).parent 
        root_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logging(args.output_dir, args.verbose, getattr(args, 'debug_http', False))
    else:
        logger = setup_logging(None, args.verbose, getattr(args, 'debug_http', False))    
    
    if args.verbose:
        logger.info(f"model path: {args.model_path}")
        logger.info(f"processing mode: {args.mode}")

    mode = args.mode 
    system_prompt = args.system_prompt 
    api_base = f"http://localhost:{args.port}/v1"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    if args.input_dir and args.output_dir:
        logger.info(f"=== Start processing! ===")
        logger.info(f"input directory: {args.input_dir}")
        logger.info(f"output directory: {args.output_dir}")
        logger.info(f"processing mode: {args.mode}")
        
        process_txt_folder(args.input_dir, args.output_dir, model, tokenizer, system_prompt, mode, api_base)
        logger.info("Completed processing txt files!")
        
    elif args.input_text:
        logger.info("=== Start processing the input text! ===")
        text_parts, titles = split_text_by_structure(args.input_text)
        
        for i, part in enumerate(tqdm(text_parts, desc="Processing text parts", unit="part")):
            logger.info(f"\n=== text part {i+1} ===")
            if args.verbose:
                logger.debug(f"original text: {part[:100]}...")
            
            response = multithread_process_text(
                part,
                args.model_name,
                api_base="http://localhost:8000/v1",
                system_prompt=args.system_prompt,
                mode=args.mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            logger.info(f"processed text:\n{response}")
            logger.info("-" * 50)
            
    elif args.input_file:
        logger.info(f"=== Start processing: {args.input_file} ===")
        with open(args.input_file, "r", encoding="utf-8") as f:
                original_text = f.read()
        
        if not original_text or not original_text.strip():
            logger.warning(f"Input file {args.input_file} is empty or contains only whitespace")
            original_text = ""
        
        logger.debug(f"Input file read successfully, length: {len(original_text)}")
        
        try:
            text_parts, titles = split_text_by_structure(original_text)
            logger.info(f"Found {len(text_parts)} text parts to process")
        except Exception as e:
            logger.error(f"Error splitting text structure: {e}")
            exit(1)

        try:
            rewritten_parts = multithread_processing(
                text_parts,
                titles,
                api_base,
                args
            )
            torch_gc()

            if not rewritten_parts:
                logger.warning("No rewritten parts returned from processing")
                rewritten_text = ""
            else:
                rewritten_text = '\n'.join(rewritten_parts)
                logger.info(f"Processing completed, output length: {len(rewritten_text)}")
        except Exception as e:
            logger.error(f"Error during text processing: {type(e).__name__}: {e}")
            exit(1)
        
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(rewritten_text)
            logger.info(f"Output written to: {args.output_file}")
        except PermissionError:
            logger.error(f"Permission denied writing to file: {args.output_file}")
            exit(1)
        except Exception as e:
            logger.error(f"Error writing output file {args.output_file}: {type(e).__name__}: {e}")
            exit(1)
    
    torch_gc()
    logger.info("\n=== processing completed! ===")

