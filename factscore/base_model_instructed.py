import argparse
import os
import subprocess
import torch
import tqdm
import transformers

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def recover_instruct_llama(cpt_path, output_path, device="cpu", test_recovered_model=True):
    """
    continual_pretrain + (instruct - base) = improved model
    """
    model_cpt = transformers.AutoModelForCausalLM.from_pretrained(
        cpt_path,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model_instruct = transformers.AutoModelForCausalLM.from_pretrained(
        "/root/CKM/model/Meta-Llama-3-8B-Instruct",
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model_base = transformers.AutoModelForCausalLM.from_pretrained(
        "/root/CKM/model/Meta-Llama-3-8B",
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer_base = transformers.AutoTokenizer.from_pretrained("/root/CKM/model/Meta-Llama-3-8B")
    if tokenizer_base.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=model_base,
            tokenizer=tokenizer_base,
        )
    tokenizer_recovered = transformers.AutoTokenizer.from_pretrained("/root/CKM/model/Meta-Llama-3-8B-Instruct")
    
    state_dict_cpt = model_cpt.state_dict()
    state_dict_instruct = model_instruct.state_dict()
    state_dict_base = model_base.state_dict()
    
    for key in tqdm.tqdm(state_dict_cpt):
        # wdiff = instruct - base
        # result = cp + (instruct - base) = cp + instruct - base
        state_dict_cpt[key].add_(state_dict_instruct[key])
        state_dict_cpt[key].sub_(state_dict_base[key])
    
    model_cpt.load_state_dict(state_dict_cpt)

    if output_path is not None:
        model_cpt.save_pretrained(output_path)
        tokenizer_recovered.save_pretrained(output_path)

    if test_recovered_model:
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
        )
        inputs = tokenizer_recovered(input_text, return_tensors="pt")
        out = model_recovered.generate(inputs=inputs.input_ids, max_new_tokens=100)
        output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
        output_text = output_text[len(input_text) :]
        print(f"Input: {input_text}\nCompletion: {output_text}")

    return model_recovered, tokenizer_recovered

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore")
    parser.add_argument('--llama_7B_HF_path',
                        type=str,
                        default=None)

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    download_file("1sbW6pkYl6cc9gooD4WLaeoFKcAj3poZu", "demos.zip", args.data_dir)
    download_file("155exEdKs7R21gZF4G-x54-XN3qswBcPo", "data.zip", args.data_dir)
    download_file("1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I", "enwiki-20230401.db", args.data_dir)

    if args.llama_7B_HF_path:
        recover_instruct_llama(args.llama_7B_HF_path, os.path.join(args.model_dir, "inst-llama-7B"))

    # download the roberta_stopwords.txt file
    subprocess.run(["wget https://raw.githubusercontent.com/shmsw25/FActScore/main/roberta_stopwords.txt"], shell=True)

    # move the files to the data directory
    subprocess.run(["mv demos %s" % args.data_dir], shell=True)
    subprocess.run(["mv enwiki-20230401.db %s" % args.data_dir], shell=True)

