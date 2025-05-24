import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from llada import generate_llada_middle
    

from eval.utils import generate_completions
from data_processing.process_utils import *
from data_processing.answer_extraction import *
from eval.eval_script import *
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_data(path):
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                data.append(line)
    else:
        raise NotImplementedError()
    return data

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_index_of_compressed(origin, compressed):
    # print(origin, compressed)
    index_list = []
    j = 0
    for i in range(len(origin)):
        if j < len(compressed) and compressed[j] not in origin:
            j += 1
            continue
        if j == len(compressed):
            break
        if origin[i] == compressed[j]:
            index_list.append(i)
            j += 1
        
    return index_list

def get_masked_index(tokenizer, origin_cot, compressed_cot):
    tokenize_kwargs = dict(
                return_tensors='pt',
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=4096
            )
    origin_unmasked_cot = tokenizer.batch_encode_plus([origin_cot], **tokenize_kwargs)

    com_unmasked_cot = tokenizer.batch_encode_plus([compressed_cot], **tokenize_kwargs)
    unmask_index = find_index_of_compressed(origin_unmasked_cot.input_ids[0], com_unmasked_cot.input_ids[0])
    return unmask_index,origin_unmasked_cot,com_unmasked_cot


            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression_ratio", type=float, default=0.9)
    args, unparsed_args = parser.parse_known_args()
    model_path = "/cpfs02/shared/llmit6/liudawei/xpuyu_work_dirs/internlm2-1_8b-myds-llada-sft-v3/pretrain-310000-yhc-padto2power/20250430200836/hf-9epoch-068310-7593"
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cuda:4", attn_implementation='flash_attention_2')
    model = model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    

    print(f"Evaluating {model_path}", flush=True)
    file_name = '/cpfs02/user/yihao/TokenSkip-llada/outputs/LLaDA-8B-Base/gsm8k/8b/Compression/train_outputs_compressed_ratio_{}.jsonl'.format(args.compression_ratio)
    with open(file_name, "r") as file:
        data = []
        for line in file:
            line = json.loads(line)
            data.append(line)
    acc = []
    denoise_steps = []
    original_denoise_steps = []
    for i in range(len(data)):
        origin_cot = data[i]['output'].replace("\n", " ")
        pos = origin_cot.find("$\\boxed{")
        if pos != -1:
            origin_cot = origin_cot[:pos]
        compressed_cot = data[i]['compressed_cot'].replace("\n", " ")
        pos = compressed_cot.find("$\\boxed{")
        if pos != -1:
            compressed_cot = compressed_cot[:pos]
        ground_truth = data[i]['answer']
        # print("origin_cot:", origin_cot)
        # print("compressed_cot:", compressed_cot)
        unmask_index,origin_unmasked_cot,com_unmasked_cot = get_masked_index(tokenizer, origin_cot, compressed_cot)
        original_denoise_steps.append(len(origin_unmasked_cot.input_ids[0])- len(com_unmasked_cot.input_ids[0]))
        
        problem = data[i]['input']

        tokenize_kwargs = dict(
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                    max_length=4096
                )
        tokens = tokenizer.batch_encode_plus([problem], **tokenize_kwargs)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        gen_len = 512
        block_len = 64

        if len(unmask_index) == 0:
            continue
        last_block_idx = ( max(unmask_index))// block_len

        outputs,total_denoise = generate_llada_middle(
                    model,
                    tokens, # L -> 1, L
                    mask_id=128108,
                    tokenizer=tokenizer,
                    denoising_steps=1,
                    gen_length=gen_len,
                    block_length=block_len,
                    temperature=0,
                    cfg_scale=0,
                    remasking='low_confidence',
                    unmask_map=unmask_index,
                    unmask_tokens=com_unmasked_cot.input_ids[0], # 1, L
                    last_block_idx = last_block_idx,
                    last_block = 10
                )
        # print('===================== FINAL OUTPUT =====================')
        # print(tokenizer.decode(outputs[0][-gen_len:]), (outputs[0][-gen_len:]==2).sum(), gen_len - (outputs[0][-gen_len:]==2).sum())
        answer = tokenizer.decode(outputs[0][-gen_len:])
        matches = re.findall(r'\\boxed\{(\d+)\}', answer)
        if matches:
            print(len(origin_unmasked_cot.input_ids[0])- len(com_unmasked_cot.input_ids[0])," v.s. ", total_denoise)
            print("final answer:", matches[0], matches[0] == ground_truth)
            acc.append(matches[0] == ground_truth)
            denoise_steps.append(total_denoise)
            
        
    print("total accuracy:", sum(acc) / len(acc), "out of ",len(acc)," answers")
    print("total original denoising step: ", sum(original_denoise_steps))
    print("total denoising step: ", sum(denoise_steps))

            
    #     print("final answer:", answer    
    
    # for src, info in test_conf.items():
    #     fname = os.path.join(args.output_dir, "test_data", "test.jsonl")
    #     input_dir = os.path.dirname(fname)
    #     os.makedirs(input_dir, exist_ok=True)
    #     metric_path = os.path.join(args.output_dir, "samples", "metrics.json")
        
    #     if os.path.exists(metric_path) and read_data(metric_path)['n_samples'] > 0:
    #         continue
    #     with open(fname, "w") as file:
    #         data = read_data(info['test_path'])
    #         for i, sample in enumerate(tqdm(data, desc=f'processing {src}')):
    #             fn = eval(info['process_fn'])
    #             sample['id'] = sample.get('id', f"{src}-{i}")
    #             for j, item in enumerate(fn(sample)):
    #                 item['dataset'] = src
    #                 item['id'] = f"{src}-test-{i}-{j}"
    #                 assert 'answer' in item
    #                 print(json.dumps(item), file=file, flush=True)

    #         output_dir = os.path.join(args.output_dir, "samples")
    #         os.makedirs(output_dir, exist_ok=True)
    #     set_random_seed(args.seed)

    #     print("Loading data...")
    #     test_data = []
    #     with open(os.path.join(input_dir, f"test.jsonl")) as fin:
    #         for line in fin:
    #             example = json.loads(line)
    #             messages = example['messages']
    #             assert messages[-1]['role'] == 'assistant'
    #             example['reference'] = example.get('reference', '') or [mess['content'] for mess in messages if
    #                                                                     mess['role'] == 'assistant']
    #             for mess in messages:
    #                 if mess['role'] == 'assistant':
    #                     mess['content'] = ''
    #             example['messages'] = messages
    #             test_data.append(example)

    #     if args.max_num_examples and len(test_data) > args.max_num_examples:
    #         test_data = random.sample(test_data, args.max_num_examples)

    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir, exist_ok=True)

    #     results, total_time = infer(args, test_data, info['answer_extraction_fn'])

    #     print("Finished inference...")

    #     os.environ['TOKENIZERS_PARALLELISM'] = "false"

    #     invalid_outputs = []
    #     labels = []
    #     for item in results:
    #         if len(item['prediction']) == 0:
    #             invalid_outputs.append({'prompt': item['prompt'], 'output':  item['model_output'], 'answer': item['prediction']})
    #             res = False
    #             extract_ans = None
    #         else:
    #             extract_ans = item['prediction']
    #             res = eval_math(item)
    #         labels.append(res)

    #     for item, label in zip(results, labels):
    #         item['accuracy'] = label

    #     print("Calculating accuracy...")
    #     acc = 0
    #     for item in results:
    #         acc += item['accuracy']
    #     print("output acc = {:.5f}".format(acc / len(results) * 100), flush=True)

    #     avg_cot_length = sum(item['cot_length'] for item in results) / len(results)
    #     print("output avg_cot_length = {:.5f}".format(avg_cot_length), flush=True)

    #     print("number of invalid outputs: {}".format(len(invalid_outputs)), flush=True)

    #     pred_fname = f"predictions_{dist.get_rank()}.jsonl"
    #     for item in results:
    #         with open(os.path.join(output_dir, pred_fname), 'a+', encoding='utf-8') as fout:
    #             line = json.dumps(item, ensure_ascii=False)
    #             fout.write(line + '\n')

    #     metric_fname = f"metrics_{dist.get_rank()}.json"
    #     with open(os.path.join(output_dir, metric_fname), "w") as fout:
    #         json.dump({
    #             "n_samples": len(results),
    #             "accuracy": sum(item['accuracy'] for item in results) / len(results),
    #             "avg_cot_length": avg_cot_length,
    #             'sample_latency': total_time / len(test_data),
    #         }, fout, indent=4)
