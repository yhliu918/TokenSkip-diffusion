import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.parallel import DataParallel
import re
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def generate_llada(
    model, 
    prompt, 
    mask_id, 
    denoising_steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0., 
    cfg_scale=0., 
    remasking='low_confidence'
):
    '''
    Args:
        model: Mask predictor (支持多 GPU 的模型，如 DataParallel 或 device_map="auto")
        prompt: {"input_ids": tensor, "attention_mask": tensor}（自动对齐到 model.device
        mask_id: [MASK] 的 token id
    '''
    input_ids = prompt['input_ids']
    attention_mask = prompt['attention_mask']
    
    # 初始化生成序列（自动对齐到 model 所在的设备）
    device = next(model.parameters()).device 
    print(f"Model device: {device}")
    x = torch.full(
        (input_ids.shape[0], input_ids.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device  # 关键修改：用 model.device 而非硬编码 "cuda"
    )
    x[:, :input_ids.shape[1]] = input_ids.clone()
    
    attention_mask_diffusion = torch.full(
        (input_ids.shape[0], input_ids.shape[1] + gen_length),
        1,
        dtype=torch.long,
        device=device
    )
    attention_mask_diffusion[:, :input_ids.shape[1]] = attention_mask.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert denoising_steps % num_blocks == 0
    denoising_steps = denoising_steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, input_ids.shape[1] + num_block * block_length: 
              input_ids.shape[1] + (num_block + 1) * block_length] == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, denoising_steps)
        
        for i in range(denoising_steps):
            mask_index = (x == mask_id)
            
            # 处理 CFG（Classifier-Free Guidance）
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                
                # 多 GPU 下，model(x_) 会自动处理数据分发
                logits = model(x_).logits  
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask_diffusion).logits

            # Gumbel 噪声采样（保持设备一致）
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Remasking 策略
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == 'random':
                x0_p = torch.rand(x0.shape, device=device)  # 确保在正确设备
            else:
                raise NotImplementedError(remasking)

            x0_p[:, input_ids.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # 选择要更新的 token（支持多 GPU）
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], 
                    k=num_transfer_tokens[j, i].item()  # 确保 k 是标量
                )
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def generate_llada_middle(
    model, prompt, mask_id,tokenizer, denoising_steps=128, gen_length=128, block_length=128, 
    temperature=0., cfg_scale=0., remasking='low_confidence', unmask_map=None, unmask_tokens = None, last_block_idx = 0, last_block=10):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        denoising_steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK]
    '''
    input_ids = prompt['input_ids']
    attention_mask = prompt['attention_mask']
    x = torch.full((input_ids.shape[0], input_ids.shape[1] + gen_length),
                   mask_id, dtype=torch.long)
    if unmask_map is not None:
        for i in range(len(unmask_map)):
            x[0,unmask_map[i]+input_ids.shape[1] ] = unmask_tokens[i]
    # decode = tokenizer.decode(x[0], skip_special_tokens=True)
    # print("input:",decode)
        
    x = x.to(model.device)
    x[:, :input_ids.shape[1]] = input_ids.clone()
    attention_mask_diffsion = torch.full((input_ids.shape[0], input_ids.shape[1] + gen_length),
                   1, dtype=torch.long).to(model.device)
    attention_mask_diffsion[:, :input_ids.shape[1]] = attention_mask.clone()

    prompt_index = (x != mask_id)
    # print(x)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    # assert denoising_steps % num_blocks == 0
    # denoising_steps = denoising_steps // num_blocks

    denoising_steps_per_block = denoising_steps
    total_denoise = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, input_ids.shape[1] + num_block *
                            block_length: input_ids.shape[1] + (num_block + 1) * block_length] == mask_id)
        # print(block_mask_index, block_mask_index.sum())
        if block_mask_index.sum() == 0:
            continue
        if num_block >= last_block_idx:
            denoising_steps_per_block = last_block
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, denoising_steps_per_block)
        for i in range(denoising_steps_per_block):
            total_denoise += 1
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # mask_indices = block_mask_index[i].nonzero(as_tuple=True)[0]
                logits = model(x, attention_mask=attention_mask_diffsion).logits

            logits_with_noise = add_gumbel_noise(
                logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    

            if remasking == 'low_confidence':
                # p = F.softmax(logits.to(torch.float64), dim=-1)
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, input_ids.shape[1] + (num_block + 1)
                 * block_length:] = -np.inf
            # print(x0_p)

            x0 = torch.where(mask_index, x0, x)
            # print(x0)
            # print(mask_index)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                # print(select_index-input_ids.shape[1])
            x[transfer_index] = x0[transfer_index]
            decode = tokenizer.decode(x[0][-gen_length:], skip_special_tokens=False)
            # print('======================')
            # print(f"transfered step {total_denoise}, block {num_block}")
            # print(decode)
            matches = re.findall(r'\\boxed\{(\d+)\}', decode)
            if matches:
                print("total denoised steps:", total_denoise)
                return x,total_denoise
            # 
    print("total denoised steps:", total_denoise)
    return x,total_denoise
