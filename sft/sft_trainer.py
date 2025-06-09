import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist

class dLLMConsistencyTrainer(Trainer):
    """
    Trainer for dLLM models with consistency loss.
    """
    def __init__(self, loss_type="kl", recon_weight=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.recon_weight = recon_weight
    
    def compute_loss(self, model, inputs, num_items_in_batch=None,return_outputs=False):
        time_steps = inputs.pop("t")
        labels = inputs.pop("labels")
        clean_labels = inputs.pop("clean_labels")
        num_prompt_tokens = inputs.pop("num_prompt_tokens", 0)
        # reshape inputs to (batch_size, sequence_length)
        inputs["input_ids"] = inputs["input_ids"].reshape(inputs["input_ids"].shape[0]*inputs["input_ids"].shape[1], -1)
        # print("inputs: ", inputs["input_ids"].shape, time_steps.shape, clean_labels.shape)
        outputs = model(**inputs)
        pred_logits = outputs.logits
        # pred_logits to list
        all_preds = pred_logits.split(1, dim=0)
        time_steps = time_steps.split(1, dim=1)
        # print("all_preds: ",len(all_preds) )
        # print("time_steps: ", len(time_steps))
        
    
        # index -2 is used to select the prediction with the lowest noise
        anchor_index = -2 
        anchor_pred = all_preds[anchor_index].detach()  
        
        consistency_loss = 0.0
        num_pairs = 0
        max_weight = 1.0 / (time_steps[-3].mean(dim=1) + 1e-6) 
        # print("max_weight: ", max_weight)
        for i, pred in enumerate(all_preds):
            if i == len(all_preds) - 2 or i== len(all_preds) - 1:  # Skip the anchor and the last prediction
                continue
            
            if self.loss_type == "kl":
                loss = F.kl_div(
                    F.log_softmax(pred, dim=-1),
                    F.softmax(anchor_pred, dim=-1),
                    reduction="none"
                ).sum(-1)
            else:
                loss = F.mse_loss(pred, anchor_pred, reduction="none").mean(-1)
            # print(loss.shape, time_steps[i].shape)
            
            weight = 1.0 / (time_steps[i].mean(dim=1) + 1e-6)  # 平均时间步作为权重因子
            # print("weight: ", weight)
            
            # Normalize by the maximum weight
            weighted_loss = loss * weight[:, None]/ max_weight[:, None]  
            # print(weighted_loss)
            
            consistency_loss += weighted_loss.mean()
            num_pairs += 1
        
        consistency_loss /= num_pairs
        
        # loss between clean labels and the highest noise prediction
        reconstruction_loss = F.cross_entropy(
            all_preds[0].view(-1, all_preds[0].shape[-1]), 
            clean_labels.view(-1),
            ignore_index=-100 
        )
        # reconstruction_loss = reconstruction_loss.sum() / (clean_labels.numel() - num_prompt_tokens)
        

        total_loss = consistency_loss + self.recon_weight * reconstruction_loss
        
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "consistency_loss": consistency_loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "total_loss": total_loss.item()
            })
        
        return total_loss if not return_outputs else (total_loss, outputs)


class dLLMDataConsistencyCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]
    def forward_process(self, batch, num_versions=6, eps=1e-3):
        input_ids = batch["input_ids"]
        device = input_ids.device
        B, N = input_ids.shape
    
        # t_list = [torch.rand((B,), device=device) for _ in range(num_versions)]
        
        # t_list = sorted(t_list, key=lambda x: x.mean())
        
        start = 0.1
        end = 0.9
        fixed_values = torch.linspace(start, end, num_versions, device=device)

        t_list = [torch.full((B,), value, device=device) for value in fixed_values]
            
        noisy_inputs = []
        expanded_t = []
        labels = []
        
        for t in t_list:
            t_clamped = (1 - eps) * t + eps
            
            t_exp = t_clamped[:, None].expand(B, N)
            expanded_t.append(t_exp)
            
            mask = torch.rand((B, N), device=device) < t_exp
            noisy = torch.where(mask, self.mask_token_id, input_ids)
            noisy_inputs.append(noisy)
            
            label = input_ids.clone()
            label[~mask] = -100
            labels.append(label)
        
        clean_input = input_ids.clone()
        noisy_inputs.append(clean_input)
        labels.append(clean_input.clone())
        labels = torch.stack(labels, dim=1)  # shape: (B, num_versions, N)
        # noisy_inputs to tensor
        noisy_inputs = torch.stack(noisy_inputs, dim=1)
        expanded_t.append(torch.zeros_like(expanded_t[0]))  # t=0表示干净数据
        
        # print(f"noisy_inputs shape: {noisy_inputs.shape}, expanded_t shape: {torch.stack(expanded_t, dim=1).shape}, labels shape: {labels.shape}")
        return noisy_inputs, torch.stack(expanded_t, dim=1), labels
        

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["clean_labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], batch["labels"] = self.forward_process(batch)
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[2]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            prompt_mask_expanded = prompt_mask.unsqueeze(1).expand_as(noisy_batch)
            if batch["input_ids"].dim() == 2:  # (1, 4096)
                original_input = batch["input_ids"].unsqueeze(1).expand(noisy_batch.shape[0], noisy_batch.shape[1], -1)  # (B, 1, 4096)
            else: 
                original_input = batch["input_ids"]
            # check dimention of noisy_batch
            # if batch["input_ids"].shape[1:] == noisy_batch.shape:
            #     return None
            # print(batch["input_ids"].shape, noisy_batch.shape)
            
            original_input_expanded = original_input.expand_as(noisy_batch) 
            noisy_batch[prompt_mask_expanded] = original_input_expanded[prompt_mask_expanded].clone()
            batch["labels"][prompt_mask_expanded] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        batch["input_ids"] = noisy_batch.long()
        return batch
  
    
class dLLMTrainer(Trainer):
        
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        print(inputs)
        print(inputs["input_ids"].shape)
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss if not return_outputs else (loss, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        batch["input_ids"] = noisy_batch.long()
        return batch


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""


def preprocess_dataset(data, model, tokenizer, max_length, test_split=0.01):
    preprocessed_data = []
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        trajectory = f"<reasoning>{data[i]['thinking_trajectories'][0]}</reasoning>\n<answer>{data[i]['attempt']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)
        num_tokens = tokenized_input.shape[0]
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data
