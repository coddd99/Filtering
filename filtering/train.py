import copy
import json
import os
import pathlib

from dataclasses import dataclass, field
from typing import Sequence, Dict, Any, Optional, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import transformers
from transformers import HfArgumentParser, AutoProcessor, BatchEncoding, Trainer

from train_utils import *
from torch.utils.data import Dataset
from trainer import Qwen2VLForConditionalGeneration_SelfFilter, Qwen2VLTrainer_SelfFilter
from qwen_vl_utils import process_vision_info




IGNORE_INDEX = -100


def parse_llava_style_conversations_tmp(conversations: List[Dict], image_folder : str) -> List[Dict]:
    messages = []
    for turn in conversations:
        if turn['from'] == 'human':
            tmp_message = []
            role = 'user'
            tmp_msg = {'role' : role, 'content' : []}
            tmp_msg['content'].append({"type":"text", "text":turn['value'].replace('\n<image>', '')})
            tmp_message.append(tmp_msg)
        else:
            role = 'assistant'
            tmp_msg = {'role' : role, 'content' : []}
            tmp_msg['content'].append({"type":"text", "text":turn['value']})
            tmp_message.append(tmp_msg)
            messages.append(tmp_message)


    if 'image' in sample and sample['image']:
        if len(messages) > 0 and messages[0][0]["role"] == 'user':
            filename = sample["image"]
            if image_folder:
                full_path = os.path.join(image_folder, filename)
                image_path = f"file://{full_path}"
            for message in messages:
                message[0]['content'].insert(0, {"type":"image","image":image_path})


    return messages

def parse_llava_style_conversations(sample : dict, image_folder : str) -> List[Dict]:
    conversations = sample['conversations']
    messages = []
    for turn in conversations:
        if turn['from'] == 'human':
            role = 'user'
            tmp_msg = {'role' : role, 'content' : []}
            tmp_msg['content'].append({"type":"text", "text":turn['value'].replace('\n<image>', '')})
            messages.append(tmp_msg)
        else:
            role = 'assistant'
            tmp_msg = {'role' : role, 'content' : []}
            tmp_msg['content'].append({"type":"text", "text":turn['value']})
            messages.append(tmp_msg)

    if 'image' in sample and sample['image']:
        if len(messages) > 0 and messages[0]["role"] == 'user':
            filename = sample["image"]
            if image_folder:
                full_path = os.path.join(image_folder, filename)
                image_path = f"file://{full_path}"
            messages[0]['content'].insert(0, {"type":"image","image":image_path})

    return messages

def labeling(input_ids, processor):

    text = processor.tokenizer.decode(
        input_ids, 
        skip_special_tokens=False
    )

    label_seq = [-100] * len(input_ids)

    start_substr = "<|im_start|>assistant\n"
    end_substr   = "<|im_end|>"

    pos = 0
    while True:

        start_pos = text.find(start_substr, pos)
        if start_pos == -1:
            break

        region_start = start_pos + len(start_substr)
        end_pos = text.find(end_substr, region_start)
        if end_pos == -1:
            region_end = len(text)
            pos = region_end
        else:
            region_end = end_pos
            pos = end_pos + len(end_substr)

 
        prefix_text = text[:region_start]
        prefix_ids = processor.tokenizer.encode(
            prefix_text,
            add_special_tokens=False
        )
        start_token_idx = len(prefix_ids)


        prefix_text2 = text[:region_end]
        prefix_ids2 = processor.tokenizer.encode(
            prefix_text2,
            add_special_tokens=False
        )
        end_token_idx = len(prefix_ids2)

        for i in range(start_token_idx, min(end_token_idx, len(label_seq))):
            label_seq[i] = input_ids[i]

    return label_seq

class Qwen2VLListDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
        image_folder: Optional[str] = None,
        partial_assistant: bool = True,
    ):
        super().__init__()
        with open(data_path, "r") as f:
            self.data_list = json.load(f)  # => list of dict

        if not isinstance(self.data_list, list):
            raise ValueError("JSON must be a list of samples. Got something else.")

        self.processor = processor
        self.image_folder = image_folder
        self.partial_assistant = partial_assistant

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        sample = self.data_list[idx]


        if "conversations" not in sample:
            raise KeyError(
                f"Item at index={idx} missing 'conversations' key. keys={sample.keys()}"
            )

        messages = parse_llava_style_conversations(sample, self.image_folder)
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs: BatchEncoding = self.processor(
            text=text_prompt,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
 
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        labels = labeling(input_ids, self.processor)


        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        item["unique_idx"] = sample["unique_idx"]

        return item


@dataclass
class Qwen2VLListCollator:
    processor: AutoProcessor
    max_length: int = 2048

    def __call__(self, batch: List[dict]) -> dict:
        input_ids_list = [b["input_ids"] for b in batch]
        attn_list = [b["attention_mask"] for b in batch]
        labels_list = [torch.tensor(b["labels"]) for b in batch]

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            print('No padding token')
            pad_id = 0

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attn_list, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )

        # truncate
        input_ids = input_ids[:, : self.max_length]
        attention_mask = attention_mask[:, : self.max_length]
        labels = labels[:, : self.max_length]

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # images
        if "images" in batch[0]:
            try:
                out["images"] = torch.stack([b["images"] for b in batch], dim=0)
            except:
                out["images"] = [b["images"] for b in batch]

        unique_indices = [instance['unique_idx'] for instance in batch]
        out['unique_indices'] = unique_indices
        return out


def make_qwen2vl_list_data_module(
    data_path: str,
    processor: AutoProcessor,
    image_folder: Optional[str] = None,
    partial_assistant: bool = True,
    max_length: int = 2048,
):
    dataset = Qwen2VLListDataset(
        data_path=data_path,
        processor=processor,
        image_folder=image_folder,
        partial_assistant=partial_assistant,
    )
    collator = Qwen2VLListCollator(
        processor=processor,
        max_length=max_length
    )

    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": collator
    }



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="./checkpoint/Qwen2-VL-7B-Instruct",
        metadata={"help": "Hugging Face model name or local path"},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "If True, freeze all model backbone parameters."},
    )
    feature_extractor_setting: Optional[str] = field(
        default='clip',
        metadata={
            "help": "For SelfFilter logic. E.g. 'clip' or 'scores'. If None, no special weighting."
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="./data/llava_instruct_80k_add_idx.json",
        metadata={"help": "Path to the JSON training data, containing conversations + (optional) image field."},
    )
    image_folder: Optional[str] = field(
        default='./data/cocodataset/train2017',
        metadata={"help": "If local images are used, specify the folder prefix."}
    )
    image_aspect_ratio: str = field(
        default="square",
        metadata={"help": "How to handle image aspect ratio (e.g. 'square', 'pad', etc.)"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    output_dir : str = "./checkpoint/qwen_result"

    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length for Qwen2-VL model. Typically 2048 or 4096, etc."
        },
    )


    double_quant: bool = field(
        default=False,
        metadata={
            "help": "If using 4bit, compress quantization statistics via double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type: 'fp4' or 'nf4'."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use. E.g. 16, 8, or 4."},
    )
    bf16 : bool = True
    # LoRA
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")

    # Group by modality length (optional)
    group_by_modality_length: bool = field(default=False)
    clip_feature_path: Optional[str] = field(
        default=None,
        metadata={"help": "CLIP feature path."}
    )
    # Reporting integration
    report_to: str = field(
        default="none",
        metadata={"help": "Reporting integration, e.g. 'wandb' or 'none'."},
    )



def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "score_net" not in name:
            lora_module_names.add(name)
    return list(lora_module_names)

def rank0_print(*args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)


class Qwen2VLTrainer_nopoint(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _save_checkpoint(self, model, trial, metrics=None):

        pass


def train_qwen2vl(attn_implementation='flash_attention_2'):

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )


    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(
            dict(
                load_in_4bit=(training_args.bits == 4),
                load_in_8bit=(training_args.bits == 8),
                # device_map="auto" or {"": training_args.device} 식으로 가능
                device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=(training_args.bits == 4),
                    load_in_8bit=(training_args.bits == 8),
                    llm_int8_skip_modules=[],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    if model_args.feature_extractor_setting not in ["clip", "scores"]:
        raise ValueError("Unknown feature_extractor_setting. Choose 'clip' or 'scores' or modify code.")

    rank0_print("Loading Qwen2-VL model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_args.feature_extractor_setting == "clip":
        model = Qwen2VLForConditionalGeneration_SelfFilter.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            **bnb_model_from_pretrained_args
        ).to(device)
    else:
        model = Qwen2VLForConditionalGeneration_SelfFilter.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            **bnb_model_from_pretrained_args
        ).to(device)

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if "score_net" in name:
                param.requires_grad = True 

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            # 16-bit일 때 fp16/bf16 맞춰서 모델로.
            if training_args.bf16:
                model.to(torch.bfloat16)
            elif training_args.fp16:
                model.to(torch.float16)

        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    rank0_print("Loading Qwen2-VL tokenizer/processor ...")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token


    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

   
    print('prepareing dataaset....')
    data_module = make_qwen2vl_list_data_module(
        data_path=data_args.data_path,  # 위 JSON
        processor=processor,
        image_folder=data_args.image_folder,   # 실제 이미지 폴더
        partial_assistant=True,
        max_length=2048
    )
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    print('prepareing trainer....')
    trainer = Qwen2VLTrainer_SelfFilter(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        feature_extractor_setting = model_args.feature_extractor_setting
    )


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, training_args.output_dir)

if __name__ == "__main__":

    train_qwen2vl(attn_implementation='flash_attention_2')
