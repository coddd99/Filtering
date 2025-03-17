import torch
import os
import json
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info

from typing import Dict, List
import torch.nn.functional as F

path = "./data/llava_instruct_80k.json"
save_path = "/root/vol1/video_content/ppl_metrics_2B.json"
train_data = json.load(open(path, "r"))
idx = 0
train_data = train_data[idx:]

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype=torch.bfloat16, 
    attn_implementation='flash_attention_2',
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


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
            label_seq[i] = input_ids[i].item()
    label_seq = torch.tensor(label_seq)
    label_seq = label_seq.unsqueeze(0)
    return label_seq



def parse_qwen_conversations(sample : dict, image_folder : str) -> List[Dict]:
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


def parse_qwen_y_conversations(sample : dict) -> List[Dict]:
    conversations = sample['conversations']
    messages = []
    for turn in conversations:
        if turn['from'] == 'gpt':
            role = 'assistant'
            tmp_msg = {'role' : role, 'content' : []}
            tmp_msg['content'].append({"type":"text", "text":turn['value']})
            messages.append(tmp_msg)

    return messages



batch_size = 16  
all_results = []
n_data = len(train_data)

for start_idx in tqdm(range(0, n_data, batch_size)):
    end_idx = start_idx + batch_size
    batch_data = train_data[start_idx:end_idx]

    text_prompts = []
    text_y_prompts = [] 
    image_inputs_list = []
    video_inputs_list = []

    for line in batch_data:
        messages = parse_qwen_conversations(line, image_folder="/root/vol1/cocodataset/train2017")
        messages_y = parse_qwen_y_conversations(line, image_folder="/root/vol1/cocodataset/train2017")
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )[:-1]
        
        text_y_prompt = processor.apply_chat_template(
            messages_y, tokenize= False, add_generation_prompt = False
        )
        
        text_prompts.append(text_prompt)
        text_y_prompts.append(text_y_prompt)

        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs_list.append(image_inputs)
        video_inputs_list.append(video_inputs)


    inputs = processor(
        text=text_prompts,
        images=image_inputs_list,
        videos=None,
        return_tensors="pt",
        padding=True  # 배치의 길이 맞추기
    ).to(model.device)

    
    input_labels_list = []
    for i in range(len(batch_data)):
        labels_i = labeling(inputs.input_ids[i], processor)  
        #labels_i = labels_i.unsqueeze(0)  # 차원 맞추기 (1, seq_len)
        input_labels_list.append(labels_i)

    input_labels = torch.cat(input_labels_list, dim=0).to(model.device)


    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    logits = outputs.logits
    pad_id = processor.tokenizer.pad_token_id

    loss_per_token_cond = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        input_labels.view(-1),
        reduction='none'
    )
    loss_per_token_cond = loss_per_token_cond.view(logits.size(0), -1)

    mask_cond = (input_labels != -100).float()
    loss_per_token_cond = loss_per_token_cond * mask_cond
    token_count_cond = mask_cond.sum(dim=1)
    sample_loss_cond = loss_per_token_cond.sum(dim=1) / token_count_cond
    sample_ppl_cond = torch.exp(sample_loss_cond)


    inputs_y = processor(
        text=text_y_prompts,
        images = None,
        videos = None,
        return_tensors = "pt",
        padding = True
    ).to(model.device)

    with torch.no_grad():
        outputs_y = model(**inputs_y, return_dict=True)

    logits_uncond = outputs_y.logits
    loss_per_token_uncond = F.cross_entropy(
        logits_uncond.view(-1, logits_uncond.size(-1)),
        inputs_y.input_ids.view(-1),
        reduction='none'
    ).view(logits_uncond.size(0), -1)

    mask_uncond = (inputs_y.input_ids != pad_id).float()
    loss_per_token_uncond = loss_per_token_uncond * mask_uncond
    token_count_uncond = mask_uncond.sum(dim=1)
    sample_loss_uncond = loss_per_token_uncond.sum(dim=1) / token_count_uncond
    sample_ppl_uncond = torch.exp(sample_loss_uncond)

    try:
        sample_ifd = sample_ppl_cond / sample_ppl_uncond
    except ZeroDivisionError:
        print("zeroDivisionError ")
        sample_ifd = torch.tensor([0]*batch_size)
    
    
    for i, line in enumerate(batch_data):
        temp_data_i = {
            "id": line["id"],
            "ppl": [0, sample_ppl_uncond[i].item(), 0, sample_ppl_cond[i].item()],
            "loss": [0, sample_loss_uncond[i].item(), 0, sample_loss_cond[i].item()],
            "ifd" : sample_ifd[i].item()
        }
        all_results.append(temp_data_i)


with open(save_path, "w") as f:
    for item in all_results:
        f.write(json.dumps(item) + "\n")
