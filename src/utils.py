"""basic functions"""

import json
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor


def load_sugarcrepe(folder_path):
    """load sugarcrepe dataset from local"""

    # add attribute
    with open (folder_path+'add_att.json', encoding='utf8') as f:
        add_attribute = json.load(f)

    # add object
    with open (folder_path+'add_obj.json', encoding='utf8') as f:
        add_object = json.load(f)

    # replace attribute
    with open (folder_path+'replace_att.json', encoding='utf8') as f:
        replace_attribute = json.load(f)

    # replace object
    with open (folder_path+'replace_obj.json', encoding='utf8') as f:
        replace_object = json.load(f)

    # replace relation
    with open (folder_path+'replace_rel.json', encoding='utf8') as f:
        replace_relation = json.load(f)

    # swap attribute
    with open (folder_path+'swap_att.json', encoding='utf8') as f:
        swap_attribute = json.load(f)

    # swap object
    with open (folder_path+'swap_obj.json', encoding='utf8') as f:
        swap_object = json.load(f)

    # collate together
    dataset = {
        'add_attribute': add_attribute, 'add_object': add_object, 
        'replace_attribute': replace_attribute, 'replace_object': replace_object, 
        'replace_relation': replace_relation, 'swap_attribute': swap_attribute, 
        'swap_object': swap_object,
    }

    return dataset


def llava_load_model(model_name, flash_attention=True):
    """load llava model"""
    if flash_attention:
        model =  LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        model =  LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def llava_process_image(image, processor, device, prompt='default'):
    """Process image and prompt into input for llava"""  
    if prompt == 'default':
        prompt = "USER: <image>\nDescribe the image. ASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    return inputs


def llava_forward_pass(inputs, model, output_hidden_states=False, output_attentions=False):
    """Forward pass through the model to collect logits, hidden_states, attentions"""
    output = model(
        **inputs,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    logits = output['logits']
    hidden_states = output['hidden_states'] if output_hidden_states else None
    attentions = output['attentions'] if output_attentions else None
    return logits, hidden_states, attentions


def llava_generate(
        inputs,
        model,
        output_hidden_states=True,
        num_beams=5,
        temperature=1.0,
        max_new_tokens=512
    ):
    """Generate to collect logits, hidden_states, attentions"""
    output = model.generate(
        **inputs,
        temperature=temperature,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        output_hidden_states=output_hidden_states,
        return_dict_in_generate=True,
    )
    return output

# End-of-file (EOF)
