import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor


def llava_load_model(model_name, flash_attention=True, torch_dtype=torch.bfloat16):
    """load llava model"""
    # TODO: dtype
    if flash_attention:
        model =  LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        model =  LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_name)
    #processor.patch_size = 14
    #processor.vision_feature_select_strategy = 'default'

    return model, processor


def llava_process_image(image, processor, device, prompt='default'):
    """Process image and prompt into input for llava"""  
    if prompt == 'default':
        prompt = "USER: <image>\nDescribe the image. ASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    return inputs


def llava_forward_pass(inputs, model, output_hidden_states=False, output_attentions=False):
    """Forward pass through the model to collect logits, hidden_states, attentions"""
    with torch.no_grad():
        output = model(
            **inputs,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
    return output


def llava_generate(
        inputs,
        model,
        output_hidden_states=False,
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

