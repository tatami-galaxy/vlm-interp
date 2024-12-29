"""basic functions"""

from PIL import Image


def llava_process_image(image_file, processor, device, prompt='default'):
    """Process image and prompt into input for llava"""  
    image = Image.open(image_file)
    if prompt == 'default':
        prompt = "USER: <image>\nDescribe the image. ASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    return inputs


def forward_pass(inputs, model, output_hidden_states=True, output_attentions=True):
    """Forward pass through the model to collect logits, hidden_states, attentions"""
    assert input.device == model.device
    output = model(
        **inputs,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    logits = output['logits']
    hidden_states = output['hidden_states'] if output_hidden_states else None
    attentions = output['attentions'] if output_attentions else None
    return logits, hidden_states, attentions


# End-of-file (EOF)
