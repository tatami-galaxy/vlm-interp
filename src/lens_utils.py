"""logit lens and tuned lens functions"""

import torch
from transformers.generation.logits_process import TopKLogitsWarper
from transformers.generation.logits_process import LogitsProcessorList


def llava_get_image_representations(inputs, hidden_states, config, image_token_index=None):
    """get image representations from llava hidden_states"""
    if image_token_index is None:
        image_token_index = config.image_token_index
    image_indices = torch.nonzero((inputs['input_ids'][0] == image_token_index).long())


def llava_logit_lens(inputs, model, outputs):
    """get logit lens distribution over vocab"""

    assert outputs['hidden_states'] is not None
    input_ids = inputs['input_ids']
    logits_warper = TopKLogitsWarper(top_k=50, filter_value=float("-inf"))
    logits_processor = LogitsProcessorList([])

    # first forward pass
    hidden_states = torch.stack(outputs.hidden_states[0])
    
    with torch.inference_mode():
        curr_layer_logits = model.language_model.lm_head(hidden_states).cpu().float()
        print(curr_layer_logits.shape)
        quit()
        logit_scores = torch.nn.functional.log_softmax(curr_layer_logits, dim=-1)
        logit_scores_processed = logits_processor(input_ids, logit_scores)
        logit_scores = logits_warper(input_ids, logit_scores_processed)
        softmax_probs = torch.nn.functional.softmax(logit_scores, dim=-1)
