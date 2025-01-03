"""logit lens and tuned lens functions"""

import torch
from transformers.generation.logits_process import TopKLogitsWarper


def llava_logit_lens(inputs, model, outputs, topk=50, norm=False):
    """get logit lens distribution over vocab"""

    assert outputs['hidden_states'] is not None
    input_ids = inputs['input_ids']
    logits_warper = TopKLogitsWarper(top_k=topk, filter_value=float("-inf"))
    image_token_id = model.config.image_token_index

    # first forward pass
    hidden_states = torch.stack(outputs.hidden_states[0])
    
    with torch.inference_mode():
        if norm:
            hidden_states[:-1, :, :, :] = model.language_model.model.norm(hidden_states[:-1, :, :, :])
        curr_layer_logits = model.language_model.lm_head(hidden_states).cpu().float()
        logit_scores = torch.nn.functional.log_softmax(curr_layer_logits, dim=-1)
        # remove all tokens with a probability less than the last token of the top-k
        logit_scores = logits_warper(input_ids, logit_scores)
        # softmax -> re-distribute probability mass
        softmax_probs = torch.nn.functional.softmax(logit_scores, dim=-1)
        softmax_probs = softmax_probs.detach().cpu().numpy()

        # get scores of image tokens
        image_token_index = input_ids.tolist()[0].index(image_token_id)
        softmax_probs = softmax_probs[
            :, :, image_token_index : image_token_index + (24 * 24) # 576 -> compute differently?
        ]
        
        # transpose to (vocab_dim, num_layers, num_tokens, num_beams)
        softmax_probs = softmax_probs.transpose(3, 0, 2, 1)

        # maximum over all beams
        # vocab_dim, num_layers, num_tokens
        softmax_probs = softmax_probs.max(axis=3)

        return softmax_probs
