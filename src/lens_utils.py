"""logit lens and tuned lens functions"""

import torch


def llava_get_image_representations(inputs, hidden_states, config, image_token_index=None):
    """get image representations from llava hidden_states"""
    if image_token_index is None:
        image_token_index = config.image_token_index
    image_indices = torch.nonzero((inputs['input_ids'][0] == image_token_index).long())



# End-of-file (EOF)
