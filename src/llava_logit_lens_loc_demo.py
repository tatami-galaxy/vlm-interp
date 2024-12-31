"""do logit lens object localization for an image"""

from dataclasses import dataclass, field

from PIL import Image
import matplotlib.pyplot as plt
from transformers import HfArgumentParser

from utils import llava_load_model, llava_process_image, llava_forward_pass


@dataclass
class Arguments:
    """arguments"""

    dataset_path: str = field(
        default='/home/drdo/vlm-compositionality/data/raw/sugarcrepe',
    )
    image_folder: str = field(
        default='/home/drdo/vlm-compositionality/data/raw/coco_val_2017'
    )
    model_name: str = field(default="llava-hf/llava-1.5-13b-hf")
    image_file: str = field(default=None)


if __name__ == "__main__":

    # parse cl arguments
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # load image
    # TODO: random image
    assert args.image_file is not None
    image = Image.open(args.image_folder+'/'+args.image_file)

    # load model, processor
    model, processor = llava_load_model(args.model_name)

    # process image and prompt(default)
    inputs = llava_process_image(image, processor, device=model.device)

    # forward pass
    _, hidden_states, _ = llava_forward_pass(inputs, model, output_hidden_states=True)

    # TODO: norm before unembedding
    


    plt.show()
# End-of-file (EOF)