"""do logit lens object localization for an image"""

from dataclasses import dataclass, field

from PIL import Image
import matplotlib.pyplot as plt
from transformers import HfArgumentParser

from utils import llava_load_model, llava_process_image, llava_generate
from lens_utils import llava_logit_lens

# uv run python llava_logit_lens_loc_demo.py --model_name llava-hf/llava-1.5-7b-hf --image_file 000000008690.jpg


@dataclass
class Arguments:
    """arguments"""

    data_path: str = field(
        default='/home/drdo/vlm-compositionality/data'
    )
    dataset_folder: str = field(
        default='raw/sugarcrepe',
    )
    image_folder: str = field(
        default='raw/coco_val_2017'
    )
    model_name: str = field(default="llava-hf/llava-1.5-13b-hf")
    image_file: str = field(default=None)


if __name__ == "__main__":

    # parse cl arguments
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # append paths
    dataset_folder = args.data_path+'/'+args.dataset_folder
    image_folder = args.data_path+'/'+args.image_folder

    # load image
    # TODO: random image
    assert args.image_file is not None
    image = Image.open(image_folder+'/'+args.image_file)

    # load model, processor
    model, processor = llava_load_model(args.model_name)

    # process image and prompt(default)
    inputs = llava_process_image(image, processor, device=model.device)

    # generate
    outputs = llava_generate(inputs, model)

    # get logit lens
    llava_logit_lens(inputs, model, outputs)


    # TODO: norm before unembedding
    


    plt.show()
