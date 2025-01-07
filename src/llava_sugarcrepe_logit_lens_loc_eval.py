"""do logit lens object localization for an image"""

from dataclasses import dataclass, field

from PIL import Image
import matplotlib.pyplot as plt

from transformers import HfArgumentParser
from pycocotools.coco import COCO

from utils import(
    llava_load_model, 
    llava_process_image, 
    llava_generate,
    load_sugarcrepe,
)
from lens_utils import llava_logit_lens


@dataclass
class Arguments:
    """arguments"""

    project_folder: str = field(
        default='/home/drdo/vlm-compositionality'
    )
    dataset_folder: str = field(
        default=None,
    )
    image_folder: str = field(
        default=None
    )
    ann_folder: str = field(
        default=None,
    )
    ann_file: str = field(
        default=None,
    )
    model_name: str = field(default="llava-hf/llava-1.5-7b-hf")


if __name__ == "__main__":

    # parse cl arguments
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # append paths
    # dataset
    if args.dataset_folder is None:
        dataset_folder = args.project_folder+'/data/raw/sugarcrepe'
    else: 
        dataset_folder = args.dataset_folder
    # coco images
    if args.image_folder is None:
        image_folder = args.project_folder+'/data/raw/coco/val2017'
    else:
        image_folder = args.image_folder
    # coco annotations
    if args.ann_folder is None:
        ann_folder = args.project_folder+'/data/raw/coco/annotations'
    else:
        ann_folder = args.ann_folder
    if args.ann_file is None:
        ann_file = ann_folder+'/instances_val2017.json'

    # load sugarcrepe
    sugarcrepe = load_sugarcrepe(dataset_folder)

    # load annotations
    coco=COCO(ann_file)
    coco_ids = list(coco.anns.keys())

    # TODO: filter images
    

    # load model, processor
    #model, processor = llava_load_model(args.model_name)

    # process image and prompt(default)
    #inputs = llava_process_image(image, processor, device=model.device)

    # generate
    #outputs = llava_generate(inputs, model)

    # get logit lens
    # vocab_dim, num_layers, num_tokens
    #softmax_probs = llava_logit_lens(inputs, model, outputs)


    #plt.show()