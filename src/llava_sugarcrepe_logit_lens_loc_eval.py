"""do logit lens object localization for an image"""

from dataclasses import dataclass, field

from PIL import Image
from sklearn.metrics import f1_score

from transformers import HfArgumentParser
from pycocotools.coco import COCO


from data_utils import(
    load_sugarcrepe,
    coco_cats,
    coco_object_mask,
    filter_sugarcrepe_distict_objects,
)

from llava_utils import(
    llava_load_model, 
    llava_process_image, 
    llava_generate,
)
from lens_utils import(
    llava_logit_lens,
    get_mask_from_lens,
)


@dataclass
class Arguments:
    """arguments"""

    project_dir: str = field(
        default='/home/drdo/vlm-compositionality'
    )
    dataset_dir: str = field(
        default=None,
    )
    image_dir: str = field(
        default=None
    )
    ann_dir: str = field(
        default=None,
    )
    ann_file: str = field(
        default=None,
    )
    model_name: str = field(default="llava-hf/llava-1.5-7b-hf")
    topk: int = field(default=5)
    num_patches: int = field(default=24)


if __name__ == "__main__":

    # parse cl arguments
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # conditionally append paths
    # dataset
    dataset_dir = args.project_dir+'/data/raw/sugarcrepe' if args.dataset_dir is None else args.dataset_dir
    # coco images
    image_dir = args.project_dir+'/data/raw/coco/val2017' if args.image_dir is None else args.image_dir
    # coco annotations folder
    ann_dir = args.project_dir+'/data/raw/coco/annotations' if args.ann_dir is None else args.ann_dir
    # coco annotations file
    ann_file = ann_dir+'/instances_val2017.json' if args.ann_file is None else args.ann_file

    # load sugarcrepe
    sugarcrepe = load_sugarcrepe(dataset_dir)

    # load annotations
    coco=COCO(ann_file)

    # get coco image ids
    image_ids = coco.getImgIds()

    # filter images
    image_ids = filter_sugarcrepe_distict_objects(coco, image_ids)

    # load model, processor
    model, processor = llava_load_model(args.model_name)

    # eval loop
    # get image, get all object tokens for the image
    # for each object token generate logit lens mask and compare with coco mask
    for image_id in image_ids:

        # image info
        image_info = coco.loadImgs(image_id)[0]
        image_file = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        # object tokens
        tokens = coco_cats(coco, image_id)

        # get coco masks
        token_to_mask = coco_object_mask(coco, image_id)

        # load image
        image = Image.open(image_dir+'/'+image_file).convert("RGB")

        # process image and prompt(default)
        inputs = llava_process_image(image, processor, device=model.device)

        # generate
        outputs = llava_generate(inputs, model)

        # get logit lens
        # vocab_dim, num_layers, num_tokens
        # TODO: what if token not in topk?
        softmax_probs = llava_logit_lens(inputs, model, outputs, topk=args.topk)

        # compare for each token

        for token in tokens:
            # get non zero mask from lens
            ll_mask = get_mask_from_lens(
                softmax_probs,
                token,
                processor,
                args.num_patches,
                image_width, image_height
            )

            # compare coco mask and logit lens mask
            coco_mask = token_to_mask[token]

            # f1 score to estimate overlap
            f1 = f1_score(coco_mask.flatten(), ll_mask.flatten())
            print("f1 score for {} : {}".format(token, f1))
            quit()



        


