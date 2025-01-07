from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils import llava_load_model, llava_process_image, llava_generate
from lens_utils import llava_logit_lens