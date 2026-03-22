import os
import random
import numpy as np

import torch

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_gcp_env(args):
    os.environ["GOOGLE_CLOUD_LOCATION"] = args.GOOGLE_CLOUD_LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = args.GOOGLE_GENAI_USE_VERTEXAI
    os.environ["TOKENIZERS_PARALLELISM"] = args.TOKENIZERS_PARALLELISM
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.GOOGLE_CLOUD_PROJECT


dx_task_measurement = ['rotation', 'projection', 'cardiomegaly',
                       'mediastinal_widening', 'carina_angle', 'aortic_knob_enlargement',
                       'descending_aorta_enlargement', 'descending_aorta_tortuous']

dx_task_multi_bodyparts = ['inspiration', 'rotation', 'projection', 'cardiomegaly',
                           'trachea_deviation', 'mediastinal_widening', "aortic_knob_enlargement",
                           "ascending_aorta_enlargement", "descending_aorta_enlargement"]