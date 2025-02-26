import os
import torch
import numpy as np
import gdown
import shutil
import gradio as gr

from PIL import Image
from pathlib import Path
from uuid import uuid4
from argparse import Namespace

# NOTE change this to select other models
MODEL = "ttpla-resnet50-640x360"
ARGS = Namespace(
    # See defaults on eval.py, put some adaptations of my own
    display_lincomb=False,
    crop=True,  # by default --no_crop is False in eval.py
    score_threshold=0,
    display_masks=True,
    display_fps=False,
    display_bboxes=True,
    display_text=True,
    display_scores=True,
    top_k=5,
)

# See these links at https://github.com/R3ab/ttpla_dataset
MODEL_HUB = {
    "ttpla-resnet50-640x360": {
        "checkpoint_gdrive_id": "1arVhEIz_DQ-1wALSk9S3TJwCFzOWPNik",
        "checkpoint_path": "yolact_img640_secondval_399_30000_resnet50.pth",
        "config_gdrive_id": "1ocoYiTDFBcdI8Es8dZlMbsbFGkaLKw98",
        "config_path": "config_img640_resnet50_aspect.py",
        "config_name": "yolact_img640_test_config",
    },
    "ttpla-resnet101-700x700": {
        "checkpoint_gdrive_id": "1IDfQlBJ2VAIpyaOSUs2Ecmf_rsl8nSdc",
        "checkpoint_path": "yolact_img700_399_45100_resnet101_b8.pth",
        "config_gdrive_id": "1QfPvi2FTJv1JByqM70qM7nQjGpNI_kNi",
        "config_path": "config_img700_resnet101.py",
        "config_name": "yolact_img700_test_config",
    },
    "yolact_plus-resnet101-550x550": {
        "checkpoint_gdrive_id": "15id0Qq5eqRbkD-N3ZjDZXdCvRyIaHpFB",
        "checkpoint_path": "yolact_plus_base_54_800000.pth",
        "config_gdrive_id": "",
        "config_path": "data/config.py",
        "config_name": "yolact_plus_base_config",
    }
}


# install model checkpoint
def get_checkpoint() -> str:
    yolact_checkpoint_path = Path(__file__).parent / MODEL_HUB[MODEL]["checkpoint_path"]
    yolact_checkpoint_gdrive_id = MODEL_HUB[MODEL]["checkpoint_gdrive_id"]
    yolact_config_path = Path(__file__).parent / MODEL_HUB[MODEL]["config_path"]
    yolact_config_gdrive_id = MODEL_HUB[MODEL]["config_gdrive_id"]
    yolact_test_config_name = MODEL_HUB[MODEL]["config_name"]

    if not os.path.exists(yolact_checkpoint_path):
        gdown.download(id=yolact_checkpoint_gdrive_id, fuzzy=True)
    else:
        print(f"Checkpoint file {yolact_checkpoint_path} already exists. Skipping download")

    if not os.path.exists(yolact_config_path):
        gdown.download(id=yolact_config_gdrive_id, fuzzy=True)
    else:
        print(f"Config file {yolact_config_path} already exists. Skipping download")

    if yolact_config_path != Path(__file__).parent / "data" / "config.py":
        shutil.copy(yolact_config_path, Path(__file__).parent / "data" / "config.py")

    # NOTE: global variable cfg must be changed in set_cfg BEFORE other modules import it
    # for the changes to be reflected, since "from ... import x"-style imports initialize a copy of the variable
    # SEE: https://stackoverflow.com/questions/15959534/visibility-of-global-variables-in-imported-modules
    # NOTE 2: yolact repo can't be a module either for some reason
    from data.config import set_cfg, resnet50_backbone
    set_cfg(yolact_test_config_name)
    print(f"Config globally set to {yolact_test_config_name}")
    # some cfg attributes are not set
    set_cfg("{'mask_proto_debug':    False}")
    return str(yolact_test_config_name), str(yolact_checkpoint_path)

yolact_config_name, yolact_checkpoint_path = get_checkpoint()


### YOLACT SPECIFIC CODE

import eval
from yolact import Yolact
from eval import evalimage

eval.args = ARGS  # HACK: change global variable args in eval code

def describe_network(net: torch.nn.Module) -> None:
    for name, tensor in net.named_parameters():
        print(f"{name}: {tensor.shape}")

def load_model() -> Yolact:
    net = Yolact()
    net.load_weights(yolact_checkpoint_path)
    net = net.cuda()
    net.eval()  # PUT INTO EVAL MODE!
    num_parameters = sum([int(np.prod(t.shape)) for t in net.parameters()])
    print(f"Yolact from config {yolact_config_name}, weights at '{yolact_checkpoint_path}' loaded with {num_parameters} parameters")
    return net

NET = load_model()
UUID = uuid4()
INPUT_TEMPFILE = Path(f"/tmp/{UUID}/input.png")
OUTPUT_TEMPFILE = Path(f"/tmp/{UUID}/output.png")

def process_image(image: Image.Image) -> Image.Image:
    for fpath in [INPUT_TEMPFILE, OUTPUT_TEMPFILE]:
        os.makedirs(fpath.parent, exist_ok=True)
        if os.path.exists(fpath):
            os.remove(fpath)
    image.save(INPUT_TEMPFILE, format="png")
    with torch.no_grad():
        evalimage(NET, INPUT_TEMPFILE, OUTPUT_TEMPFILE)
    img = Image.open(OUTPUT_TEMPFILE)
    for fpath in [INPUT_TEMPFILE, OUTPUT_TEMPFILE]:
        if os.path.exists(fpath.parent):
            shutil.rmtree(fpath.parent)
    return img

with gr.Blocks() as demo:
    gr.Markdown("# Image Processing Demo")
    with gr.Row():
        inp = gr.Image(type="pil", label="Input Image")
        out = gr.Image(type="pil", label="Output Image")
    btn = gr.Button("Process")
    btn.click(process_image, inputs=inp, outputs=out)

demo.launch()
