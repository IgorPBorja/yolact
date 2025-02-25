# NOTE change this to select other models
MODEL = "yolact_plus-resnet101-550x550"

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

import os
import torch
import numpy as np
import gdown
import shutil
import gradio as gr

from PIL import Image
from matplotlib import cm
from pathlib import Path

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

from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess as postprocess_raw_yolact_output

def describe_network(net: torch.nn.Module) -> None:
    for name, tensor in net.named_parameters():
        print(f"{name}: {tensor.shape}")

def load_model() -> Yolact:
    net = Yolact()
    net.load_weights(yolact_checkpoint_path)
    num_parameters = sum([int(np.prod(t.shape)) for t in net.parameters()])
    print(f"Yolact from config {yolact_config_name}, weights at '{yolact_checkpoint_path}' loaded with {num_parameters} parameters")
    return net

def evalimage(net: Yolact, img: np.ndarray):  # BGR opencv image
    frame = torch.from_numpy(img).cuda().float()
    net = net.cuda()
    net.eval()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    with torch.no_grad():
        raw_preds = net(batch)
    # NOTE skipping interpolation to original width/height and using totally raw output
    h, w = batch.shape[-2:]
    classes, scores, boxes, masks = postprocess_raw_yolact_output(raw_preds, w, h)
    return classes, scores, boxes, masks

def resize_and_overlay_masks(masks: torch.Tensor, new_size) -> np.ndarray[np.uint8]:
    if (len(masks.shape) < 3):    # allows to deal with a single mask
        masks = masks.unsqueeze(0)
    num_masks = len(masks)
    colormap = cm.get_cmap("tab10", num_masks)    # Choose a colormap
    colors = torch.tensor(colormap(np.arange(num_masks))[:, :3] * 255, dtype=torch.uint8).to(device=masks.device) # Convert to RGB (0-255)
    # print(colors)
    # masks = F.interpolate(masks.unsqueeze(0).float(), size=new_size, mode="bilinear", align_corners=False).squeeze()
    # masks = (resized_masks > 0.5).float()

    # Expand for broadcasting and apply colors
    colored_masks = masks.unsqueeze(-1) * colors.view(num_masks, 1, 1, 3)    # Shape (num_masks, H, W, 3)

    # Merge masks (taking max color per pixel to avoid overwrites)
    final_image = colored_masks.max(dim=0)[0].cpu().numpy().astype(np.uint8)
    return final_image

def process_image(image: Image.Image) -> Image.Image:
    torch.cuda.empty_cache()
    net = load_model()
    classes, scores, boxes, masks = evalimage(net, np.array(image))
    # clean /tmp/masks
    if os.path.isdir("/tmp/masks"):
        shutil.rmtree("/tmp/masks")
    os.makedirs("/tmp/masks")
    for i, mask in enumerate(masks):
        fpath = f"/tmp/masks/{i}.jpeg"
        numpy_mask = mask.unsqueeze(dim=0).repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()  # 0/1 float array, shape (H, W, 3)
        bin_img = Image.fromarray((numpy_mask * 255).astype(np.uint8), mode="RGB")
        bin_img.save(fpath, format="jpeg")
    print(f"Saved all {len(masks)} segmentation masks under /tmp/masks")

    combined_masks = resize_and_overlay_masks(masks, 2000)
    # Example processing: Convert image to grayscale
    return Image.fromarray(combined_masks)

with gr.Blocks() as demo:
    gr.Markdown("# Image Processing Demo")
    with gr.Row():
        inp = gr.Image(type="pil", label="Input Image")
        out = gr.Image(type="pil", label="Output Image")
    btn = gr.Button("Process")
    btn.click(process_image, inputs=inp, outputs=out)

demo.launch()
