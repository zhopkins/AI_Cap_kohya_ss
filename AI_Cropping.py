#install all 3 og these below

# !pip install diffusers transformers accelerate scipy safetensors

# !pip install -e .

# !git clone https://github.com/IDEA-Research/GroundingDINO.git

import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import os
os.chdir('/home/capstone/Desktop/fall2023/AI_Cap_kohya_ss')


# # If you have multiple GPUs, you can set the GPU to use here.
# # The default is to use the first GPU, which is usually GPU 0.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

from huggingface_hub import hf_hub_download
import cv2
import numpy as np

import torch

from PIL import Image




def cropping(local_image_path):


    def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model

    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    # import io


    # def download_image(url, image_file_path):
    #     r = requests.get(url, timeout=4.0)
    #     if r.status_code != requests.codes.ok:
    #         assert False, 'Status code error: {}.'.format(r.status_code)

    #     with Image.open(io.BytesIO(r.content)) as im:
    #         im.save(image_file_path)

    #     print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))

    # # download_image(image_url, local_image_path)



    TEXT_PROMPT = "man"
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(local_image_path)

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    Image.fromarray(image_source)


    Image.fromarray(annotated_frame)

    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    # Ensure you have the bounding boxes in the correct format (xyxy).
    # This is done as you have defined it: boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    # Convert the boxes to numpy format for easier manipulation.

    boxes_xyxy = boxes_xyxy.numpy()

    # Loop through each bounding box and crop the image.
    cropped_images = []
    for box in boxes_xyxy:
        # Convert the box coordinates to integers to use as slicing indices.
        box = box.astype(int)

        # Crop the image using the box coordinates.
        cropped_image = image_source[box[1]:box[3], box[0]:box[2]]

        # Append the cropped image to the list.
        cropped_images.append(cropped_image)

    # Now you have a list of cropped images, each corresponding to a bounding box.
    # You can further process, save, or display these cropped images as needed.


    Image.fromarray(cropped_image)

    dim = boxes_xyxy[0].tolist()
    dim
    dim = [int(item) for item in dim]

    path = local_image_path




    # Load your image
    image = cv2.imread(path)

    # Define the coordinates of the box (x, y, width, height)
    box_coordinates = (dim)  # Adjust these values according to your specific box

    # Copy the original image to work on
    result = image.copy()

    # Create a mask for the entire image filled with zeros
    mask = np.zeros_like(image, dtype=np.uint8)

    # Define the coordinates of the box region
    x, y, w, h = box_coordinates

    # Create a white rectangle in the mask for the box
    cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), thickness=cv2.FILLED)

    # Create a blurred version of the entire image
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=5)

    # Combine the blurred image with the original image inside the box
    res = cv2.bitwise_and(image, mask) + cv2.bitwise_and(blurred_image, 255 - mask)
    blurr = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    

    return cropped_images,blurr

