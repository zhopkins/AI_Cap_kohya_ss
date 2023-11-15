import clip
import numpy as np
import torch
#from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np



def subset_Images(text, ip, op, number_of_imgs):
    #checks the device avaiable 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the CLIP model and processor

    model, preprocess = clip.load("ViT-B/32")
    model.to(device)
    model.eval()


    # text_descriptions = "a picture of a cat"
    text_tokens = clip.tokenize(text)
    text_tokens = text_tokens.to(device)
    t_dir = ip
    ims = []


    image_list = [os.path.join(t_dir, filename) for filename in os.listdir(t_dir) if filename.endswith(".png") or filename.endswith(".jpg")]
    for filename in image_list:
        image = Image.open(filename).convert("RGB")
        ims.append(preprocess(image))


    image_input = torch.tensor(np.stack(ims))
    image_input = image_input.to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    #checks to move it back to cpu if need be    
    text_features = text_features.cpu()
    image_features = image_features.cpu()

    similarity = text_features.numpy() @ image_features.numpy().T


    sorted_indices = np.argsort(similarity).tolist()[0]
    #check if the number of images wanted is larger than the list itself
    if len(sorted_indices) < number_of_imgs:
         number_of_imgs = len(sorted_indices)
    tl = sorted_indices[-number_of_imgs:]

    for filename in os.listdir(op):
            file_path = os.path.join(op, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for i in tl:
        image_path = image_list[i]
        highest_similarity_image = Image.open(image_path).convert("RGB")
        save_folder = op
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, image_path.split('/')[-1])
        highest_similarity_image.save(save_path)

    print(f"{len(tl)} Images saved")


# txt = input("Enter text input: ")
#input_folder = "animals"
#output_folder = "output_images"
#subset_Images('a picture of a dog',input_folder,output_folder)
