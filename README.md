# Capstone Additions

The project is about adding command line automation to Koyha's stable diffustion fine tunnings. 

By: Zackary Hopkins, Nithish Maley, Surya Dendukuru

[Link](https://docs.google.com/presentation/d/1LYh7AD-wfc8BU77FLVdASIxkENsxsa_fGWVj8gc4bh4/edit?usp=sharing) to our final presentation about this project

Here are the softwear versions we used to run this on a linx 
* Python Version: 3.10.12
* Cuda Version: 11.4
    * Driver Version: 470.223.02
* Koyha commit hash : 9eb5da23002c6e00f084a19d62da6fc82074c773

## Files that are added

* AI_Cap_finetune_lora.py - Koyha automation code
* AI_Cap_clip.py - Image subset generation code
* AI_Cropping.py - Image bluring and cropping with grounding dino model code
* default_config.json - this is a manual made json obj of the default parameters of Koyha's parameters
* Test_Command.txt - A list of different test commands we used 

# How to Run
Before running a few steps are need to run the .\AI_cap_finetune_lora.py script 

1. First run the setup file from kohya_ss

For Windows:
.\setup.bat

For Linux:
.\setup.sh

2. cd into AI_Cap_kohya_ss and run the virtual enviroment

For Windows:
```
.\venv\Scripts\activate.bat
```

For Linux:
```
source venv/bin/activate
```

3. run this command from the python venv

Both Windows and Linux:
```
pip install git+https://github.com/openai/CLIP.git
```

Now you can run .\AI_Cap_finetune_lora.py from inside the venv Bellow is the example and some descriptions of the function

## AI_Cap_finetune_lora - Koyha Automation Script

```
python3 .\AI_Cap_finetune_lora.py --prompt str --img_Filepath "path\to\images" --number_of_subset int --img_subset_output "path\to\img\output" --output_dir "path\to\model\output" --logging_dir "path\to\logging\output" --config_Filepath "path\to\json\configs" --output_name "Model_Name" -learning_rate float -learning_rate_stop float -learning_rate_step float
```

Agruments

prompt - (str) this is descrioption of the object in your images you want to train on

img_Filepath - (str) where the images you want to train are

number_of_subset - (int)(default=5) The size of the subset of images you want to pull out

img_subset_output - (str)(optional) an output location you would like to have hold that subset

output_dir - (str) the output location for the model

logging_dir - (str) the file location for the logging of the training

config_Filepath - (str)(optional) The file path too different json configs

output_name - (str) the name of the model outputs. The naming convention will be the given model name plus underscore number run. Ex we have two runs so New_model_1 and New_model_2

batch_size - (int)(default=1) the batch size of the blip captioning

num_beams - (int)(default=1) The number of beams for the blip captioning

top_p - (float)(default=.9) The top p values for blip captioning: [0,1]

max_length - (int)(default=75) The max lengths of captions for blip captioning

min_length - (int)(default=5) The min lengths of captions for blip captioning

beam_search - (bool)(default=True) Boolean for wanting to use beam_search for blip captioning
   
prefix - (str)(default='default123') The Prefix to add to the blip captioning

postfix - (str)(default='') The Post fix to add to the blip captioning

Epoch - (int)(default=1) The number of epochs the model will train for

learning_rate - (float)(default=1e-05) the starting learning rate for the model training 

learning_rate_stop - (float)(default=1e-05) the upper bound of the learning rate range 

learning_rate_step - (float)(default=1) the step size of the learning rate to generate training learning rates

optimizer_type - (string)(default="AdamW8bit") the list of optimizers to use. Inputed as one line separated by "," Ex: AdamW8bit, Adafactor

## Clip Subset Code

Overview :
This project leverages the CLIP (Contrastive Language-Image Pre-training) model to perform image retrieval based on a query image, text input, or both. Given a dataset of images, the model identifies and retrieves the most similar images to the provided query.


Requirements :

numpy
torch
packaging
transformers
Pillow
torchvision
matplotlib
git+https://github.com/openai/CLIP.git


prompt2Img_subset:

This method is used to retrieve most similar images for a given text input.
It takes text prompt, input folder, output folder and number of images as parameters and saves the output in the output folder.

img2img_subset:

This method is used to retrieve most similar images for a image input. It takes image path, input folder, output folder and number of images as parameters and saves the output in the output folder.

## Grounding Dino Bluring and Cropping Code



# Other Repos we used
* ## [kohya_ss](https://github.com/bmaltais/kohya_ss)
* ## [Clip Repo](https://github.com/openai/CLIP)
* ## [Grounding Dino](https://github.com/IDEA-Research/GroundingDINO.git)

