# Capstone Additions
Todo for documentation
1. what are the files we made and what do they do
2. how do they work together
3. challenges


##How to Run
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

python3 .\AI_Cap_finetune_lora.py --prompt str --img_Filepath "path\to\images" --number_of_subset int --img_subset_output "path\to\img\output" --output_dir "path\to\model\output" --logging_dir "path\to\logging\output" --config_Filepath "path\to\json\configs" --output_name "Model_Name" -learning_rate float -learning_rate_stop float -learning_rate_step float

Agruments

prompt - (str) this is descrioption of the object in your images you want to train on

img_Filepath - (str) where the images you want to train are

number_of_subset - (default=5)(int) The size of the subset of images you want to pull out

img_subset_output - (optional)(str) an output location you would like to have hold that subset

output_dir - (str) the output location for the model

logging_dir - (str) the file location for the logging of the training

config_Filepath - (optional)(str) The file path too different json configs

output_name - (str) the name of the model outputs. The naming convention will be the given model name plus underscore number run. Ex we have two runs so New_model_1 and New_model_2

learning_rate - (float) the starting learning rate for the model training 

learning_rate_stop - (float) the upper bound of the learning rate range 

learning_rate_step - (float) the step size of the learning rate to generate training learning rates

optimizer_type - (string) the list of optimizers to use. Inputed as one line separated by "," Ex: AdamW8bit, Adafactor


# For more information go to the parent repo [from kohya_ss gihub](https://github.com/bmaltais/kohya_ss)

