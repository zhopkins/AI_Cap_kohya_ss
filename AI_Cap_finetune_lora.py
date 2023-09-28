import argparse

def finetunr_lora(prompt, imgFilepath, configFilepath):
    """
    This function finetunes the number of lora model based on the number of Json config files given. 
        1. makes files structure out of image fold
        2. do auto caption if need be
        3. loop through json files
            3.1. for each one run the training command for the lora
    """
    #use clip function to get subset
    image_set = clip_func(prompt, imgFilepath)
    


    return 0

def clip_func(prompt, imgFilepath):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #input aguments for comand line functions
    parser.add_argument(
        '-p',
        '--prompt',
        type=str,
        help='The prompt of the focus for the finetuning',
    )

    parser.add_argument(
        '-i',
        '--Images',
        type=str,
        help='The file path too the folder of images',
    )

    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='The file path too the folder of config files',
    )

    