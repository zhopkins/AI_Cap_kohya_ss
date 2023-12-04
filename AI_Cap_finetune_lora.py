import argparse
import os
import shutil
import pathlib
import itertools 
from AI_Cap_clip import prompt2Img_subset,img2img_subset
from library.blip_caption_gui import caption_images
from lora_gui import train_model
from library.custom_logging import setup_logging
from library.class_command_executor import CommandExecutor
import json
#from GroundingDINO.ai_cropping import 

# Set up logging
log = setup_logging()

# Setup command executor
executor = CommandExecutor()

def floating_point_range(start, stop, step):
    range_list = [start]
    new_num = range_list[-1] + step
    while new_num < stop:
        range_list.append(new_num)
        new_num = range_list[-1] + step
    return range_list

#Function for creating Congiguration list
def create_configs(args_dic):
    """
    This function taks in a dictionary of arguments and create a list of configuerations for loar training.
    """
    config_list = []

    #makes that list of variables desired
    #this is for learning Rate
    lr_list = floating_point_range(args_dic["learning_rate"], args_dic["learning_rate_stop"], args_dic["learning_rate_step"])
    #This is for Optimizer
    #if the optimizer is in this list it is in kohya_ss's optimizers    
    #Should find a way to not hard code but no easy way found
    opt_choices=[
                    'AdamW',
                    'AdamW8bit',
                    'Adafactor',
                    'DAdaptation',
                    'DAdaptAdaGrad',
                    'DAdaptAdam',
                    'DAdaptAdan',
                    'DAdaptAdanIP',
                    'DAdaptAdamPreprint',
                    'DAdaptLion',
                    'DAdaptSGD',
                    'Lion',
                    'Lion8bit',
                    'PagedAdamW8bit',
                    'PagedLion8bit',
                    'Prodigy',
                    'SGDNesterov',
                    'SGDNesterov8bit',
                ]
    opt_list = [opt for opt in args_dic["optimizer_type"].replace(' ', '').split(',') if opt in opt_choices]
    


    with open('./default_config.json', 'r') as f:
        default_config = json.load(f)
        name_count = 1
        for lr,opt in itertools.product(lr_list, opt_list):
            #creates copies of the dics to not interfere with other loops
            new_config = default_config.copy()
            input_copy = args_dic.copy()
            #addes the products from the output
            input_copy["learning_rate"] = lr
            input_copy["optimizer_type"] = opt
            input_copy["output_name"] = args_dic["output_name"] + "_" + str(name_count)

            #setup the default with keys
            for key in input_copy.keys():
                new_config[key] = input_copy[key]
            #adds the new config
            config_list.append(new_config)
            name_count += 1



    return config_list


    
#opens a file dir of json configurations
def get_configs(imgFilepath, output_dir, logging_dir, configFilepath):
    #get all the json files
    json_list = [f for f in pathlib.Path(configFilepath).iterdir() if f.is_file()]
    print('Found {} config files: {}'.format(len(json_list), json_list))
    config_list = []
    for filename in json_list:
        with open(filename,'r') as f:
            new_config = json.load(f)
            new_config['img_Filepath'] = imgFilepath
            new_config['output_dir'] = output_dir
            new_config['logging_dir'] = logging_dir
            config_list.append(new_config)
    #takes the configs through the loops
    return config_list




def lora_loop(config_list):

    #run each lora training
    print(f"There are {len(config_list)} models to train.")
    for new_config in config_list:
        print('\nStarted Lora Traning on ', new_config['output_name'], '...\n')
            
        train_model(
        headless={'label':'False'},#0######
        print_only={'label':'False'},#Make True Change if you want to print comands in stead of running#
        pretrained_model_name_or_path=new_config['pretrained_model_name_or_path'],
        v2=new_config['v2'],
        v_parameterization=new_config['v_parameterization'],
        sdxl=new_config['sdxl'],
        logging_dir=new_config['logging_dir'],
        train_data_dir=new_config['img_Filepath'], ##Need to test if file path exists
        reg_data_dir=new_config['reg_data_dir'],
        output_dir=new_config['output_dir'],
        max_resolution=new_config['max_resolution'],
        learning_rate=new_config['learning_rate'],
        lr_scheduler=new_config['lr_scheduler'],
        lr_warmup=new_config['lr_warmup'],
        train_batch_size=new_config['train_batch_size'],
        epoch=new_config['Epoch'],
        save_every_n_epochs=new_config['save_every_n_epochs'],
        mixed_precision=new_config['mixed_precision'],
        save_precision=new_config['save_precision'],
        seed=new_config['seed'],
        num_cpu_threads_per_process=new_config['num_cpu_threads_per_process'],
        cache_latents=new_config['cache_latents'],
        cache_latents_to_disk=new_config['cache_latents_to_disk'],
        caption_extension=new_config['caption_extension'],
        enable_bucket=new_config['enable_bucket'],
        gradient_checkpointing=new_config['gradient_checkpointing'],
        full_fp16=new_config['full_fp16'],
        no_token_padding=new_config['no_token_padding'],
        stop_text_encoder_training_pct=0,#Not Supported
        min_bucket_reso=new_config['min_bucket_reso'],
        max_bucket_reso=new_config['max_bucket_reso'],
        # use_8bit_adam,
        xformers=new_config['xformers'],
        save_model_as=new_config['save_model_as'],
        shuffle_caption=new_config['shuffle_caption'],
        save_state=new_config['save_state'],
        resume=new_config['resume'],
        prior_loss_weight=new_config['prior_loss_weight'],
        text_encoder_lr=new_config['text_encoder_lr'],
        unet_lr=new_config['unet_lr'],
        network_dim=new_config['network_dim'],
        lora_network_weights=new_config['lora_network_weights'],
        dim_from_weights=new_config['dim_from_weights'],
        color_aug=new_config['color_aug'],
        flip_aug=new_config['flip_aug'],
        clip_skip=new_config['clip_skip'],
        gradient_accumulation_steps=new_config['gradient_accumulation_steps'],
        mem_eff_attn=new_config['mem_eff_attn'],
        output_name=new_config['output_name'],
        model_list=new_config['model_list'],  # Keep this. Yes, it is unused here but required given the common list used
        max_token_length=new_config['max_token_length'],
        max_train_epochs=new_config['max_train_epochs'],
        max_train_steps=new_config['max_train_steps'],
        max_data_loader_n_workers=new_config['max_data_loader_n_workers'],
        network_alpha=new_config['network_alpha'],
        training_comment=new_config['training_comment'],
        keep_tokens=new_config['keep_tokens'],
        lr_scheduler_num_cycles=new_config['lr_scheduler_num_cycles'],
        lr_scheduler_power=new_config['lr_scheduler_power'],
        persistent_data_loader_workers=new_config['persistent_data_loader_workers'],
        bucket_no_upscale=new_config['bucket_no_upscale'],
        random_crop=new_config['random_crop'],
        bucket_reso_steps=new_config['bucket_reso_steps'],
        v_pred_like_loss=new_config['v_pred_like_loss'],
        caption_dropout_every_n_epochs=new_config['caption_dropout_every_n_epochs'],
        caption_dropout_rate=new_config['caption_dropout_rate'],
        optimizer=new_config['optimizer_type'],
        optimizer_args=new_config['optimizer_args'],
        lr_scheduler_args=new_config['lr_scheduler_args'],
        noise_offset_type=new_config['noise_offset_type'],
        noise_offset=new_config['noise_offset'],
        adaptive_noise_scale=new_config['adaptive_noise_scale'],
        multires_noise_iterations=new_config['multires_noise_iterations'],
        multires_noise_discount=new_config['multires_noise_discount'],
        LoRA_type=new_config['LoRA_type'],
        factor=new_config['factor'],
        use_cp=new_config['use_cp'],
        decompose_both=new_config['decompose_both'],
        train_on_input=new_config['train_on_input'],
        conv_dim=new_config['conv_dim'],
        conv_alpha=new_config['conv_alpha'],
        sample_every_n_steps=new_config['sample_every_n_steps'],
        sample_every_n_epochs=new_config['sample_every_n_epochs'],
        sample_sampler=new_config['sample_sampler'],
        sample_prompts=new_config['sample_prompts'],
        additional_parameters=new_config['additional_parameters'],
        vae_batch_size=new_config['vae_batch_size'],
        min_snr_gamma=new_config['min_snr_gamma'],
        down_lr_weight=new_config['down_lr_weight'],
        mid_lr_weight=new_config['mid_lr_weight'],
        up_lr_weight=new_config['up_lr_weight'],
        block_lr_zero_threshold=new_config['block_lr_zero_threshold'],
        block_dims=new_config['block_dims'],
        block_alphas=new_config['block_alphas'],
        conv_block_dims=new_config['conv_block_dims'],
        conv_block_alphas=new_config['conv_block_alphas'],
        weighted_captions=new_config['weighted_captions'],
        unit=new_config['unit'],
        save_every_n_steps=new_config['save_every_n_steps'],
        save_last_n_steps=new_config['save_last_n_steps'],
        save_last_n_steps_state=new_config['save_last_n_steps_state'],
        use_wandb=new_config['use_wandb'],
        wandb_api_key=new_config['wandb_api_key'],
        scale_v_pred_loss_like_noise_pred=new_config['scale_v_pred_loss_like_noise_pred'],
        scale_weight_norms=new_config['scale_weight_norms'],
        network_dropout=new_config['network_dropout'],
        rank_dropout=new_config['rank_dropout'],
        module_dropout=new_config['module_dropout'],
        sdxl_cache_text_encoder_outputs=new_config['sdxl_cache_text_encoder_outputs'],
        sdxl_no_half_vae=new_config['sdxl_no_half_vae'],
        full_bf16=new_config['full_bf16'],
        min_timestep=new_config['min_timestep'],
        max_timestep=new_config['max_timestep'],
        vae = new_config['vae']
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #input aguments for comand line functions
    parser.add_argument(
        '-p',
        '--prompt',
        type=str,
        help='The prompt of the focus for the finetuning'
    )

    parser.add_argument(
        '-Ip',
        '--img_prompt',
        type=str,
        default='',
        help='The file path to an image to be used in the clip image search'
    )

    #where the images are stored (has to be in the "lora\img" structure)
    parser.add_argument(
        '-i',
        '--img_Filepath',
        type=str,
        help='The file path too the folder of images'
    )
    
    
    parser.add_argument(
        '-n',
        '--number_of_subset',
        type=int,
        default=5,
        help='The number of subset images to take'
    )

    parser.add_argument(
        '-si',
        '--img_subset_output',
        type=str,
        help='The output dir for the subset of images from clip model'
    )
    
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        help='The output path for the models'
    )

    parser.add_argument(
        '-l',
        '--logging_dir',
        type=str,
        help='The output path for the logging file'
    )
    #a folder of json files that have configurations in them
    parser.add_argument(
        '-c',
        '--config_Filepath',
        type=str,
        default=None,
        help='The file path too the folder of config files'
    )
    

    #Name for the models
    parser.add_argument(
        '-m',
        '--output_name',
        type=str,
        default='New_model',
        help='The prompt of the focus for the finetuning'
    )

    parser.add_argument(
        '-pre',
        '--prefix',
        type=str,
        default='default',
        help='The Prefix to add to captioning'
    )

    parser.add_argument(
        '-epo',
        '--Epoch',
        type=int,
        default=1,
        help='The number of epochs the model will train for'
    )

    ##Learning Rate Inputs
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=1e-05,
        help='Starting Learning rate'
    )

    parser.add_argument(
        '-lr_stop',
        '--learning_rate_stop',
        type=float,
        default=1e-05,
        help='Stopping place for learning rate'
    )

    parser.add_argument(
        '-lr_step',
        '--learning_rate_step',
        type=float,
        default=1,
        help='Stopping place for learning rate'
    )

    ##Optimizer type
    parser.add_argument(
        '-opt',
        '--optimizer_type',
        type=str,
        default="AdamW8bit",
        help='Names of optimizers'
    )

    #Grounding Dino input parameters
    

    """
    List of inputs to still go through
    --enable_bucket 
    --min_bucket_reso=256
    --max_bucket_reso=2048
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
    DONE--train_data_dir="G:/AICapstone/test_files/jojo_stone_ocean_40_imgs/lora/img/sample"
    TODO--resolution="512,512" 
    DONE--output_dir="G:/AICapstone/test_files/jojo_stone_ocean_40_imgs/lora/img" 
    --network_alpha="1" 
    --save_model_as=safetensors 
    --network_module=networks.lora 
    TODO--text_encoder_lr=5e-05 
    TODO--unet_lr=0.0001 
    --network_dim=8 
    DONE--output_name="last" 
    --lr_scheduler_num_cycles="1" 
    --no_half_vae 
    DONE --learning_rate="0.0001" 
    TODO--lr_scheduler="cosine" 
    --train_batch_size="1" 
    --save_every_n_epochs="1" 
    --mixed_precision="fp16" 
    --save_precision="fp16" 
    --cache_latents 
    DONE--optimizer_type="AdamW8bit" 
    --max_data_loader_n_workers="0" 
    --bucket_reso_steps=64 
    --xformers 
    --bucket_no_upscale 
    --noise_offset=0.0
    """
    args = parser.parse_args()
    
    #set up lora file path in image folder if not there
    subset_Filepath = os.path.join(args.img_Filepath, f'lora/img/1_{args.output_name}')
    os.makedirs(subset_Filepath, exist_ok=True)

    #checks to see if any files are in the output folder
    if len(os.listdir(args.output_dir))>0:
        #asks the user for input
        user_input = input(f"Files are already located in the output folder given: {args.output_dir}. \nCan this program REMOVE all files located there?(Y/n)\n")    
        #if the input isn't what we are expecting then ask again'
        while not((user_input == 'Y') or (user_input == 'n')): 
            user_input = input(f"Incorrect input. Files are already located in the output folder given: {args.output_dir}. \nCan this program REMOVE all files located there?(Y/n)\n")    
        #lastly if the user said yes
        if user_input == 'Y':
            #this removes all the files from the output filepath
            for filename in os.listdir(args.output_dir):
                file_path = os.path.join(args.output_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            #raises an exception
            raise Exception("The output file needs to be empty.")


    #This sets up the images in the given output folder
    print("Running Clip Model")
    ###checks if an image was given as a prompt
    if args.img_prompt != '':
        img2img_subset(args.img_prompt, args.img_Filepath, subset_Filepath, args.number_of_subset)
    else:    
        prompt2Img_subset(args.prompt, args.img_Filepath, subset_Filepath, args.number_of_subset)
        

    ###
    ###GROUNDING DINO
    

    ###
    ###

    #Checks if the input folder has captions
    if any(file.endswith(".txt") for file in os.listdir(args.img_Filepath)):
        subset_Names = os.listdir(subset_Filepath)
        
        txt_names = [(file.split('.')[0] + f'.txt') for file in subset_Names]
        
        for file in txt_names:
            shutil.copy(os.path.join(args.img_Filepath, file), subset_Filepath)
    #runs the blip captioning
    else:
        #hard coded defaults from koya for blip captioning
        caption_images(subset_Filepath, ".txt", 1, 1, .9, 75, 5, True, args.prefix, "")

    args.img_Filepath = os.path.join(args.img_Filepath, f'lora/img')

    #Makes the dic list of all runs
    config_list = []
    if args.config_Filepath != None:
        config_list = get_configs(args.img_Filepath, args.output_dir, args.logging_dir, args.config_Filepath)
    else:
        config_list = create_configs(vars(args))

    if len(config_list) < 1: 
        raise Exception("There are no Configs to use")    

    lora_loop(config_list)

    
    
