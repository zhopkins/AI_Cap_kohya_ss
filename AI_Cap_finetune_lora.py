import argparse
import os
import pathlib
import itertools 

import json
import math
import os
import argparse
from datetime import datetime
from library.common_gui import (
    get_file_path,
    get_any_file_path,
    get_saveasfile_path,
    color_aug_changed,
    run_cmd_advanced_training,
    run_cmd_training,
    update_my_data,
    check_if_model_exist,
    output_message,
    verify_image_folder_pattern,
    SaveConfigFile,
    save_to_file,
    check_duplicate_filenames,
)

from library.class_command_executor import CommandExecutor

from library.utilities import utilities_tab
from library.class_sample_images import SampleImages, run_cmd_sample
from library.class_lora_tab import LoRATools

from library.dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from library.dataset_balancing_gui import gradio_dataset_balancing_tab

from library.custom_logging import setup_logging
from localization_ext import add_javascript

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


def create_configs(args_dic):
    """
    This function taks in a dictionary of arguments and create a list of configuerations for loar training.
    """
    config_list = []

    #makes that list of variables desired
    #this is for learning Rate
    lr_list = floating_point_range(args_dic["learning_rate"], args_dic["learning_rate_stop"], args_dic["learning_rate_step"])
    #This is for Optimizer
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
            input_copy["output_name"] = input_copy["output_name"] + "_" + str(name_count)
            #setup the default with keys
            for key in input_copy.keys():
                new_config[key] = input_copy[key]
            #adds the new config
            config_list.append(new_config)
            name_count += 1



    return config_list


    
    
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




def lora_loop(prompt, config_list):
    ###THIS IS WHERE TO ADD THE CLIP AND GROUNDING DINO PARTS
    
    ###
    #run each lora training
    for new_config in config_list:
        print('\nStarted Lora Traning on ', new_config['output_name'], '...\n')
            
        popout_train_model(
        headless={'label':'False'},#0######
        print_only={'label':'True'},#Make True Change if you want to print comands in stead of running#
        pretrained_model_name_or_path=new_config['pretrained_model_name_or_path'],
        v2=new_config['v2'],
        v_parameterization=new_config['v_parameterization'],
        sdxl='',
        logging_dir=new_config['logging_dir'],
        train_data_dir=new_config['img_Filepath'], ##Need to test if file path exists
        reg_data_dir=new_config['reg_data_dir'],
        output_dir=new_config['output_dir'],
        max_resolution=new_config['max_resolution'],
        learning_rate=new_config['learning_rate'],
        lr_scheduler=new_config['lr_scheduler'],
        lr_warmup=new_config['lr_warmup'],
        train_batch_size=new_config['train_batch_size'],
        epoch=new_config['epoch'],
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
        )


def popout_train_model(
    headless,
    print_only,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training_pct,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    v_pred_like_loss,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    lr_scheduler_args,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    LoRA_type,
    factor,
    use_cp,
    decompose_both,
    train_on_input,
    conv_dim,
    conv_alpha,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    down_lr_weight,
    mid_lr_weight,
    up_lr_weight,
    block_lr_zero_threshold,
    block_dims,
    block_alphas,
    conv_block_dims,
    conv_block_alphas,
    weighted_captions,
    unit,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    scale_weight_norms,
    network_dropout,
    rank_dropout,
    module_dropout,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    full_bf16,
    min_timestep,
    max_timestep,
):
    # Get list of function parameters and values
    parameters = list(locals().items())
    global command_running

    print_only_bool = True if print_only.get('label') == 'True' else False
    log.info(f'Start training LoRA {LoRA_type} ...')
    headless_bool = True if headless.get('label') == 'True' else False

    if pretrained_model_name_or_path == '':
        output_message(
            msg='Source model information is missing', headless=headless_bool
        )
        return

    if train_data_dir == '':
        output_message(
            msg='Image folder path is missing', headless=headless_bool
        )
        return

    # Check if there are files with the same filename but different image extension... warn the user if it is the case.
    check_duplicate_filenames(train_data_dir)

    if not os.path.exists(train_data_dir):
        output_message(
            msg='Image folder does not exist', headless=headless_bool
        )
        return

    if not verify_image_folder_pattern(train_data_dir):
        return

    if reg_data_dir != '':
        if not os.path.exists(reg_data_dir):
            output_message(
                msg='Regularisation folder does not exist',
                headless=headless_bool,
            )
            return

        if not verify_image_folder_pattern(reg_data_dir):
            return

    if output_dir == '':
        output_message(
            msg='Output folder path is missing', headless=headless_bool
        )
        return

    if int(bucket_reso_steps) < 1:
        output_message(
            msg='Bucket resolution steps need to be greater than 0',
            headless=headless_bool,
        )
        return

    if noise_offset == '':
        noise_offset = 0

    if float(noise_offset) > 1 or float(noise_offset) < 0:
        output_message(
            msg='Noise offset need to be a value between 0 and 1',
            headless=headless_bool,
        )
        return

    # if float(noise_offset) > 0 and (
    #     multires_noise_iterations > 0 or multires_noise_discount > 0
    # ):
    #     output_message(
    #         msg="noise offset and multires_noise can't be set at the same time. Only use one or the other.",
    #         title='Error',
    #         headless=headless_bool,
    #     )
    #     return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if stop_text_encoder_training_pct > 0:
        output_message(
            msg='Output "stop text encoder training" is not yet supported. Ignoring',
            headless=headless_bool,
        )
        stop_text_encoder_training_pct = 0

    if check_if_model_exist(
        output_name, output_dir, save_model_as, headless=headless_bool
    ):
        return

    # if optimizer == 'Adafactor' and lr_warmup != '0':
    #     output_message(
    #         msg="Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.",
    #         title='Warning',
    #         headless=headless_bool,
    #     )
    #     lr_warmup = '0'

    # If string is empty set string to 0.
    if text_encoder_lr == '':
        text_encoder_lr = 0
    if unet_lr == '':
        unet_lr = 0

    # Get a list of all subfolders in train_data_dir
    subfolders = [
        f
        for f in os.listdir(train_data_dir)
        if os.path.isdir(os.path.join(train_data_dir, f))
    ]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        try:
            # Extract the number of repeats from the folder name
            repeats = int(folder.split('_')[0])

            # Count the number of images in the folder
            num_images = len(
                [
                    f
                    for f, lower_f in (
                        (file, file.lower())
                        for file in os.listdir(
                            os.path.join(train_data_dir, folder)
                        )
                    )
                    if lower_f.endswith(('.jpg', '.jpeg', '.png', '.webp'))
                ]
            )

            log.info(f'Folder {folder}: {num_images} images found')

            # Calculate the total number of steps for this folder
            steps = repeats * num_images

            # log.info the result
            log.info(f'Folder {folder}: {steps} steps')

            total_steps += steps

        except ValueError:
            # Handle the case where the folder name does not contain an underscore
            log.info(
                f"Error: '{folder}' does not contain an underscore, skipping..."
            )

    if reg_data_dir == '':
        reg_factor = 1
    else:
        log.warning(
            'Regularisation images are used... Will double the number of steps required...'
        )
        reg_factor = 2

    log.info(f'Total steps: {total_steps}')
    log.info(f'Train batch size: {train_batch_size}')
    log.info(f'Gradient accumulation steps: {gradient_accumulation_steps}')
    log.info(f'Epoch: {epoch}')
    log.info(f'Regulatization factor: {reg_factor}')

    if max_train_steps == '' or max_train_steps == '0':
        # calculate max_train_steps
        max_train_steps = int(
            math.ceil(
                float(total_steps)
                / int(train_batch_size)
                / int(gradient_accumulation_steps)
                * int(epoch)
                * int(reg_factor)
            )
        )
        log.info(
            f'max_train_steps ({total_steps} / {train_batch_size} / {gradient_accumulation_steps} * {epoch} * {reg_factor}) = {max_train_steps}'
        )

    # calculate stop encoder training
    if stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    log.info(f'stop_text_encoder_training = {stop_text_encoder_training}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    log.info(f'lr_warmup_steps = {lr_warmup_steps}')
    ##Added "start cmd /k" to give it's own 
    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process}'
    if sdxl:
        run_cmd += f' "./sdxl_train_network.py"'
    else:
        run_cmd += f' "./train_network.py"'

    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if enable_bucket:
        run_cmd += f' --enable_bucket --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso}'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    if weighted_captions:
        run_cmd += ' --weighted_captions'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution="{max_resolution}"'
    run_cmd += f' --output_dir="{output_dir}"'
    if not logging_dir == '':
        run_cmd += f' --logging_dir="{logging_dir}"'
    run_cmd += f' --network_alpha="{network_alpha}"'
    if not training_comment == '':
        run_cmd += f' --training_comment="{training_comment}"'
    if not stop_text_encoder_training == 0:
        run_cmd += (
            f' --stop_text_encoder_training={stop_text_encoder_training}'
        )
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if not float(prior_loss_weight) == 1.0:
        run_cmd += f' --prior_loss_weight={prior_loss_weight}'

    if LoRA_type == 'LoCon' or LoRA_type == 'LyCORIS/LoCon':
        try:
            import lycoris
        except ModuleNotFoundError:
            log.info(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "algo=lora"'

    if LoRA_type == 'LyCORIS/LoHa':
        try:
            import lycoris
        except ModuleNotFoundError:
            log.info(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "use_cp={use_cp}" "algo=loha"'
        # This is a hack to fix a train_network LoHA logic issue
        if not network_dropout > 0.0:
            run_cmd += f' --network_dropout="{network_dropout}"'

    if LoRA_type == 'LyCORIS/iA3':
        try:
            import lycoris
        except ModuleNotFoundError:
            log.info(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "train_on_input={train_on_input}" "algo=ia3"'
        # This is a hack to fix a train_network LoHA logic issue
        if not network_dropout > 0.0:
            run_cmd += f' --network_dropout="{network_dropout}"'

    if LoRA_type == 'LyCORIS/DyLoRA':
        try:
            import lycoris
        except ModuleNotFoundError:
            log.info(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "use_cp={use_cp}" "block_size={unit}" "algo=dylora"'
        # This is a hack to fix a train_network LoHA logic issue
        if not network_dropout > 0.0:
            run_cmd += f' --network_dropout="{network_dropout}"'

    if LoRA_type == 'LyCORIS/LoKr':
        try:
            import lycoris
        except ModuleNotFoundError:
            log.info(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "factor={factor}" "use_cp={use_cp}" "algo=lokr"'
        # This is a hack to fix a train_network LoHA logic issue
        if not network_dropout > 0.0:
            run_cmd += f' --network_dropout="{network_dropout}"'

    if LoRA_type in ['Kohya LoCon', 'Standard']:
        kohya_lora_var_list = [
            'down_lr_weight',
            'mid_lr_weight',
            'up_lr_weight',
            'block_lr_zero_threshold',
            'block_dims',
            'block_alphas',
            'conv_block_dims',
            'conv_block_alphas',
            'rank_dropout',
            'module_dropout',
        ]

        run_cmd += f' --network_module=networks.lora'
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ''
        if LoRA_type == 'Kohya LoCon':
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'

    if LoRA_type in [
        'LoRA-FA',
    ]:
        kohya_lora_var_list = [
            'down_lr_weight',
            'mid_lr_weight',
            'up_lr_weight',
            'block_lr_zero_threshold',
            'block_dims',
            'block_alphas',
            'conv_block_dims',
            'conv_block_alphas',
            'rank_dropout',
            'module_dropout',
        ]

        run_cmd += f' --network_module=networks.lora_fa'
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ''
        if LoRA_type == 'Kohya LoCon':
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'

    if LoRA_type in ['Kohya DyLoRA']:
        kohya_lora_var_list = [
            'conv_dim',
            'conv_alpha',
            'down_lr_weight',
            'mid_lr_weight',
            'up_lr_weight',
            'block_lr_zero_threshold',
            'block_dims',
            'block_alphas',
            'conv_block_dims',
            'conv_block_alphas',
            'rank_dropout',
            'module_dropout',
            'unit',
        ]

        run_cmd += f' --network_module=networks.dylora'
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ''

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'

    if not (float(text_encoder_lr) == 0) or not (float(unet_lr) == 0):
        if not (float(text_encoder_lr) == 0) and not (float(unet_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --unet_lr={unet_lr}'
        elif not (float(text_encoder_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --network_train_text_encoder_only'
        else:
            run_cmd += f' --unet_lr={unet_lr}'
            run_cmd += f' --network_train_unet_only'
    else:
        if float(learning_rate) == 0:
            output_message(
                msg='Please input learning rate values.',
                headless=headless_bool,
            )
            return

    run_cmd += f' --network_dim={network_dim}'

    # if LoRA_type not in ['LyCORIS/LoCon']:
    if not lora_network_weights == '':
        run_cmd += f' --network_weights="{lora_network_weights}"'
        if dim_from_weights:
            run_cmd += f' --dim_from_weights'

    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'
    if not lr_scheduler_num_cycles == '':
        run_cmd += f' --lr_scheduler_num_cycles="{lr_scheduler_num_cycles}"'
    else:
        run_cmd += f' --lr_scheduler_num_cycles="{epoch}"'
    if not lr_scheduler_power == '':
        run_cmd += f' --lr_scheduler_power="{lr_scheduler_power}"'

    if scale_weight_norms > 0.0:
        run_cmd += f' --scale_weight_norms="{scale_weight_norms}"'

    if network_dropout > 0.0:
        run_cmd += f' --network_dropout="{network_dropout}"'

    if sdxl_cache_text_encoder_outputs:
        run_cmd += f' --cache_text_encoder_outputs'

    if sdxl_no_half_vae:
        run_cmd += f' --no_half_vae'

    if full_bf16:
        run_cmd += f' --full_bf16'

    run_cmd += run_cmd_training(
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        save_every_n_epochs=save_every_n_epochs,
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        seed=seed,
        caption_extension=caption_extension,
        cache_latents=cache_latents,
        cache_latents_to_disk=cache_latents_to_disk,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        lr_scheduler_args=lr_scheduler_args,
    )

    run_cmd += run_cmd_advanced_training(
        max_train_epochs=max_train_epochs,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_token_length=max_token_length,
        resume=resume,
        save_state=save_state,
        mem_eff_attn=mem_eff_attn,
        clip_skip=clip_skip,
        flip_aug=flip_aug,
        color_aug=color_aug,
        shuffle_caption=shuffle_caption,
        gradient_checkpointing=gradient_checkpointing,
        full_fp16=full_fp16,
        xformers=xformers,
        # use_8bit_adam=use_8bit_adam,
        keep_tokens=keep_tokens,
        persistent_data_loader_workers=persistent_data_loader_workers,
        bucket_no_upscale=bucket_no_upscale,
        random_crop=random_crop,
        bucket_reso_steps=bucket_reso_steps,
        v_pred_like_loss=v_pred_like_loss,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        noise_offset_type=noise_offset_type,
        noise_offset=noise_offset,
        adaptive_noise_scale=adaptive_noise_scale,
        multires_noise_iterations=multires_noise_iterations,
        multires_noise_discount=multires_noise_discount,
        additional_parameters=additional_parameters,
        vae_batch_size=vae_batch_size,
        min_snr_gamma=min_snr_gamma,
        save_every_n_steps=save_every_n_steps,
        save_last_n_steps=save_last_n_steps,
        save_last_n_steps_state=save_last_n_steps_state,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        scale_v_pred_loss_like_noise_pred=scale_v_pred_loss_like_noise_pred,
        min_timestep=min_timestep,
        max_timestep=max_timestep,
    )

    run_cmd += run_cmd_sample(
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        output_dir,
    )

    if print_only_bool:
        log.warning(
            'Here is the trainer command as a reference. It will not be executed:\n'
        )
        print(run_cmd)

        save_to_file(run_cmd)
    else:
        # Saving config file for model
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d-%H%M%S')
        file_path = os.path.join(
            output_dir, f'{output_name}_{formatted_datetime}.json'
        )

        log.info(f'Saving training config to {file_path}...')

        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=['file_path', 'save_as', 'headless', 'print_only'],
        )

        log.info(run_cmd)
        # Run the command
        executor.execute_command(run_cmd=run_cmd)
        ##Added logic to check if the process ran 
        status = executor.process.wait()
        _, errs = executor.process.communicate()
        print("The Training exit code:", errs)
            

        # # check if output_dir/last is a folder... therefore it is a diffuser model
        # last_dir = pathlib.Path(f'{output_dir}/{output_name}')

        # if not last_dir.is_dir():
        #     # Copy inference model for v2 if required
        #     save_inference_file(
        #         output_dir, v2, v_parameterization, output_name
        #     )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #input aguments for comand line functions
    parser.add_argument(
        '-p',
        '--prompt',
        type=str,
        help='The prompt of the focus for the finetuning'
    )

    #where the images are stored (has to be in the "lora\img" structure)
    parser.add_argument(
        '-i',
        '--img_Filepath',
        type=str,
        help='The file path too the folder of images'
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
        default=0,
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

    """
    List of inputs to still go through
    --enable_bucket 
    --min_bucket_reso=256
    --max_bucket_reso=2048
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
    DONE--train_data_dir="G:/AICapstone/test_files/jojo_stone_ocean_40_imgs/lora/img/sample"
    --resolution="512,512" 
    DONE--output_dir="G:/AICapstone/test_files/jojo_stone_ocean_40_imgs/lora/img" 
    --network_alpha="1" 
    --save_model_as=safetensors 
    --network_module=networks.lora 
    --text_encoder_lr=5e-05 
    --unet_lr=0.0001 
    --network_dim=8 
    --output_name="last" 
    --lr_scheduler_num_cycles="1" 
    --no_half_vae 
    DONE --learning_rate="0.0001" 
    --lr_scheduler="cosine" 
    --train_batch_size="1" 
    --save_every_n_epochs="1" 
    --mixed_precision="fp16" 
    --save_precision="fp16" 
    --cache_latents 
    --optimizer_type="AdamW8bit" 
    --max_data_loader_n_workers="0" 
    --bucket_reso_steps=64 
    --xformers 
    --bucket_no_upscale 
    --noise_offset=0.0
    """
    args = parser.parse_args()
    config_list = []
    if args.config_Filepath != None:
        config_list = get_configs(args.img_Filepath, args.output_dir, args.logging_dir, args.config_Filepath)
    else:
        config_list = create_configs(vars(args))

    if len(config_list) < 1: 
        raise Exception("There are no Configs to use")
    
    lora_loop(args.prompt, config_list)

    
    