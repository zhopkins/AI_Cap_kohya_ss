import argparse
from PIL import Image
import subprocess
import json
from lora_gui import train_model
import os
import pathlib


def finetuner_lora(prompt, imgFilepath, configFilepath):
    #get all the json files
    json_list = [f for f in pathlib.Path(configFilepath).iterdir() if f.is_file()]
    print('Found {} config files: {}'.format(len(json_list), json_list))
    
    #run each lora training
    for filename in json_list:
        with open(filename,'r') as f:
            new_config = json.load(f)
            print()
            print(new_config)
            print()
            
            train_model(
            headless={'label':'False'},#0######
            print_only={'label':'False'},#Change if you want to print comands in stead of running#
            pretrained_model_name_or_path=new_config['pretrained_model_name_or_path'],
            v2=new_config['v2'],
            v_parameterization=new_config['v_parameterization'],
            sdxl='',
            logging_dir=new_config['logging_dir'],
            train_data_dir=new_config['train_data_dir'],
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
            optimizer=new_config['optimizer'],
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
        '-i',
        '--imgFilepath',
        type=str,
        help='The file path too the folder of images'
    )

    parser.add_argument(
        '-c',
        '--configFilepath',
        type=str,
        help='The file path too the folder of config files'
    )
    args = parser.parse_args()
    finetuner_lora(args.prompt, args.imgFilepath, args.configFilepath)

    