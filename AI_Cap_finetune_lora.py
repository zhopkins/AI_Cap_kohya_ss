import argparse
from PIL import Image
import subprocess
import json
#from lora_gui import train_model
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
            print(new_config)
            print()

            # train_model(
            # headless,
            # print_only,
            # pretrained_model_name_or_path,
            # v2,
            # v_parameterization,
            # sdxl,
            # logging_dir,
            # train_data_dir,
            # reg_data_dir,
            # output_dir,
            # max_resolution,
            # learning_rate,
            # lr_scheduler,
            # lr_warmup,
            # train_batch_size,
            # epoch,
            # save_every_n_epochs,
            # mixed_precision,
            # save_precision,
            # seed,
            # num_cpu_threads_per_process,
            # cache_latents,
            # cache_latents_to_disk,
            # caption_extension,
            # enable_bucket,
            # gradient_checkpointing,
            # full_fp16,
            # no_token_padding,
            # stop_text_encoder_training_pct,
            # min_bucket_reso,
            # max_bucket_reso,
            # # use_8bit_adam,
            # xformers,
            # save_model_as,
            # shuffle_caption,
            # save_state,
            # resume,
            # prior_loss_weight,
            # text_encoder_lr,
            # unet_lr,
            # network_dim,
            # lora_network_weights,
            # dim_from_weights,
            # color_aug,
            # flip_aug,
            # clip_skip,
            # gradient_accumulation_steps,
            # mem_eff_attn,
            # output_name,
            # model_list,  # Keep this. Yes, it is unused here but required given the common list used
            # max_token_length,
            # max_train_epochs,
            # max_train_steps,
            # max_data_loader_n_workers,
            # network_alpha,
            # training_comment,
            # keep_tokens,
            # lr_scheduler_num_cycles,
            # lr_scheduler_power,
            # persistent_data_loader_workers,
            # bucket_no_upscale,
            # random_crop,
            # bucket_reso_steps,
            # v_pred_like_loss,
            # caption_dropout_every_n_epochs,
            # caption_dropout_rate,
            # optimizer,
            # optimizer_args,
            # lr_scheduler_args,
            # noise_offset_type,
            # noise_offset,
            # adaptive_noise_scale,
            # multires_noise_iterations,
            # multires_noise_discount,
            # LoRA_type,
            # factor,
            # use_cp,
            # decompose_both,
            # train_on_input,
            # conv_dim,
            # conv_alpha,
            # sample_every_n_steps,
            # sample_every_n_epochs,
            # sample_sampler,
            # sample_prompts,
            # additional_parameters,
            # vae_batch_size,
            # min_snr_gamma,
            # down_lr_weight,
            # mid_lr_weight,
            # up_lr_weight,
            # block_lr_zero_threshold,
            # block_dims,
            # block_alphas,
            # conv_block_dims,
            # conv_block_alphas,
            # weighted_captions,
            # unit,
            # save_every_n_steps,
            # save_last_n_steps,
            # save_last_n_steps_state,
            # use_wandb,
            # wandb_api_key,
            # scale_v_pred_loss_like_noise_pred,
            # scale_weight_norms,
            # network_dropout,
            # rank_dropout,
            # module_dropout,
            # sdxl_cache_text_encoder_outputs,
            # sdxl_no_half_vae,
            # full_bf16,
            # min_timestep,
            # max_timestep,
            # )
    


def clip_func(prompt, imgFilepath):
    print('Working on it')



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
    clip_func(args.prompt, args.imgFilepath)
    finetuner_lora(args.prompt, args.imgFilepath, args.configFilepath)

    