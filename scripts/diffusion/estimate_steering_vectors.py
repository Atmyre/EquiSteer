import os
import numpy as np
import tqdm
import pickle
import typing as tp

import torch

# local imports
from core.prompts import get_prompts_concrete, get_prompts_style, get_prompts_human_related, read_prompt_file
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.controller import DiffusionVectorControlMode
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, run_image_model

# parsing arguments
import argparse
from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode

def gather_stats_for_prompt_pairs(
        pipe,
        args: argparse.Namespace,
        checkpoint_steps: list[int],
        device: tp.Any,
        prompts_pos: list[str],
        prompts_neg: list[str],
):
    
    pos_stats_handler = CrossAttentionOutputStatsCollector(
        mode=args.control_mode,
        token_aggregation_mode=TokenAggregationMode.AVERAGE if args.patch_average else TokenAggregationMode.ALL,
        normalize=args.normalize_vectors,
    )
    neg_stats_handler = CrossAttentionOutputStatsCollector(
        mode=args.control_mode,
        token_aggregation_mode=TokenAggregationMode.AVERAGE if args.patch_average else TokenAggregationMode.ALL,
        normalize=args.normalize_vectors,
    )
    
    # Register hooks on the appropriate model component
    model_component = getattr(pipe, 'transformer', None) or pipe.unet
    diffusion_register_vector_controls_with_hooks(
        model_component,
        pos_stats_handler,
        neg_stats_handler,
        model_type=DiffusionModelType.from_model(args.model),
    )

    # gender
    if args.attribute == "gender_male":
        prompts_pos = []
        B = ['a boy', 'two men', 'two male people', 'a man', 'an old man', 'boys', 'men', 'group of male people', 'a male human']
        C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
            'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']
        for b in B:
            for c in C:
                prompts_pos.append(b+' '+c)

        prompts_neg = []
        B = ['a girl', 'two women', 'two female people', 'a woman', 'an old woman', 'girls', 'women', 'group of female people', 'a female human']
        C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
            'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']
        for b in B:
            for c in C:
                prompts_neg.append(b+' '+c)

    elif args.attribute == "gender_female":
        prompts_neg = []
        B = ['a boy', 'two men', 'two male people', 'a man', 'an old man', 'boys', 'men', 'group of male people', 'a male human']
        C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
            'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']
        for b in B:
            for c in C:
                prompts_neg.append(b+' '+c)

        prompts_pos = []
        B = ['a girl', 'two women', 'two female people', 'a woman', 'an old woman', 'girls', 'women', 'group of female people', 'a female human']
        C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
            'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']
        for b in B:
            for c in C:
                prompts_pos.append(b+' '+c)
    
    elif args.attribute == "race":
        # race 
        B = ['a girl', 'a boy', 'two men', 'two women', 'two people', 'a man', 'a woman', 'an old man', 'an old woman', 'boys', 'girls', 'men', 'women', 'group of people', 'a human']
        C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
            'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']
        prompts_pos = []
        prompts_neg = []
        for b in B:
            for c in C:
                prompts_pos.append(b+f' of {args.concept_pos} race '+c)
                prompts_neg.append(b+' '+c)
    elif args.attribute == "eyeglasses":
        # glasses
        B = ['a girl', 'a boy', 'two men', 'two women', 'two people', 'a man', 'a woman', 'an old man', 'an old woman', 'boys', 'girls', 'men', 'women', 'group of people', 'a human']
        C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
            'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']
        prompts_pos = []
        prompts_neg = []
        for b in B:
            for c in C:
                prompts_pos.append(b+f' wearing {args.concept_pos} '+c)
                prompts_neg.append(b+f' '+c)
    elif args.attribute == "custom":
        # Generic "subject + specifier + context" prompt builder.
        # --concept_pos provides the specifier (e.g. "wearing a hijab",
        # "in a wheelchair", "with visible tattoos"); the surrounding subject
        # and context lists are reused.
        B = ['a girl', 'a boy', 'two men', 'two women', 'two people', 'a man', 'a woman', 'an old man', 'an old woman', 'boys', 'girls', 'men', 'women', 'group of people', 'a human']
        C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
            'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']
        prompts_pos = []
        prompts_neg = []
        for b in B:
            for c in C:
                prompts_pos.append(f'{b} {args.concept_pos} {c}')
                prompts_neg.append(f'{b} {c}')
    elif args.attribute == "custom_identity":
        # Identity-prefix prompts: "a {identity} {subject}" instead of
        # "a {subject} wearing a {marker}". Compares whether the gate fires on
        # an attribute named purely by an identity word (e.g. "a muslim man")
        # vs by a visible marker (e.g. "a man wearing a hijab").
        B = ['a girl', 'a boy', 'two men', 'two women', 'two people', 'a man', 'a woman', 'an old man', 'an old woman', 'boys', 'girls', 'men', 'women', 'group of people', 'a human']
        C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
            'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']
        identity = args.concept_pos.strip()

        def _inject_identity(subj: str, ident: str) -> str:
            parts = subj.split()
            if parts[0] in ('a', 'an', 'the'):
                return f"{parts[0]} {ident} {' '.join(parts[1:])}"
            if parts[0] == 'group' and len(parts) > 2 and parts[1] == 'of':
                return f"group of {ident} {' '.join(parts[2:])}"
            if parts[0] in ('two', 'three'):
                return f"{parts[0]} {ident} {' '.join(parts[1:])}"
            return f"{ident} {subj}"

        prompts_pos = []
        prompts_neg = []
        for b in B:
            for c in C:
                prompts_pos.append(f'{_inject_identity(b, identity)} {c}')
                prompts_neg.append(f'{b} {c}')
    elif args.attribute == "age":
        # age: drop 'old man'/'old woman' subjects since they bake in age
        B = ['a girl', 'a boy', 'two men', 'two women', 'two people', 'a man', 'a woman', 'boys', 'girls', 'men', 'women', 'group of people', 'a human', 'a person', 'people']
        C = ['', 'gloomy image', 'zoomed in', 'talking', 'on the street', 'in a strange pose', 'realism', \
            'colorful background', 'on a beach', 'playing guitar', 'enjoying nature', 'smiling', 'in a futuristic spaceship', 'with kittens']
        # concept_pos in {young, middle-aged, elderly} -- use natural age phrasing
        age_phrase = args.concept_pos.replace('_', '-')
        prompts_pos = []
        prompts_neg = []
        for b in B:
            for c in C:
                prompts_pos.append(b+f' who is {age_phrase} '+c)
                prompts_neg.append(b+f' '+c)
    

    print("Gathering statistics for concept prompts...", flush=True)
    import gc as _gc
    import time as _time
    import traceback as _tb

    def _gpu_mem():
        if torch.cuda.is_available():
            return f"{torch.cuda.memory_allocated()/1e9:.2f}G alloc / {torch.cuda.memory_reserved()/1e9:.2f}G reserved"
        return "n/a"

    for idx, (pos_prompt, neg_prompt) in enumerate(zip(prompts_pos, prompts_neg)):
        t0 = _time.time()
        print(f"[iter {idx}] pos='{pos_prompt}'  neg='{neg_prompt}'  gpu={_gpu_mem()}", flush=True)
        if idx in checkpoint_steps:
            print(f"[iter {idx}] checkpoint", flush=True)
            try:
                write_checkpoint(
                    output_dir=args.output_dir,
                    step=idx,
                    pos_handler=pos_stats_handler,
                    neg_handler=neg_stats_handler,
                )
            except Exception:
                print(f"[iter {idx}] checkpoint failed:\n{_tb.format_exc()}", flush=True)

        try:
            pos_stats_handler.active = True
            neg_stats_handler.active = False
            image = run_image_model(
                model_type=args.model,
                pipe=pipe,
                prompt=pos_prompt,
                seed=0,
                device=device,
            )[0]
            pos_stats_handler.reset()

            pos_stats_handler.active = False
            neg_stats_handler.active = True
            image = run_image_model(
                model_type=args.model,
                pipe=pipe,
                prompt=neg_prompt,
                seed=0,
                device=device,
            )[0]
            neg_stats_handler.reset()
        except Exception:
            print(f"[iter {idx}] generation failed:\n{_tb.format_exc()}", flush=True)
            # Try to free anything we can
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # Periodic memory hygiene
        if idx % 10 == 0:
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[iter {idx}] done in {_time.time()-t0:.2f}s  gpu={_gpu_mem()}", flush=True)

    write_checkpoint(
        output_dir=args.output_dir,
        step=idx+1,
        pos_handler=pos_stats_handler,
        neg_handler=neg_stats_handler,
    )


def write_checkpoint(
        output_dir: str,
        step: int,
        pos_handler: CrossAttentionOutputStatsCollector,
        neg_handler: CrossAttentionOutputStatsCollector,
):
    pos_handler.save_stats(means_path=f'{output_dir}/pos_means_{step}.pickle')
    neg_handler.save_stats(means_path=f'{output_dir}/neg_means_{step}.pickle')

    casteer_vectors = calculate_casteer(pos_means=pos_handler.means, neg_means=neg_handler.means)
    with open(f'{output_dir}/casteer_{step}.pickle', 'wb') as fout:
        pickle.dump(casteer_vectors, fout)


def calculate_casteer(pos_means: dict, neg_means: dict) -> dict:
    result = {}
    EPS = 1e-8
    for denoising_step in pos_means.keys():
        result[denoising_step] = {}
        for place_in_unet in pos_means[denoising_step].keys():
            result[denoising_step][place_in_unet] = []
            for block_idx in range(len(pos_means[denoising_step][place_in_unet])):
                print(f'Processing step={denoising_step}, block={place_in_unet}, layer={block_idx}', flush=True)
                pos = pos_means[denoising_step][place_in_unet][block_idx]
                neg = neg_means[denoising_step][place_in_unet][block_idx]
                # promote to float32 on CPU before subtraction so a stray NaN/Inf
                # in the GPU-side fp16 accumulators doesn't poison subsequent ops
                pos = pos.detach().to(torch.float32).cpu()
                neg = neg.detach().to(torch.float32).cpu()
                steering_vector = pos - neg
                norm = torch.linalg.norm(steering_vector, dim=1, keepdim=True)
                steering_vector = steering_vector / (norm + EPS)
                steering_vector = torch.nan_to_num(steering_vector, nan=0.0, posinf=0.0, neginf=0.0)
                result[denoising_step][place_in_unet].append(steering_vector.numpy())
    return result


def run(args: argparse.Namespace):
    pipe = init_pipeline_for_image_model(args.model)
    pipe.set_progress_bar_config(disable=True)

    device = get_device()


    # if args.mode == 'concrete':
    #     prompts_pos, prompts_neg = get_prompts_concrete(concept_pos=args.concept_pos, 
    #                                                     concept_neg=args.concept_neg)
    # elif args.mode == 'human-related':
    #     prompts_pos, prompts_neg = get_prompts_human_related(concept_pos=args.concept_pos, 
    #                                                         concept_neg=args.concept_neg)
    # elif args.mode == 'style':
    #     prompts_pos, prompts_neg = get_prompts_style(concept_pos=args.concept_pos, 
    #                                                 concept_neg=args.concept_neg)
    # elif args.mode == 'file':
    #     prompts_pos = read_prompt_file(args.prompts_pos_file)
    #     prompts_neg = read_prompt_file(args.prompts_neg_file)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_steps = set(map(int, args.checkpoint_steps.split(',')))

    # assert prompts_pos is not None and prompts_neg is not None, "both positive and negative prompts must be provided"

    gather_stats_for_prompt_pairs(
        pipe=pipe,
        args=args,
        checkpoint_steps=checkpoint_steps,
        device=device,
        prompts_pos=None,
        prompts_neg=None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True)
    parser.add_argument('--mode', type=str, choices=['concrete', 'human-related', 'style', 'file'], default="style")
    parser.add_argument('--control_mode', type=DiffusionVectorControlMode, choices=[str(x) for x in DiffusionVectorControlMode], default='attn_output', help='Vector control mode')
    parser.add_argument('--prompts_pos_file', type=str, default=None,
                        help="If --mode is set to 'file', path to the text file containing positive prompts")
    parser.add_argument('--prompts_neg_file', type=str, default=None,
                        help="If --mode is set to 'file', path to the text file containing negative prompts")
    parser.add_argument('--concept_pos', type=str, default="anime")
    parser.add_argument('--concept_neg', type=str, default=None)
    parser.add_argument('--attribute', type=str, choices=['gender_male', 'gender_female', 'race', 'eyeglasses', 'age', 'custom', 'custom_identity'], default="gender_male")
    parser.add_argument('--patch_average', action='store_true', help='Average across patches for each prompt before updating statistics')
    parser.add_argument('--normalize_vectors', action='store_true', help='Whether to normalize vectors before computing the statistics')
    parser.add_argument('--output_dir', type=str, default=None, required=True, help='path to saving steering vectors')
    parser.add_argument('--checkpoint_steps', type=str, default='100,500,1000,5000,10000', help='A comma separated list of integers representing steps at which to write checkpoints')
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()