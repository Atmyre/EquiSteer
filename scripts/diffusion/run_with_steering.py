import argparse
import os
import math
import typing as tp

from diffusers import DiffusionPipeline
from tqdm import tqdm

from core.controller import CrossAttentionOutputSteering, DiffusionVectorControlMode, ModelToSteer, VectorControl
from core.controller_multi import CrossAttentionOutputSteeringMulti
from core.dataset import CocoDataset, TemplateDataset, dumb_tokenizer_fn
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.pickle import unpickle
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, run_image_model

SAVE_OPTIONS = {
    'PNG': {},
    'JPEG': {
        'subsampling': '4:4:4',
        'quality': 95,
    },
}

EXTENSIONS = {
    'PNG': 'png',
    'JPEG': 'jpg',
}

def hook_model(pipeline: DiffusionPipeline, device: tp.Any, args: argparse.Namespace) -> VectorControl:
    if args.command is None:
        return None
    

    mu_neutral, sigma_neutral = None, None

    
    if args.command == 'erase':
        source_concept = unpickle(args.concept_path)
        target_concept = mu_neutral
    elif args.source_concept_path is not None:
        source_concepts = [unpickle(args.source_concept_path)]
        target_concepts = [unpickle(args.target_concept_path)]
    else:
        source_concepts = None
        target_concepts = None

    if args.multi_attribute_debias:
        if args.multi_attributes:
            attrs = tuple(a.strip().lower() for a in args.multi_attributes.split(",") if a.strip())
        else:
            attrs = ("race", "gender")
        vector_control = CrossAttentionOutputSteeringMulti(
            model_to_steer=ModelToSteer.UNET,
            mode=args.control_mode,
            strength=args.steering_strength,
            device=device,
            model_name=args.model_name,
            attributes=attrs,
            use_first_diffusion_step=not args.use_all_diffusion_steps,
            renormalize_after_steering=args.renormalize_after_steering,
            intermediate_clipping=args.intermediate_clipping,
            do_debias=args.do_debias,
            do_erase=args.do_erase,
            do_threshold=args.do_threshold,
        )
    else:
        vector_control = CrossAttentionOutputSteering(
            model_to_steer=ModelToSteer.UNET,
            mode=args.control_mode,
            steer_type=args.steering_method,
            target_concepts=target_concepts,
            source_concepts=source_concepts,
            steer_only_up=False,
            steer_back=True,
            strength=args.steering_strength,
            device=device,
            intermediate_clipping=args.intermediate_clipping,
            renormalize_after_steering=args.renormalize_after_steering,
            use_first_diffusion_step=not args.use_all_diffusion_steps,
            save_vectors = args.save_vectors,
            save_vectors_path = args.save_vectors_path,
            attribute=args.attribute,
            do_debias=args.do_debias,
            do_erase=args.do_erase,
            do_threshold=args.do_threshold,
            gate_threshold_multiplier=args.gate_threshold_multiplier,
            model_name=args.model_name,
        )

    # Register hooks on the appropriate model component
    model_component = getattr(pipeline, 'transformer', None) or pipeline.unet
    diffusion_register_vector_controls_with_hooks(
        model_component,
        vector_control,
        model_type=DiffusionModelType.from_model(args.model_name),
    )
    return vector_control


def main(args: argparse.Namespace):
    if args.steering_method is not None and args.steering_strength is None:
        raise ValueError(f'--steering_strength (float) must be specified for --steering_method={args.steering_method}')

    if args.command is None and args.steering_method is not None:
        raise ValueError(f'--steering_method is provided but no steering action (erase or flip) specified')
    
    if args.steering_method is None and args.command is not None:
        raise ValueError(f'Cannot {args.command} concept with no --steering_method specified')
    
    if (args.steering_method in ('leace', 'mean_matching') or args.command == 'erase') and args.covariances_dir is None:
        raise ValueError('')

    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)
    device = get_device()

    if args.command == 'translate':
        vector_control = hook_model(pipeline, device, args)
    else:
        vector_control = None
        print('ololosh')

    
    if args.generate_concept != 'coco':
        # we will generate a single prompt for the provided concept and create multiple image for that prompt
        # dataset = [f'a photo of a {args.generate_concept}']
        if args.prompt is not None:
            # User-provided full prompt. Substitute {concept} (and optional {concept2}) if present.
            concept_pretty = ' '.join(args.generate_concept.split('_'))
            sub = {'concept': concept_pretty}
            if args.generate_concept2 is not None:
                sub['concept2'] = ' '.join(args.generate_concept2.split('_'))
            dataset = [args.prompt.format(**sub)]
        else:
            dataset = [f'a photo of a {' '.join(args.generate_concept.split('_'))}']
        num_images_per_prompt = args.num_images_per_prompt
    else:
        import pandas as pd
        data_csv = pd.read_csv('/data/home/acw685/CA_diffusion_debiasing-main/coco-30k.csv')
        dataset = data_csv['prompt']
        seeds = data_csv['evaluation_seed']
        num_images_per_prompt = 1
    

    
    skipped = generated = 0
    print(f'Generating images for concept {dataset[0]} and method {args.steering_method} with strength {args.steering_strength}')
    
    if args.generate_concept != 'coco':
        for prompt in tqdm(dataset):
            num_batches = math.ceil(num_images_per_prompt / args.batch_size)
            for batch_id in range(0, num_batches):
                seed = args.seed + batch_id
                num_images = min(args.batch_size, num_images_per_prompt - batch_id * args.batch_size)

                output_paths = [f'{args.output_dir}/{prompt}/{seed}-{idx}.{EXTENSIONS[args.file_format]}' for idx in range(num_images)]
                if all(os.path.exists(path) for path in output_paths):
                    skipped += num_images
                    continue
                generated += num_images
                images = run_image_model(
                    model_type=args.model_name,
                    pipe=pipeline,
                    prompt=prompt,
                    seed=seed,
                    device=device,
                    num_images=num_images,
                )
                if vector_control is not None:
                    vector_control.reset()
                os.makedirs(os.path.dirname(output_paths[0]), exist_ok=True)
                for path, image in zip(output_paths, images):
                    image.save(path, format=args.file_format, **SAVE_OPTIONS[args.file_format])

    else:
        total = len(dataset)
        coco_start = max(0, int(getattr(args, 'coco_start', 0) or 0))
        coco_end = int(getattr(args, 'coco_end', None) or total)
        coco_end = min(total, coco_end)
        for num_prompt in tqdm(range(coco_start, coco_end)):
            prompt = dataset[num_prompt]
            seed = int(seeds[num_prompt])
            output_paths = [f'{args.output_dir}/{num_prompt}.{EXTENSIONS[args.file_format]}']
            if all(os.path.exists(path) for path in output_paths):
                    skipped += 1
                    continue
            images = run_image_model(
                model_type=args.model_name,
                pipe=pipeline,
                prompt=prompt,
                seed=seed,
                device=device,
                num_images=1,
            )
            if vector_control is not None:
                vector_control.reset()
            os.makedirs(os.path.dirname(output_paths[0]), exist_ok=True)
            for path, image in zip(output_paths, images):
                image.save(path, format=args.file_format, **SAVE_OPTIONS[args.file_format])
            generated += 1
        


    print(f'Skipped {skipped} images, generated {generated} images')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main_parser = parser.add_argument_group('Common arguments')

    # Generation params
    main_parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True,
                             help='Diffusion model name used for generation')
    main_parser.add_argument('--generate_concept', type=str, required=True, help='Concept for which to generate images')
    main_parser.add_argument('--generate_concept2', type=str, default=None,
                             help='Optional second concept (used by --prompt with {concept2} substitution)')
    main_parser.add_argument('--prompt', type=str, default=None,
                             help='Full prompt template. Use {concept} (and optionally {concept2}) for '
                                  'substitution. If not given, falls back to "a photo of a {generate_concept}".')
    main_parser.add_argument('--output_dir', type=str, required=True, help='Directory where generated images should be written')
    main_parser.add_argument('--num_images_per_prompt', type=int, default=10, help='Number of images to generate for each prompt')
    main_parser.add_argument('--batch_size', type=int, default=1, help='Batch size used for image generation')
    main_parser.add_argument('--seed', type=int, default=0, help='Starting seed for each prompt')
    main_parser.add_argument('--file_format', type=str, choices=['PNG', 'JPEG'], default='PNG', help='File format for generated images')
    main_parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use from the dataset')
    main_parser.add_argument('--coco_start', type=int, default=None, help='[coco mode] First CSV row index (inclusive) to process')
    main_parser.add_argument('--coco_end', type=int, default=None, help='[coco mode] Last CSV row index (exclusive) to process')

    # Steering params
    main_parser.add_argument('--steering_method', type=str, choices=['casteer'], default=None)
    main_parser.add_argument('--steering_strength', type=float, default=None)
    main_parser.add_argument('--control_mode', type=DiffusionVectorControlMode, choices=[str(x) for x in DiffusionVectorControlMode],
                        default='attn_output', help='Vector control mode for steering diffusion models')
    main_parser.add_argument('--intermediate_clipping', action='store_true', help='Apply intermediate clipping like CASteer for leace and mean_matching')
    main_parser.add_argument('--renormalize_after_steering', action='store_true', help='Renormalize vectors after steering for leace and mean_matching')
    main_parser.add_argument('--use_all_diffusion_steps', action='store_true', help='Use all diffusion steps for leace and mean_matching')
    main_parser.add_argument('--multi_attribute_debias', action='store_true', help='Enable simultaneous multi-attribute debiasing using controller_multi')
    main_parser.add_argument('--multi_attributes', type=str, default=None, help='Comma-separated list of attributes for multi-attribute debiasing (e.g. "race,gender,age,body"). Defaults to "race,gender" if --multi_attribute_debias is set.')

    subparsers = parser.add_subparsers(dest='command')

    # Params for concept erasure
    erase_parser = subparsers.add_parser('erase')
    erase_parser.add_argument('--concept_path', type=str, required=True,
                              help='Path to concept vectors which are used to erase the concept from the generated images')

    # Params for concept translation
    translate_parser = subparsers.add_parser('translate')
    translate_parser.add_argument('--source_concept_path', type=str, required=False,
                                  help='Path to concept vectors which should be translated to the other concept')
    translate_parser.add_argument('--target_concept_path', type=str, required=False,
                                  help='Path to concept vectors which should be the target for translation')
    
    translate_parser.add_argument('--save_vectors', action='store_true',
                                  help='Vectors to be saved for analysis later')
    
    translate_parser.add_argument('--save_vectors_path', type=str,
                                  help='path where vectors are to be stored')
    
    translate_parser.add_argument('--attribute', type=str,
                                  help='this is for reading the appropriate threshold')

    translate_parser.add_argument('--do_debias', type=str,
                                  help='if do debiasing')

    translate_parser.add_argument('--do_erase', type=str,
                                  help='if do erasing when debiasing')

    translate_parser.add_argument('--do_threshold', type=str,
                                  help='if do thresholding when debiasing')

    translate_parser.add_argument('--gate_threshold_multiplier', type=float, default=1.0,
                                  help='multiply the gating threshold thr^a (Eq. 5) by this scalar; '
                                       'used for the threshold-sensitivity rebuttal experiment. '
                                       'Default 1.0 = paper midpoint.')

    translate_parser = subparsers.add_parser('none')
    args = parser.parse_args()
    
    main(args)
