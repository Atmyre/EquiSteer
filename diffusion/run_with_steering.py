import argparse
import os
import math
import typing as tp

from diffusers import DiffusionPipeline

from core.controller import CrossAttentionOutputSteering, DiffusionVectorControlMode, ModelToSteer, VectorControl
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
    else:
        source_concept = unpickle(args.source_concept_path)
        target_concept = unpickle(args.target_concept_path)

    vector_control = CrossAttentionOutputSteering(
        model_to_steer=ModelToSteer.UNET,
        mode=args.control_mode,
        steer_type=args.steering_method,
        target_concepts=[target_concept],
        source_concepts=[source_concept],
        mu_neutral=mu_neutral,
        sigma_neutral=sigma_neutral if not args.id_cov else None,
        steer_only_up=False,
        steer_back=True,
        strength=args.steering_strength,
        device=device,
        intermediate_clipping=args.intermediate_clipping,
        renormalize_after_steering=args.renormalize_after_steering,
        use_first_diffusion_step=not args.use_all_diffusion_steps,
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

    vector_control = hook_model(pipeline, device, args)

    if args.generate_concept != 'coco':
        dataset = TemplateDataset(
            template_path='exp/datasets/eval/imagenet/template.json',
            concept=args.generate_concept,
            tokenizer_fn=dumb_tokenizer_fn,
        )
        num_images_per_prompt = args.num_images_per_prompt
    else:
        dataset = CocoDataset(
            coco_path='exp/datasets/eval/coco/coco_30k.csv',
            max_samples=args.max_samples,
        )
        num_images_per_prompt = 1
    skipped = generated = 0

    print(f'Generating images for concept {args.generate_concept} and method {args.steering_method} with strength {args.steering_strength}')
    for prompt in dataset:
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

    print(f'Skipped {skipped} images, generated {generated} images')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main_parser = parser.add_argument_group('Common arguments')

    # Generation params
    main_parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True,
                             help='Diffusion model name used for generation')
    main_parser.add_argument('--generate_concept', type=str, required=True, help='Concept for which to generate images')
    main_parser.add_argument('--output_dir', type=str, required=True, help='Directory where generated images should be written')
    main_parser.add_argument('--num_images_per_prompt', type=int, default=10, help='Number of images to generate for each prompt')
    main_parser.add_argument('--batch_size', type=int, default=1, help='Batch size used for image generation')
    main_parser.add_argument('--seed', type=int, default=0, help='Starting seed for each prompt')
    main_parser.add_argument('--file_format', type=str, choices=['PNG', 'JPEG'], default='PNG', help='File format for generated images')
    main_parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use from the dataset')

    # Steering params
    main_parser.add_argument('--steering_method', type=str, choices=['casteer', 'leace', 'mean_matching'], default=None)
    main_parser.add_argument('--steering_strength', type=float, default=None)
    main_parser.add_argument('--control_mode', type=DiffusionVectorControlMode, choices=[str(x) for x in DiffusionVectorControlMode],
                        default='attn_output', help='Vector control mode for steering diffusion models')
    main_parser.add_argument('--intermediate_clipping', action='store_true', help='Apply intermediate clipping like CASteer for leace and mean_matching')
    main_parser.add_argument('--renormalize_after_steering', action='store_true', help='Renormalize vectors after steering for leace and mean_matching')
    main_parser.add_argument('--id_cov', action='store_true', help='Use the identity covariance matrix for leace and mean_matching')
    main_parser.add_argument('--use_all_diffusion_steps', action='store_true', help='Use all diffusion steps for leace and mean_matching')

    subparsers = parser.add_subparsers(dest='command')

    # Params for concept erasure
    erase_parser = subparsers.add_parser('erase')
    erase_parser.add_argument('--concept_path', type=str, required=True,
                              help='Path to concept vectors which are used to erase the concept from the generated images')

    # Params for concept translation
    translate_parser = subparsers.add_parser('translate')
    translate_parser.add_argument('--source_concept_path', type=str, required=True,
                                  help='Path to concept vectors which should be translated to the other concept')
    translate_parser.add_argument('--target_concept_path', type=str, required=True,
                                  help='Path to concept vectors which should be the target for translation')


    args = parser.parse_args()
    
    main(args)
