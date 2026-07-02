import os
import torch
from diffusers import FluxPipeline

HF_TOKEN = os.environ.get('HF_TOKEN')

from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image, Transformer2DModel, LCMScheduler

try:
    # Try the official diffusers import first
    from diffusers import SanaPipeline
except ImportError:
    # If not available in diffusers, try the app.sana_pipeline approach
    try:
        from diffusers.pipelines.sana import SanaPipeline
    except ImportError:
        # Fallback to local SANA installation
        try:
            import sys
            import os
            # Add potential SANA paths
            possible_sana_paths = ['./Sana', '../Sana', './sana', '../sana']
            for path in possible_sana_paths:
                if os.path.exists(path):
                    sys.path.insert(0, path)
                    break
            from app.sana_pipeline import SanaPipeline
        except ImportError:
            SanaPipeline = None
            print("Warning: SANA is not available. Install SANA or ensure diffusers supports SanaPipeline.")

# Try to import SANA-Sprint pipeline
try:
    from diffusers import SanaSprintPipeline
except ImportError:
    SanaSprintPipeline = None
    print("Warning: SANA-Sprint is not available. Install the latest diffusers or ensure SanaSprintPipeline is available.")

# Try to import PixArt pipeline
try:
    from diffusers import PixArtAlphaPipeline
except ImportError:
    PixArtAlphaPipeline = None
    print("Warning: PixArt is not available. Install the latest diffusers or ensure PixArtAlphaPipeline is available.")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _get_sana_load_config() -> dict:
    """
    Pick a safe dtype / placement config for SANA models.
    """
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return {"torch_dtype": dtype, "device_map": "balanced"}
    if torch.mps.is_available():
        return {"torch_dtype": torch.float16}
    return {"torch_dtype": torch.float32}

SUPPORTED_DIFFUSION_MODELS = ['sd14', 'sd15', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo', 'flux', 'flux-schnell', 'sana', 'sana15', 'sana-sprint', 'pixart', 'pixart-alpha', 'flash-pixart', 'sana-06', 'sana-sprint-06']
def init_pipeline_for_image_model(model: str) -> DiffusionPipeline:
    if model == 'sd14':
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16, 
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'sd15':
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            # torch_dtype=torch.float16, 
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'sd21':
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            # torch_dtype=torch.float16, 
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            device_map='balanced',
        )
    elif model == 'sd21-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo", 
            # torch_dtype=torch.float16, 
            # variant="fp16",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            device_map='balanced',
        )
    elif model == 'sdxl':
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'sdxl-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'flux':
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
#             device_map='balanced'
        )
        pipe.enable_model_cpu_offload()
    elif model == 'flux-schnell':
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
#             device_map='balanced'
        )
        pipe.enable_model_cpu_offload()
    elif model == 'sana15':
        if SanaPipeline is None:
            raise ValueError("SANA is not available. Please install SANA or ensure diffusers supports SanaPipeline.")
        sana_config = _get_sana_load_config()
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
            **sana_config,
        )
    elif model == 'sana':
        if SanaPipeline is None:
            raise ValueError("SANA is not available. Please install SANA or ensure diffusers supports SanaPipeline.")
        sana_config = _get_sana_load_config()
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_teacher_diffusers",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
            **sana_config,
        )
    elif model == 'sana-06':
        if SanaPipeline is None:
            raise ValueError("SANA is not available. Please install SANA or ensure diffusers supports SanaPipeline.")
        sana_config = _get_sana_load_config()
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
            **sana_config,
        )
    elif model == 'sana-sprint':
        if SanaSprintPipeline is None:
            raise ValueError("SANA-Sprint is not available. Please install the latest diffusers or ensure SanaSprintPipeline is available.")
        sana_config = _get_sana_load_config()
        pipe = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
            **sana_config,
        )
    elif model == 'sana-sprint-06':
        if SanaSprintPipeline is None:
            raise ValueError("SANA-Sprint is not available. Please install the latest diffusers or ensure SanaSprintPipeline is available.")
        sana_config = _get_sana_load_config()
        pipe = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
            **sana_config,
        )
    elif model in ['pixart', 'pixart-alpha']:
        if PixArtAlphaPipeline is None:
            raise ValueError("PixArt is not available. Please install the latest diffusers or ensure PixArtAlphaPipeline is available.")
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=torch.float16,
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
        )
    elif model == 'flash-pixart':
        if PixArtAlphaPipeline is None:
            raise ValueError("PixArt is not available. Please install the latest diffusers or ensure PixArtAlphaPipeline is available.")
        # Flash-PixArt is typically PixArt with LoRA adaptations
        # For now, we'll use the base PixArt model - users can add LoRA loading if needed
        transformer = Transformer2DModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            subfolder="transformer",
            torch_dtype=torch.float16,
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
        )
        transformer = PeftModel.from_pretrained(
            transformer,
            "jasperai/flash-pixart",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
        )

        # Pipeline
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            transformer=transformer,
            torch_dtype=torch.float16,
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
        )

        # Scheduler
        pipe.scheduler = LCMScheduler.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            subfolder="scheduler",
            timestep_spacing="trailing",
            cache_dir='/gpfs/scratch/acw685/CA_diffusion_debiasing-main/cache',
            token=HF_TOKEN,
        )
    else:
        raise ValueError(f'Unknown model: {model}')
    return pipe


def get_num_denoising_steps(model: str) -> int:
    if model in ('sd14', 'sd15', 'sd21'):
        return 50
    elif model in ('sd21-turbo', 'sdxl-turbo'):
        return 1
    elif model in ('sdxl',):
        return 30
    elif model in ('flux',):
        return 28  # FLUX.1-dev typically uses 28 steps
    elif model in ('sana', 'sana-06', 'sana15'):
        return 20  # SANA typically uses 20 inference steps
    elif model in ('flux-schnell', 'sana-sprint', 'sana-sprint-06', 'flash-pixart'):
        return 1   # SANA-Sprint is optimized for 1-4 steps, using 1 as default for quality/speed balance
    elif model in ('pixart', 'pixart-alpha'):
        return 20  # PixArt typically uses 20 inference steps for good quality
    else:
        raise ValueError('Unknown model type')


def run_image_model(model_type: str, pipe, prompt: str, seed: int, device: torch.device, num_images: int = 1):
    if model_type in ['sd14', 'sd15', 'sd21', 'sdxl']:
        images = pipe(prompt=prompt,
                     num_inference_steps=get_num_denoising_steps(model_type),
                     generator=torch.Generator(device=device).manual_seed(seed),
                     num_images_per_prompt=num_images,
                    ).images

    elif model_type in ['sd21-turbo', 'sdxl-turbo']:
        images = pipe(prompt=prompt,
                     num_inference_steps=get_num_denoising_steps(model_type),
                     guidance_scale=0.0,
                     generator=torch.Generator(device=device).manual_seed(seed),
                     num_images_per_prompt=num_images,
                    ).images
    elif model_type in ['flux']:
        images = pipe(
            prompt,
            guidance_scale=3.5,
            num_inference_steps=get_num_denoising_steps(model_type),
            max_sequence_length=512,
            generator=torch.Generator('cpu').manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    elif model_type in ['flux-schnell']:
        images = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=get_num_denoising_steps(model_type),
            max_sequence_length=256,
            generator=torch.Generator('cpu').manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    elif model_type in ['sana', 'sana-06', 'sana15']:
        images = pipe(
            prompt=prompt,
            num_inference_steps=get_num_denoising_steps(model_type),
            height=1024,
            width=1024,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    elif model_type in ['sana-sprint', 'sana-sprint-06']:
        images = pipe(
            prompt=prompt,
            num_inference_steps=get_num_denoising_steps(model_type),
            height=1024,  # SANA-Sprint is optimized for 1024px images
            width=1024,
            intermediate_timesteps=None,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    elif model_type in ['pixart', 'pixart-alpha']:
        images = pipe(
            prompt=prompt,
            num_inference_steps=get_num_denoising_steps(model_type),
            guidance_scale=4.5,  # Default guidance scale for PixArt
            height=1024,  # PixArt is optimized for 1024px images
            width=1024,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    elif model_type in ['flash-pixart']:
        images = pipe(
            prompt=prompt,
            num_inference_steps=get_num_denoising_steps(model_type),
            guidance_scale=0,  # Default guidance scale for flash-PixArt
            height=1024,  # PixArt is optimized for 1024px images
            width=1024,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images

    return images