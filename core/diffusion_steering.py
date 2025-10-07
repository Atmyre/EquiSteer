import logging
import warnings
import numpy as np
import torch
import abc
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple
import weakref

import enum
import torch.nn.functional as F
from core.math import convert_to_widest_dtype


from .controller import (
    logger,
    EPS,
    DiffusionVectorControlMode,
    ModelToSteer,
    VectorControl,
)

class DiffusionModelType(enum.StrEnum):
    FLUX = 'flux'
    SANA = 'sana'
    SD = 'sd'
    PIXART = 'pixart'

    @staticmethod
    def from_model(model: str) -> 'DiffusionModelType':
        """Convert string to DiffusionModelType enum"""
        model = model.lower().strip()
        
        if model in ['flux', 'flux-schnell']:
            return DiffusionModelType.FLUX
        elif model in ['sana', 'sana-sprint', 'sana-06', 'sana-sprint-06', 'sana15']:
            return DiffusionModelType.SANA
        elif model in ['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo']:
            return DiffusionModelType.SD
        elif model in ['pixart', 'pixart-alpha', 'pixart-xl', 'flash-pixart']:
            return DiffusionModelType.PIXART
        else:
            raise ValueError(f"Unknown model type: {model}. Supported types: {list(DiffusionModelType)}")


class HookManager:
    """Manages registration and cleanup of hooks for vector controls"""
    
    def __init__(self, model_type: DiffusionModelType, flux_image_seq_len: int = None):
        self.hooks = []
        self.controls = []
        self.module_info = {}  # Maps modules to their place_in_unet info
        self.flux_image_seq_len = flux_image_seq_len  # Optional: specify image sequence length for FLUX single blocks
        self.model_type = model_type

    def register_vector_controls_with_hooks(self, model, *controls: VectorControl):
        """Register vector controls using PyTorch hooks instead of method overrides"""
        self.controls = list(controls)
        self._clear_hooks()
        
        # Find all attention modules and register hooks
        block_count = self._register_hooks_recursive(model)
        
        # Set the number of attention layers for all controls
        for control in self.controls:
            control.num_attn_layers = block_count
            
        return self.hooks
    
    def _register_hooks_recursive(self, model) -> int:
        """Recursively find transformer blocks and register hooks"""
        block_count = 0
        
        #Check if this is a FLUX model
        if self.model_type == DiffusionModelType.FLUX:
            block_count += self._register_flux_model_hooks(model)
        # Check if this is a SANA model
        elif self.model_type == DiffusionModelType.SANA:
            block_count += self._register_sana_model_hooks(model)
        # Check if this is a PixArt model
        elif self.model_type == DiffusionModelType.PIXART:
            block_count += self._register_pixart_model_hooks(model)
        elif self.model_type == DiffusionModelType.SD:
            # Traditional SD UNet structure
            for name, module in model.named_children():
                if "down" in name:
                    block_count += self._register_hooks_in_submodule(module, "down")
                elif "up" in name:
                    block_count += self._register_hooks_in_submodule(module, "up")
                elif "mid" in name:
                    block_count += self._register_hooks_in_submodule(module, "mid")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Supported types: {list(DiffusionModelType)}")

        return block_count

    def _register_sana_model_hooks(self, model) -> int:
        """Register hooks for SANA model architecture"""
        block_count = 0
        
        # SANA models have a transformer structure with blocks
        # Look for transformer blocks
        if hasattr(model, 'blocks'):
            for block in model.blocks:
                self._register_hooks_for_sana_block(block, "sana")
                block_count += 1
        
        # Also check transformer submodule
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            for block in model.transformer.blocks:
                self._register_hooks_for_sana_block(block, "sana")
                block_count += 1
        
        # Fallback: search recursively for any SANA blocks
        if block_count == 0:
            for name, module in model.named_modules():
                class_name = module.__class__.__name__
                # print(class_name)
                if any(sana_name in class_name for sana_name in ['SanaTransformerBlock']):
                    self._register_hooks_for_sana_block(module, "sana")
                    block_count += 1
        
        return block_count
    
    def _register_pixart_model_hooks(self, model) -> int:
        """Register hooks for PixArt model architecture"""
        block_count = 0
        
        # PixArt models have a transformer structure with blocks
        # Look for transformer blocks
        if hasattr(model, 'blocks'):
            for block in model.blocks:
                self._register_hooks_for_pixart_block(block, "pixart")
                block_count += 1
        
        # Also check transformer submodule
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            for block in model.transformer.blocks:
                self._register_hooks_for_pixart_block(block, "pixart")
                block_count += 1
        
        # Fallback: search recursively for any PixArt blocks
        if block_count == 0:
            for name, module in model.named_modules():
                class_name = module.__class__.__name__
                # print(class_name)
                if any(pixart_name in class_name for pixart_name in ['PixArtTransformerBlock', 'Transformer2DModelBlock', 'BasicTransformerBlock']):
                    self._register_hooks_for_pixart_block(module, "pixart")
                    block_count += 1
        
        return block_count
    
    def _register_flux_model_hooks(self, model) -> int:
        """Register hooks for FLUX model architecture"""
        block_count = 0
        
        # FLUX models have a flat transformer structure
        # Look for joint_blocks and single_blocks
        if hasattr(model, 'joint_blocks'):
            for block in model.joint_blocks:
                self._register_hooks_for_flux_block(block, "joint")
                block_count += 1
                
        if hasattr(model, 'single_blocks'):
            for block in model.single_blocks:
                self._register_hooks_for_flux_block(block, "single")
                block_count += 1
        
        # Fallback: search recursively for any FLUX blocks
        if block_count == 0:
            for name, module in model.named_modules():
                class_name = module.__class__.__name__
                # print(class_name)
                if class_name in ['FluxTransformerBlock', 'FluxSingleTransformerBlock']:
                    # Determine place based on block type or name
                    if class_name == 'FluxSingleTransformerBlock':
                        place = "single"
                    else:
                        place = "joint"
                    self._register_hooks_for_flux_block(module, place)
                    block_count += 1
        
        return block_count
    
    def _register_hooks_in_submodule(self, module, place_in_unet: str) -> int:
        """Register hooks in a submodule for a specific place in UNet"""
        block_count = 0
        
        for name, submodule in module.named_modules():
            class_name = submodule.__class__.__name__
            
            # Support both SD and FLUX architectures
            if class_name == 'BasicTransformerBlock':
                # Standard Stable Diffusion blocks
                self._register_hooks_for_sd_block(submodule, place_in_unet)
                block_count += 1
            elif class_name in ['JointTransformerBlock', 'SingleTransformerBlock', 'MMDiTBlock', 'FluxTransformerBlock', 'FluxSingleTransformerBlock']:
                # FLUX DiT blocks (various naming conventions)
                self._register_hooks_for_flux_block(submodule, place_in_unet)
                block_count += 1
            elif any(sana_name in class_name for sana_name in ['SanaTransformerBlock']):
                # SANA transformer blocks
                self._register_hooks_for_sana_block(submodule, place_in_unet)
                block_count += 1
            elif any(pixart_name in class_name for pixart_name in ['PixArtTransformerBlock', 'Transformer2DModelBlock']):
                # PixArt transformer blocks
                self._register_hooks_for_pixart_block(submodule, place_in_unet)
                block_count += 1
            elif hasattr(submodule, 'attn') and hasattr(submodule.attn, 'to_q'):
                # Generic transformer block detection for FLUX variants
                if self._is_flux_attention_block(submodule):
                    self._register_hooks_for_flux_block(submodule, place_in_unet)
                    block_count += 1
                
        return block_count
    
    def _register_hooks_for_sd_block(self, block, place_in_unet: str):
        """Register hooks for a specific BasicTransformerBlock (Stable Diffusion)"""
        # Store module info for the hook callbacks
        self.module_info[id(block)] = place_in_unet
        
        # Register hooks based on control modes
        for control in self.controls:
            if control._mode == DiffusionVectorControlMode.ATTN_OUTPUT:
                # Hook into the cross-attention output
                if hasattr(block, 'attn2') and block.attn2 is not None:
                    hook = block.attn2.register_forward_hook(
                        self._create_attn_output_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode == DiffusionVectorControlMode.ATTN_HEADS:
                # Hook into attention heads - need to hook into the attention mechanism itself
                if hasattr(block, 'attn2') and block.attn2 is not None:
                    hook = block.attn2.register_forward_hook(
                        self._create_attn_heads_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode in [DiffusionVectorControlMode.ATTN_KEY, 
                                 DiffusionVectorControlMode.ATTN_VALUE,
                                 DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                # Hook into key/value computation
                if hasattr(block, 'attn2') and block.attn2 is not None:
                    # Hook into to_k and to_v modules
                    if control._mode in [DiffusionVectorControlMode.ATTN_KEY, DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                        hook = block.attn2.to_k.register_forward_hook(
                            self._create_key_hook(control, place_in_unet)
                        )
                        self.hooks.append(hook)
                    if control._mode in [DiffusionVectorControlMode.ATTN_VALUE, DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                        hook = block.attn2.to_v.register_forward_hook(
                            self._create_value_hook(control, place_in_unet)
                        )
                        self.hooks.append(hook)
    
    def _register_hooks_for_flux_block(self, block, place_in_unet: str):
        """Register hooks for FLUX DiT blocks (JointTransformerBlock, SingleTransformerBlock, etc.)"""
        # Store module info for the hook callbacks
        self.module_info[id(block)] = place_in_unet
        
        # Register hooks based on control modes for FLUX architecture
        for control in self.controls:
            if control._mode == DiffusionVectorControlMode.ATTN_OUTPUT:
                # FLUX blocks may have different attention module names
                attn_module = self._get_flux_attention_module(block)
                if attn_module is not None:
                    hook = attn_module.register_forward_hook(
                        self._create_flux_attn_output_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode == DiffusionVectorControlMode.ATTN_HEADS:
                attn_module = self._get_flux_attention_module(block)
                if attn_module is not None:
                    hook = attn_module.register_forward_hook(
                        self._create_flux_attn_heads_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode in [DiffusionVectorControlMode.ATTN_KEY, 
                                 DiffusionVectorControlMode.ATTN_VALUE,
                                 DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                # Hook into FLUX key/value computation
                self._register_flux_key_value_hooks(block, control, place_in_unet)
    
    def _register_hooks_for_sana_block(self, block, place_in_unet: str):
        """Register hooks for SANA transformer blocks"""
        # Store module info for the hook callbacks
        self.module_info[id(block)] = place_in_unet
        
        # Register hooks based on control modes for SANA architecture
        for control in self.controls:
            if control._mode == DiffusionVectorControlMode.ATTN_OUTPUT:
                # SANA blocks may have different attention module names
                attn_module = self._get_sana_attention_module(block)
                # print(block)
                # print(attn_module)
                if attn_module is not None:
                    hook = attn_module.register_forward_hook(
                        self._create_sana_attn_output_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode == DiffusionVectorControlMode.ATTN_HEADS:
                attn_module = self._get_sana_attention_module(block)
                if attn_module is not None:
                    hook = attn_module.register_forward_hook(
                        self._create_sana_attn_heads_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode in [DiffusionVectorControlMode.ATTN_KEY, 
                                 DiffusionVectorControlMode.ATTN_VALUE,
                                 DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                # Hook into SANA key/value computation
                self._register_sana_key_value_hooks(block, control, place_in_unet)
    
    def _register_hooks_for_pixart_block(self, block, place_in_unet: str):
        """Register hooks for PixArt transformer blocks"""
        # Store module info for the hook callbacks
        self.module_info[id(block)] = place_in_unet
        
        # Register hooks based on control modes for PixArt architecture
        for control in self.controls:
            if control._mode == DiffusionVectorControlMode.ATTN_OUTPUT:
                # PixArt blocks may have different attention module names
                attn_module = self._get_pixart_attention_module(block)
                # print(block)
                # print(attn_module)
                if attn_module is not None:
                    hook = attn_module.register_forward_hook(
                        self._create_pixart_attn_output_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode == DiffusionVectorControlMode.ATTN_HEADS:
                attn_module = self._get_pixart_attention_module(block)
                if attn_module is not None:
                    hook = attn_module.register_forward_hook(
                        self._create_pixart_attn_heads_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode in [DiffusionVectorControlMode.ATTN_KEY, 
                                 DiffusionVectorControlMode.ATTN_VALUE,
                                 DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                # Hook into PixArt key/value computation
                self._register_pixart_key_value_hooks(block, control, place_in_unet)
    
    def _is_flux_attention_block(self, module) -> bool:
        """Check if a module is a FLUX-style attention block"""
        # Check for FLUX-specific attributes
        flux_indicators = [
            hasattr(module, 'norm1') and hasattr(module, 'norm2'),  # Common in DiT
            hasattr(module, 'attn') and hasattr(module, 'mlp'),     # DiT structure
            hasattr(module, 'adaLN_modulation'),                    # Adaptive layer norm
            hasattr(module, 'txt_attn'),                            # Text attention in double stream
            hasattr(module, 'img_attn'),                            # Image attention in double stream
        ]
        
        # Return True if it has multiple FLUX indicators
        return sum(flux_indicators) >= 2
    
    def _get_flux_attention_module(self, block):
        """Get the appropriate attention module from a FLUX block"""
        # Try different possible attention module names in FLUX
        # print(block)
        possible_names = ['txt_attn', 'attn']
        
        for name in possible_names:
            if hasattr(block, name):
                attn_module = getattr(block, name)
                if attn_module is not None and hasattr(attn_module, 'to_q'):
                    return attn_module
        
        return None
    
    def _get_sana_attention_module(self, block):
        """Get the appropriate attention module from a SANA block"""
        # Try different possible attention module names in SANA
        possible_names = ['attn2', 'attention', 'self_attn', 'Attention']
        
        for name in possible_names:
            if hasattr(block, name):
                attn_module = getattr(block, name)
                if attn_module is not None and hasattr(attn_module, 'to_q'):
                    return attn_module
        
        return None
    
    def _get_pixart_attention_module(self, block):
        """Get the appropriate attention module from a PixArt block"""
        # Try different possible attention module names in PixArt
        possible_names = ['attn1', 'attn2', 'attention', 'self_attn', 'cross_attn']
        
        for name in possible_names:
            if hasattr(block, name):
                attn_module = getattr(block, name)
                if attn_module is not None and hasattr(attn_module, 'to_q'):
                    return attn_module
        
        return None
    
    def _split_flux_single_tokens(self, tensor, image_seq_len=None):
        """
        Split FluxSingleTransformerBlock tokens into text and image parts.
        
        Args:
            tensor: Input tensor with shape [batch, seq_len, ...]
            image_seq_len: Length of image sequence. If None, try to infer from tensor shape.
            
        Returns:
            tuple: (text_tokens, image_tokens) or (None, tensor) if no split needed
        """
        total_seq_len = tensor.shape[1]
        
        if image_seq_len is None:
            # Try to infer image sequence length from common FLUX patterns
            # Common FLUX image sequence lengths (based on image resolution and patch size)
            # For example: 32x32 patches = 1024, 64x64 patches = 4096, etc.
            common_image_lengths = [256, 1024, 4096, 16384]  # Common image token counts
            
            for img_len in common_image_lengths:
                if total_seq_len > img_len and img_len > 0:
                    # Use the largest image length that fits
                    image_seq_len = img_len
                    break
            
            if image_seq_len is None:
                # If we can't infer, return the whole tensor as image tokens
                return None, tensor
        
        if image_seq_len >= total_seq_len:
            # If image length is greater than or equal to total sequence length,
            # treat everything as image tokens
            return None, tensor
        
        # Calculate text sequence length
        text_seq_len = total_seq_len - image_seq_len
        
        # Split the tensor: text tokens first, then image tokens
        text_tokens = tensor[:, :text_seq_len, ...]
        image_tokens = tensor[:, text_seq_len:, ...]
        
        return text_tokens, image_tokens
    
    def _create_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for attention output"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # Apply control to the output
            # Add extra dimension for compatibility with original code
            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            return controlled_output[..., 0, :]
        
        return hook_fn
    
    def _create_attn_heads_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for attention heads"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # For attention heads, we need to reshape the output appropriately
            # This assumes the output is from scaled_dot_product_attention
            if len(output.shape) == 4:  # [batch, heads, seq_len, head_dim]
                # Transpose to [batch, seq_len, heads, head_dim] for control
                output_transposed = output.transpose(1, 2)
                controlled_output = control(output_transposed, place_in_unet)
                return controlled_output.transpose(1, 2)
            else:
                return control(output, place_in_unet)
        
        return hook_fn
    
    def _create_flux_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for FLUX attention output"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # FLUX attention outputs may have different tensor structure
            # Handle both single tensor and tuple outputs
            if isinstance(output, tuple):
                attn_output = output[0]
                encoder_attn_output = output[1] if len(output) > 1 else None
                rest = output[2:] if len(output) > 2 else ()
            else:
                attn_output = output
                encoder_attn_output = None
                rest = ()
            
            # For FluxSingleTransformerBlock (place_in_unet == "single"), 
            # we only work with the image part of the attention output
            if place_in_unet == "single":
                # Split the attention output into text and image parts
                text_tokens, image_tokens = self._split_flux_single_tokens(attn_output, self.flux_image_seq_len)
                
                if image_tokens is not None:
                    # Apply control only to the image part
                    image_expanded = image_tokens[..., None, :]
                    controlled_image = control(image_expanded, place_in_unet)
                    controlled_image = controlled_image[..., 0, :].to(torch.bfloat16)
                    
                    # Reconstruct the full tensor with controlled image tokens
                    if text_tokens is not None:
                        controlled_output = torch.cat([text_tokens.to(torch.bfloat16), controlled_image], dim=1)
                    else:
                        controlled_output = controlled_image
                else:
                    # If we couldn't split, apply control to the whole output
                    output_expanded = attn_output[..., None, :]
                    controlled_output = control(output_expanded, place_in_unet)
                    controlled_output = controlled_output[..., 0, :].to(torch.bfloat16)
            else:
                # For joint blocks or other block types, apply control to the full output
                output_expanded = attn_output[..., None, :]
                controlled_output = control(output_expanded, place_in_unet)
                controlled_output = controlled_output[..., 0, :].to(torch.bfloat16)

            # Return in the same format as input
            # print(f"SHAPE {place_in_unet}: {controlled_output.shape}")
            if encoder_attn_output is not None or rest:
                return (controlled_output, encoder_attn_output) + rest
            else:
                return controlled_output
        
        return hook_fn
    
    def _create_flux_attn_heads_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for FLUX attention heads"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # Handle tuple outputs from FLUX attention
            if isinstance(output, tuple):
                attn_output = output[0]
                rest = output[1:]
            else:
                attn_output = output
                rest = None
            
            # Apply control with proper tensor reshaping for FLUX
            if len(attn_output.shape) == 4:  # [batch, heads, seq_len, head_dim]
                output_transposed = attn_output.transpose(1, 2)
                controlled_output = control(output_transposed, place_in_unet)
                controlled_output = controlled_output.transpose(1, 2)
            else:
                controlled_output = control(attn_output, place_in_unet)
            
            if rest is not None:
                return (controlled_output,) + rest
            else:
                return controlled_output
        
        return hook_fn
    
    
    def _create_sana_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for SANA attention output"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # SANA attention outputs are typically straightforward tensors
            # Apply control to the output
            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            controlled_output = controlled_output[..., 0, :].to(torch.bfloat16)
            
            return controlled_output
        
        return hook_fn
    
    def _create_sana_attn_heads_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for SANA attention heads"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # Apply control with proper tensor reshaping for SANA
            if len(output.shape) == 4:  # [batch, heads, seq_len, head_dim]
                output_transposed = output.transpose(1, 2)
                controlled_output = control(output_transposed, place_in_unet)
                controlled_output = controlled_output.transpose(1, 2)
            else:
                controlled_output = control(output, place_in_unet)
            
            return controlled_output
        
        return hook_fn
    
    def _create_pixart_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for PixArt attention output"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # PixArt attention outputs are typically straightforward tensors
            # Apply control to the output
            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            controlled_output = controlled_output[..., 0, :].to(torch.bfloat16)
            
            return controlled_output
        
        return hook_fn
    
    def _create_pixart_attn_heads_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for PixArt attention heads"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # Apply control with proper tensor reshaping for PixArt
            if len(output.shape) == 4:  # [batch, heads, seq_len, head_dim]
                output_transposed = output.transpose(1, 2)
                controlled_output = control(output_transposed, place_in_unet)
                controlled_output = controlled_output.transpose(1, 2)
            else:
                controlled_output = control(output, place_in_unet)
            
            return controlled_output
        
        return hook_fn
    
    
    def _create_pixart_value_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for PixArt attention values"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
                
            # Similar to PixArt key hook but for values
            batch_size, seq_len = output.shape[:2]
            
            if len(output.shape) == 3:  # [batch, seq_len, hidden_dim]
                hidden_dim = output.shape[2]
                num_heads = getattr(module, 'num_heads', 8)  # Default fallback
                head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
                
                if hidden_dim % head_dim == 0:
                    output_reshaped = output.view(batch_size, seq_len, num_heads, head_dim)
                    controlled_output = control(output_reshaped, place_in_unet)
                    return controlled_output.view(batch_size, seq_len, hidden_dim)
                    
            return control(output, place_in_unet)
        
        return hook_fn
    
    def _clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_info.clear()
    
    def remove_hooks(self):
        """Public method to remove all hooks"""
        self._clear_hooks()
    
    def reset_controls(self):
        """Reset all controls to initial state"""
        for control in self.controls:
            control.reset()


# Convenience functions for backward compatibility
def diffusion_register_vector_controls_with_hooks(model, *controls: VectorControl, model_type: DiffusionModelType, flux_image_seq_len: int = None) -> HookManager:
    """
    Register vector controls using PyTorch hooks instead of method overrides.
    
    Args:
        model: The model to register controls on
        *controls: VectorControl instances to register
        flux_image_seq_len: Optional image sequence length for FLUX single blocks (for text/image token splitting)
        
    Returns:
        HookManager: Manager object that can be used to remove hooks later
    """
    manager = HookManager(model_type=model_type, flux_image_seq_len=flux_image_seq_len)
    manager.register_vector_controls_with_hooks(model, *controls)
    return manager

