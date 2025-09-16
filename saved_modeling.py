import warnings
from typing import Any, List, Optional, Tuple, Union
import os

import torch.utils.checkpoint
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
import cv2

# Import visualization libraries
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ImportError:
    sns = None

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel
from .modeling_phi3 import Phi3ForCausalLM

logger = logging.get_logger(__name__)


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Phi3DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)
    
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_with_attention_visualization(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = True,  # Force attention output
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            visualize_attention: bool = True,
            save_path: str = "./attention_maps/"
        ):
        """
        Modified forward pass that captures and visualizes attention maps
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Ensure attention outputs are enabled
        output_attentions = True
        
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        print("before self.vision_model")
        
        # Extract vision features with attention
        vit_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=self.select_layer != -1,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        print("after self.vision_model")
        
        if self.select_layer == -1:
            vit_embeds = vit_outputs.last_hidden_state
        else:
            vit_embeds = vit_outputs.hidden_states[self.select_layer]
        
        # Store vision attention maps
        vision_attentions = vit_outputs.attentions if hasattr(vit_outputs, 'attentions') else None
        
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids_flat = input_ids.reshape(B * N)
        selected = (input_ids_flat == self.img_context_token_id)
        
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)
        
        print("before outputs from language model")

        # Forward through language model with attention output
        try: 
            outputs = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        except Exception as e:
            print(f'Error during language model forward pass: {e}')
            raise e
        
        print("after outputs from language model")
        
        # Visualize attention maps if requested
        if visualize_attention:
            self.visualize_attention_maps(
                vision_attentions=vision_attentions,
                language_attentions=outputs.attentions,
                pixel_values=pixel_values,
                input_ids=input_ids,
                save_path=save_path
            )
        
        # Return outputs with attention information
        if return_dict:
            result = CausalLMOutputWithPast(
                loss=None,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            # Add vision attention to the output
            result.vision_attentions = vision_attentions
            return result
        else:
            return (outputs.logits, outputs.past_key_values, outputs.hidden_states, 
                    outputs.attentions, vision_attentions)

    def visualize_attention_maps(
            self,
            vision_attentions: Optional[List[torch.Tensor]] = None,
            language_attentions: Optional[List[torch.Tensor]] = None,
            pixel_values: Optional[torch.Tensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            save_path: str = "./attention_maps/"
        ):
        """
        Visualize attention maps for both vision and language components
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Visualize Language Model attention (prompt tokens to image tokens)
        if language_attentions is not None:
            print("visualizing language attention")
            self._visualize_language_attention_to_images(language_attentions, input_ids, save_path)

    def _visualize_language_attention_to_images(self, language_attentions, input_ids, save_path):
        """
        Visualize prompt tokens' attention to image tokens and overlay on resized image
        """
        num_layers = len(language_attentions)
        batch_size = input_ids.shape[0]
        
        # Load the resized image
        resized_image_path = './examples/resized_images/resized_image.png'
        if os.path.exists(resized_image_path):
            resized_img = cv2.imread(resized_image_path)
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        else:
            print(f"Warning: Resized image not found at {resized_image_path}")
            resized_img = None
        
        for layer_idx, attention in enumerate(language_attentions):
            # attention shape: [batch_size, num_heads, seq_len, seq_len]
            attention = attention.detach().cpu().float().numpy()
            
            for batch_idx in range(min(batch_size, 1)):  # Process first batch
                # Average across attention heads
                avg_attention = attention[batch_idx].mean(axis=0)  # [seq_len, seq_len]
                
                # Find image token positions and prompt token positions
                input_ids_cpu = input_ids[batch_idx].cpu().numpy()
                img_token_positions = np.where(input_ids_cpu == self.img_context_token_id)[0]
                
                if len(img_token_positions) == 0:
                    print(f"No image tokens found in batch {batch_idx}")
                    continue
                
                # Find prompt tokens (tokens that are not image tokens)
                all_positions = np.arange(len(input_ids_cpu))
                prompt_positions = np.setdiff1d(all_positions, img_token_positions)
                
                if len(prompt_positions) == 0:
                    print(f"No prompt tokens found in batch {batch_idx}")
                    continue
                
                print(f"Layer {layer_idx}, Batch {batch_idx}: Found {len(img_token_positions)} image tokens, {len(prompt_positions)} prompt tokens")
                
                # Extract attention from prompt tokens to image tokens
                # avg_attention[prompt_positions, :][:, img_token_positions] gives us [num_prompt_tokens, num_image_tokens]
                prompt_to_image_attention = avg_attention[prompt_positions, :][:, img_token_positions]
                
                # Average attention across all prompt tokens to get attention for each image token
                # Shape: [num_image_tokens]
                avg_prompt_to_image = prompt_to_image_attention.mean(axis=0)
                
                print(f"Average prompt-to-image attention shape: {avg_prompt_to_image.shape}")
                print(f"Number of image tokens: {len(img_token_positions)}")
                
                # Calculate grid dimensions from actual number of image tokens
                # For 768 tokens, this should be approximately sqrt(768) ≈ 27.7
                # Let's check for common grid sizes that could result in 768 tokens
                num_image_tokens = len(img_token_positions)
                
                # Try to find the correct grid size
                # 768 = 32 * 24, but that's not square
                # Let's check if it's close to a square number
                sqrt_tokens = int(np.sqrt(num_image_tokens))
                possible_grids = [sqrt_tokens, sqrt_tokens + 1, sqrt_tokens - 1]
                
                grid_size = None
                for candidate_grid in possible_grids:
                    if candidate_grid * candidate_grid == num_image_tokens:
                        grid_size = candidate_grid
                        break
                
                # If no perfect square, try common aspect ratios
                if grid_size is None:
                    # Try some common factorizations for 768
                    factors = []
                    for i in range(1, int(np.sqrt(num_image_tokens)) + 1):
                        if num_image_tokens % i == 0:
                            factors.append((i, num_image_tokens // i))
                    
                    # Prefer factors that are closest to square
                    if factors:
                        # Sort by how close to square they are
                        factors.sort(key=lambda x: abs(x[0] - x[1]))
                        grid_h, grid_w = factors[0]
                        print(f"Using rectangular grid: {grid_h} x {grid_w} = {grid_h * grid_w}")
                        
                        # For rectangular grids, we'll use the first dimension as height
                        spatial_attention = avg_prompt_to_image.reshape(grid_h, grid_w)
                    else:
                        print(f"Cannot find suitable grid dimensions for {num_image_tokens} tokens - skipping layer {layer_idx}")
                        continue
                else:
                    print(f"Using square grid: {grid_size} x {grid_size} = {grid_size * grid_size}")
                    spatial_attention = avg_prompt_to_image.reshape(grid_size, grid_size)
                
                print(f"Number of image tokens: {num_image_tokens}")
                print(f"Final spatial attention shape: {spatial_attention.shape}")
                
                # Remove the old reshape attempt since we handled it above
                # Create attention overlay on resized image
                if resized_img is not None:
                    self._create_attention_overlay(
                        resized_img, 
                        spatial_attention, 
                        f'{save_path}/prompt_to_image_layer_{layer_idx}_batch_{batch_idx}_overlay.png',
                        f'Layer {layer_idx} - Batch {batch_idx} - Prompt Tokens Attention to Image',
                        cmap='viridis'
                    )
                
                # Also save standalone heatmap
                plt.figure(figsize=(8, 6))
                if sns is not None:
                    sns.heatmap(spatial_attention, annot=False, cmap='viridis', square=True)
                else:
                    plt.imshow(spatial_attention, cmap='viridis', aspect='equal')
                    plt.colorbar()
                plt.title(f'Layer {layer_idx} - Batch {batch_idx} - Prompt to Image Attention')
                plt.savefig(f'{save_path}/prompt_to_image_layer_{layer_idx}_batch_{batch_idx}_heatmap.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create a detailed analysis plot
                plt.figure(figsize=(15, 5))
                
                # Plot 1: Attention distribution
                plt.subplot(1, 3, 1)
                plt.bar(range(len(avg_prompt_to_image)), avg_prompt_to_image)
                plt.title(f'Layer {layer_idx} - Attention Distribution')
                plt.xlabel('Image Token Position')
                plt.ylabel('Average Attention Weight')
                
                # Plot 2: Spatial attention heatmap
                plt.subplot(1, 3, 2)
                plt.imshow(spatial_attention, cmap='viridis', aspect='equal')
                plt.colorbar()
                plt.title('Spatial Attention Map')
                
                # Plot 3: Attention statistics
                plt.subplot(1, 3, 3)
                stats_data = {
                    'Max': np.max(avg_prompt_to_image),
                    'Min': np.min(avg_prompt_to_image),
                    'Mean': np.mean(avg_prompt_to_image),
                    'Std': np.std(avg_prompt_to_image)
                }
                plt.bar(stats_data.keys(), stats_data.values())
                plt.title('Attention Statistics')
                plt.ylabel('Value')
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/prompt_to_image_layer_{layer_idx}_batch_{batch_idx}_analysis.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()

    def _create_attention_overlay(self, original_image, attention_map, save_path, title, cmap='viridis', alpha=0.6):
        """
        Create an overlay of attention map on original image
        """
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # Resize attention map to match image size
        if original_image.shape[:2] != attention_map.shape:
            attention_resized = cv2.resize(attention_map, 
                                        (original_image.shape[1], original_image.shape[0]), 
                                        interpolation=cv2.INTER_CUBIC)
        else:
            attention_resized = attention_map
        
        # Normalize attention map to 0-1
        attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
        
        # Create overlay
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Show original image
        ax.imshow(original_image)
        
        # Overlay attention map with transparency
        im = ax.imshow(attention_normalized, alpha=alpha, cmap=cmap, vmin=0, vmax=1)
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved attention overlay to: {save_path}")
    

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, num_patches_list, questions, generation_config, history=None,
                         return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                         IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        from .conversation import get_conv_template

        queries = []
        if verbose:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}, num_patches_list: {num_patches_list}')
        for idx, num_patches in enumerate(num_patches_list):
            image_token = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            question = image_token + '\n' + questions[idx]
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            queries.append(query)
        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    def chat_with_attention_visualization(
            self, tokenizer, pixel_values, question, generation_config, 
            save_path="./attention_maps/", history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', 
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, **kwargs):
        """
        Chat method that also visualizes attention maps
        """
        # Prepare inputs similar to regular chat method
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        
        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        
        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        # Replace image tokens
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        
        # Create image flags
        if pixel_values is not None:
            image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=pixel_values.device)
        else:
            image_flags = torch.zeros(1, 1, dtype=torch.long, device=input_ids.device)
        
        # Forward pass with attention visualization
        with torch.no_grad():
            print("before forward_with_attention_visualization")
            outputs = self.forward_with_attention_visualization(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                visualize_attention=True,
                save_path=save_path
            )
            print("after forward_with_attention_visualization")
        
        # Generate response
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        
        if return_history:
            return response, history
        else:
            if verbose:
                query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
                query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    
    