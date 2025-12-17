import torch
import folder_paths
import comfy.sd
import comfy.sample
import comfy.utils
import comfy.samplers
from nodes import common_ksampler, CLIPTextEncode, VAEDecode
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

class ModelComparisoner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_names": ("STRING", {"multiline": True, "default": "sd_xl_base_1.0.safetensors\nv1-5-pruned.ckpt", "tooltip": "List of checkpoint filenames, one per line or comma separated"}),
                "positive": ("STRING", {"multiline": True, "default": "positive prompt"}),
                "negative": ("STRING", {"multiline": True, "default": "negative prompt"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "latent_image": ("LATENT", ),
            },
            "optional": {
                "row_label_height": ("INT", {"default": 50, "min": 0, "max": 200}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compare_models"
    CATEGORY = "comparison"

    def compare_models(self, ckpt_names, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, latent_image, row_label_height=50):
        # Parse checkpoint names
        # Split by newline or comma
        if "\n" in ckpt_names:
            names = [x.strip() for x in ckpt_names.split("\n") if x.strip()]
        else:
            names = [x.strip() for x in ckpt_names.split(",") if x.strip()]
        
        results = []
        
        encode_node = CLIPTextEncode()
        decode_node = VAEDecode()

        for ckpt_name in names:
            print(f"ModelComparisoner: Processing {ckpt_name}")
            try:
                # Load Checkpoint
                # We use internal comfy APIs
                # folder_paths.get_full_path_or_raise("checkpoints", ckpt_name) might fail if name is partial?
                # CheckpointLoaderSimple uses get_full_path_or_raise
                
                # We need to handle potential errors gracefully
                try:
                   ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
                except ValueError:
                    print(f"ModelComparisoner: Checkpoint {ckpt_name} not found. Skipping.")
                    continue
                
                # Load model
                # Returns (model, clip, vae)
                # We rely on comfy's cache to handle memory, but we iterate sequentially so previous model *should* be offloaded if needed by LRU
                out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
                model, clip, vae = out[:3]
                
                # Encode Prompts
                # CLIPTextEncode.encode(clip, text) -> (conditioning,)
                pos_cond = encode_node.encode(clip, positive)[0]
                neg_cond = encode_node.encode(clip, negative)[0]
                
                # Sample
                # common_ksampler returns (latent_dict,)
                # We make sure to clone latent input if needed, but common_ksampler handles it?
                # common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, ...)
                
                # Important: latent_image is a dict {"samples": tensor}
                # We should probably pass a copy to avoid in-place modification risks across iterations?
                # common_ksampler copies it: "out = latent.copy()"
                
                sampled_latent = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, pos_cond, neg_cond, latent_image, denoise=denoise)[0]
                
                # Decode
                images = decode_node.decode(vae, sampled_latent)[0] # [B, H, W, C]
                
                # Add Label
                # We'll label with the checkpoint name
                annotated_images = []
                for img_tensor in images:
                    # img_tensor is [H, W, C]
                    # Convert to PIL
                    i = 255. * img_tensor.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    
                    if row_label_height > 0:
                        # Add header
                        w, h = img.size
                        new_img = Image.new("RGB", (w, h + row_label_height), (0, 0, 0))
                        new_img.paste(img, (0, row_label_height))
                        
                        draw = ImageDraw.Draw(new_img)
                        try:
                            font = ImageFont.truetype("arial.ttf", 24)
                        except IOError:
                            font = ImageFont.load_default()
                            
                        # Draw Text Centered
                        text = ckpt_name
                        # bbox = draw.textbbox((0, 0), text, font=font)
                        # text_w = bbox[2] - bbox[0]
                        # text_h = bbox[3] - bbox[1]
                        # Compatibility with older PIL?
                        text_w, text_h = draw.textsize(text, font=font) if hasattr(draw, "textsize") else draw.textlength(text, font=font) # approximate
                        
                        # Just draw at 5,5
                        draw.text((10, (row_label_height - 10)/2), text, font=font, fill=(255, 255, 255))
                        
                        img = new_img
                    
                    # Back to Tensor
                    # [H, W, C]
                    out_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
                    annotated_images.append(out_tensor)
                
                # Stack batch back to [B, H, W, C]
                final_batch_images = torch.stack(annotated_images)
                results.append(final_batch_images)
                
            except Exception as e:
                print(f"ModelComparisoner: Error processing {ckpt_name}: {e}")
                # We might want to add a blank error image?
                # Create a black image of same size as latent target size?
                # We don't know the exact pixel size unless we calculate from latent
                # Latent H*8, W*8 roughly.
                continue

        if not results:
             # Return empty batch?
             return (torch.zeros((1, 512, 512, 3)),)

        # Stitch Horizontally
        # results is list of [B, H, W, C]
        # We want [B, H, TotalW, C]
        # Assuming all have same H? If not, resize?
        # Standard generation workflow -> same latent size -> same pixel size.
        
        # Concatenate along Width (dim 2)
        grid = torch.cat(results, dim=2)
        
        return (grid,)
