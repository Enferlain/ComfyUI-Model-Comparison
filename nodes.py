import torch
import folder_paths
import comfy.sd
import comfy.sample
import comfy.utils
import comfy.samplers
import comfy.model_sampling
import latent_preview
from nodes import VAEDecode
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Copied from comfy_extras/nodes_model_advanced.py
class LCM(comfy.model_sampling.EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        timestep = self.timestep(sigma).view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        x0 = model_input - model_output * sigma

        sigma_data = 0.5
        scaled_timestep = timestep * 10.0 #timestep_scaling

        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

        return c_out * x0 + c_skip * model_input

class ModelComparisoner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_names": ("STRING", {"multiline": True, "default": "sd_xl_base_1.0.safetensors\nv1-5-pruned.ckpt", "tooltip": "List of checkpoint filenames, one per line or comma separated"}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "sampler": ("SAMPLER", ),
                "sigmas": ("SIGMAS", ),
                "latent_image": ("LATENT", ),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                # Model Sampling Inputs
                "sampling_override": (["default", "eps", "v_prediction", "lcm", "x0", "sd3", "flux"], ),
                "zsnr": ("BOOLEAN", {"default": False}),
                "sd3_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.01}),
                # Output Control
                "generate_grid": ("BOOLEAN", {"default": True, "tooltip": "Generate the grid of images (decodes with each model's VAE). Turn off to save memory/time if you only need Latents."}),
                "image_mode": (["grid", "batch"], {"default": "grid", "tooltip": "grid: Concatenate images horizontally. batch: Stack images (N individual images)."}),
                "latent_mode": (["batch", "grid"], {"default": "batch", "tooltip": "batch: Stack latents (N images). grid: Concatenate latents horizontally (1 wide image)."})
            },
            "optional": {
                "row_label_height": ("INT", {"default": 50, "min": 0, "max": 200}),
                "vae": ("VAE", {"tooltip": "Optional VAE to use for decoding. If provided, this VAE is used for ALL models instead of the checkpoint's built-in VAE."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "compare_models"
    CATEGORY = "comparison"

    def compare_models(self, ckpt_names, positive, negative, sampler, sigmas, latent_image, cfg, add_noise, noise_seed, sampling_override, zsnr, sd3_shift, generate_grid, image_mode, latent_mode, row_label_height=50, vae=None):
        # Parse checkpoint names
        if "\n" in ckpt_names:
            names = [x.strip() for x in ckpt_names.split("\n") if x.strip()]
        else:
            names = [x.strip() for x in ckpt_names.split(",") if x.strip()]
        
        results_images = []
        results_latents = []
        decode_node = VAEDecode()

        # Prepare Noise
        latent_samples = latent_image["samples"]
        noise = None

        for i, ckpt_name in enumerate(names):
            print(f"ModelComparisoner: Processing {ckpt_name}")
            try:
                # Load Checkpoint
                try:
                   ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
                except ValueError:
                    print(f"ModelComparisoner: Checkpoint {ckpt_name} not found. Skipping.")
                    continue
                
                # Load model
                out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
                model, clip, loaded_vae = out[:3]
                
                # Determine VAE
                vae_to_use = vae if vae is not None else loaded_vae

                # Apply Model Sampling Override
                if sampling_override != "default":
                    m = model.clone()
                    model_sampling = None
                    
                    if sampling_override in ["eps", "v_prediction", "lcm", "x0"]:
                        sampling_base = comfy.model_sampling.ModelSamplingDiscrete
                        if sampling_override == "eps":
                            sampling_type = comfy.model_sampling.EPS
                        elif sampling_override == "v_prediction":
                            sampling_type = comfy.model_sampling.V_PREDICTION
                        elif sampling_override == "lcm":
                            sampling_type = LCM
                            pass
                        elif sampling_override == "x0":
                            sampling_type = comfy.model_sampling.X0
                        
                        class ModelSamplingAdvanced(sampling_base, sampling_type):
                            pass
                        
                        model_sampling = ModelSamplingAdvanced(model.model.model_config, zsnr=zsnr)
                    
                    elif sampling_override == "sd3":
                        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
                        sampling_type = comfy.model_sampling.CONST
                        
                        class ModelSamplingAdvanced(sampling_base, sampling_type):
                            pass
                        
                        model_sampling = ModelSamplingAdvanced(model.model.model_config)
                        model_sampling.set_parameters(shift=sd3_shift, multiplier=1000)
                    
                    elif sampling_override == "flux":
                         sampling_base = comfy.model_sampling.ModelSamplingFlux
                         sampling_type = comfy.model_sampling.CONST
                         
                         class ModelSamplingAdvanced(sampling_base, sampling_type):
                            pass
                         
                         model_sampling = ModelSamplingAdvanced(model.model.model_config)
                         model_sampling.set_parameters(shift=1.15)

                    if model_sampling is not None:
                        m.add_object_patch("model_sampling", model_sampling)
                        model = m

                
                # Fix channels and Generate Noise
                work_latent = latent_samples.clone()
                work_latent = comfy.sample.fix_empty_latent_channels(model, work_latent)
                
                if noise is None:
                    if not add_noise:
                        noise = torch.zeros(work_latent.size(), dtype=work_latent.dtype, layout=work_latent.layout, device="cpu")
                    else:
                        batch_inds = latent_image.get("batch_index", None)
                        noise = comfy.sample.prepare_noise(work_latent, noise_seed, batch_inds)
                else:
                    if noise.shape != work_latent.shape:
                         if not add_noise:
                            noise_for_model = torch.zeros(work_latent.size(), dtype=work_latent.dtype, layout=work_latent.layout, device="cpu")
                         else:
                            batch_inds = latent_image.get("batch_index", None)
                            noise_for_model = comfy.sample.prepare_noise(work_latent, noise_seed, batch_inds)
                    else:
                        noise_for_model = noise

                # Prepare Callback
                x0_output = {}
                callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
                
                # Sample
                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                noise_mask = latent_image.get("noise_mask", None)
                
                samples = comfy.sample.sample_custom(model, noise_for_model, cfg, sampler, sigmas, positive, negative, work_latent, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
                
                # Accumulate Latents
                results_latents.append(samples)

                if generate_grid:
                    # Decode
                    decoded_images = decode_node.decode(vae_to_use, {"samples": samples})[0]
                    
                    # Add Label
                    annotated_images = []
                    for img_tensor in decoded_images:
                        i = 255. * img_tensor.cpu().numpy()
                        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                        
                        if row_label_height > 0:
                            w, h = img.size
                            new_img = Image.new("RGB", (w, h + row_label_height), (0, 0, 0))
                            new_img.paste(img, (0, row_label_height))
                            
                            draw = ImageDraw.Draw(new_img)
                            try:
                                font = ImageFont.truetype("arial.ttf", 24)
                            except:
                                font = ImageFont.load_default()
                                
                            text = ckpt_name
                            try:
                                text_w = draw.textlength(text, font=font)
                            except AttributeError:
                                 text_w = draw.textsize(text, font=font)[0]
                            
                            draw.text((10, (row_label_height - 20)/2), text, font=font, fill=(255, 255, 255))
                            img = new_img
                        
                        out_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
                        annotated_images.append(out_tensor)
                    
                    final_batch_images = torch.stack(annotated_images)
                    results_images.append(final_batch_images)
                
            except Exception as e:
                print(f"ModelComparisoner: Error processing {ckpt_name}: {e}")
                continue

        # Output Images
        if not results_images:
             out_image = torch.zeros((1, 512, 512, 3))
        else:
            try:
                if image_mode == "grid":
                     out_image = torch.cat(results_images, dim=2)
                else:
                     out_image = torch.cat(results_images, dim=0)
            except RuntimeError:
                print("ModelComparisoner: Image size mismatch.")
                out_image = results_images[0]

        # Output Latents
        if not results_latents:
            out_latent = {"samples": torch.zeros_like(latent_samples)}
        else:
            try:
                 if latent_mode == "grid":
                     out_tensor = torch.cat(results_latents, dim=3)
                 else:
                     out_tensor = torch.cat(results_latents, dim=0)

                 out_latent = {"samples": out_tensor}
            except RuntimeError:
                 print("ModelComparisoner: Latent size mismatch.")
                 out_latent = {"samples": results_latents[0]}

        return (out_image, out_latent)
