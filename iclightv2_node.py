import requests
from PIL import Image
import io
import numpy as np
import torch
import os
import configparser
import tempfile
from fal_client import submit, upload_file

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_path = os.path.join(parent_dir, "config.ini")

config = configparser.ConfigParser()
config.read(config_path)

try:
    fal_key = config['API']['FAL_KEY']
    os.environ["FAL_KEY"] = fal_key
except KeyError:
    print("Error: FAL_KEY not found in config.ini")

def upload_image(image):
    try:
        # Convert the image tensor to a numpy array
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)

        # Ensure the image is in the correct format (H, W, C)
        if image_np.ndim == 4:
            image_np = image_np.squeeze(0)  # Remove batch dimension if present
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
        elif image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

        # Normalize the image data to 0-255 range
        if image_np.dtype == np.float32 or image_np.dtype == np.float64:
            image_np = (image_np * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_np)

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            pil_image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        # Upload the temporary file
        image_url = upload_file(temp_file_path)
        return image_url
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)

class ICLightV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "square_hd"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "initial_latent": (["None", "Left", "Right", "Top", "Bottom"], {"default": "None"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "mask_image": ("IMAGE",),
                "seed": ("INT", {"default": -1}),
                "enable_hr_fix": ("BOOLEAN", {"default": False}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "lowres_denoise": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.01}),
                "highres_denoise": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hr_downscale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "FAL/Image"

    def generate_image(self, prompt, image, image_size, width, height, num_inference_steps, guidance_scale, num_images, 
                       initial_latent, negative_prompt="", mask_image=None, seed=-1, enable_hr_fix=False, 
                       cfg=1.0, lowres_denoise=0.98, highres_denoise=0.95, hr_downscale=0.5, 
                       enable_safety_checker=True, output_format="jpeg", sync_mode=False):
        # Upload input image
        image_url = upload_image(image)
        if not image_url:
            print("Failed to upload input image.")
            return self.create_blank_image()

        # Upload mask image if provided
        mask_image_url = None
        if mask_image is not None:
            mask_image_url = upload_image(mask_image)
            if not mask_image_url:
                print("Failed to upload mask image.")

        # Prepare arguments
        arguments = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_url": image_url,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "initial_latent": initial_latent,
            "cfg": cfg,
            "lowres_denoise": lowres_denoise,
            "highres_denoise": highres_denoise,
            "hr_downscale": hr_downscale,
            "enable_hr_fix": enable_hr_fix,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
            "sync_mode": sync_mode
        }

        # Add mask_image_url if available
        if mask_image_url:
            arguments["mask_image_url"] = mask_image_url

        # Add custom image size if selected
        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            arguments["image_size"] = image_size

        # Add seed if provided
        if seed != -1:
            arguments["seed"] = seed

        try:
            handler = submit("fal-ai/iclight-v2", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with ICLightV2: {str(e)}")
            return self.create_blank_image()

    def process_result(self, result):
        images = []
        for img_info in result["images"]:
            img_url = img_info["url"]
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)

        # Stack the images along a new first dimension
        stacked_images = np.stack(images, axis=0)
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(stacked_images)
        
        return (img_tensor,)

    def create_blank_image(self):   
        blank_img = Image.new('RGB', (512, 512), color='black')
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ICLightV2_fal": ICLightV2,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ICLightV2_fal": "IC Light V2 (fal)"
}