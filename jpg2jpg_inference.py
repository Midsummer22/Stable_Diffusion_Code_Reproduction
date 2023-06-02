import torch
from PIL import Image
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,DDIMScheduler, StableDiffusionImg2ImgPipeline

model_id = "stable-diffusion-v1-5"
pipe = pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

init_image = Image.open("TaylorSwift.png").convert("RGB")

#pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a girl"
negative_prompt = "nsfw,paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.331), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), (more than 2 nipples:1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), bad hands, missing fingers, extra digit, (futa:1.1), bad body, NG_DeepNegative_V1_75T,pubic hair, glans"
with autocast("cuda"):
    image = pipe(prompt,
                 negative_prompt=negative_prompt,
                 image=init_image,
                 num_inference_steps=25,
                 strength=0.8,
                 guidance_scale=7.5).images[0]

image.save("jpg2jpg.png")