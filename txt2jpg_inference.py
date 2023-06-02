import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,DDIMScheduler

model_id = "stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

#pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a car"
negative_prompt = "(EasyNegative), bad_prompt_version2, (watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), ((badhandsv5-neg)), ((badhandv4)), (bad anatomy), deformed hands, NSFW, nude, EasyNegative, (worst quality:1.4), (low quality:1.4), (normal quality:1.4),lowres,crowd"
with autocast("cuda"):
    image = pipe(prompt,
                 negative_prompt=negative_prompt,
                 num_inference_steps=25,
                 width=512,
                 height=768,
                 guidance_scale=7.5).images[0]

image.save("txt2jpg.png")