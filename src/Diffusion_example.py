import torch
from diffusers import StableDiffusionPipeline


model_id = "OFA-Sys/small-stable-diffusion-v0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("mps")

prompt = "an apple, 4k"
image = pipe(prompt).images[0]  


pipe

# Get param count
from thop import profile
macs, params = profile(pipe, inputs=(prompt, ))
print(f"Model has {macs} FLOPs and {params} parameters")




import torch
from diffusers import DiffusionPipeline, AutoencoderTiny

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
pipe = pipe.to("mps")

prompt = "slice of delicious New York-style berry cheesecake"
image = pipe(prompt, num_inference_steps=1).images[0]

image

import torch.nn as nn
class DiffusionWrapper(nn.Module):
    def __init__(self, pipe):
        super().__init__()
        self.model = pipe

    def forward(self, prompt):
        return self.model(prompt)
    
model = DiffusionWrapper(pipe)


# Get param count
from thop import profile
macs, params = profile(model, inputs=(prompt, ))
print(f"Model has {macs} FLOPs and {params} parameters")


StableDiffusionPipeline.__call__??

pipe.__call__
pipe.encode_prompt(prompt)

device,
        num_images_per_prompt,
        do_classifier_free_guidance,

batch_size = 1

device = model._execution_device

https://huggingface.co/docs/diffusers/main/en/api/models/autoencoder_tiny

# %% 3.2 Export data for ezkl to use

torch.onnx.export(model, prompt, "Diffusion/diffusion.onnx", export_params=True, do_constant_folding=True, input_names = ['input'],output_names = ['output'],dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})