from diffusers import OnnxStableDiffusionPipeline
height=512
width=512
num_inference_steps=15
guidance_scale=7.5
prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt="bad hands, blurry"

#provider = "DmlExecutionProvider"
provider = "CPUExecutionProvider"

pipe = OnnxStableDiffusionPipeline.from_pretrained("./stable_diffusion_onnx", provider=provider, safety_checker=None)
image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt).images[0] 
image.save("astronaut_rides_horse.png")
