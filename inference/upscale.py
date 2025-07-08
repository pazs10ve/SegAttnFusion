import os
import numpy as np
from PIL import Image
import torch
from RealESRGAN import RealESRGAN


def upscale(segs, model_weights_path=r'weights/RealESRGAN_x4.pth', scale=4):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=scale)
    model.load_weights(model_weights_path, download=True)
    #model.eval()
    upscaled_tensors = []
    
    with torch.no_grad():
        for tensor_image in segs:
            image = Image.fromarray((tensor_image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))        
            sr_image = model.predict(image)
            sr_tensor = torch.from_numpy(np.array(sr_image).astype('float32') / 255.0).permute(2, 0, 1)
            upscaled_tensors.append(sr_tensor)

    return torch.stack(upscaled_tensors)



"""def upscale_images(folder_path, model_weights_path, output_dir, scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=scale)
    model.load_weights(model_weights_path, download=True)
    
    upscaled_image_paths = []
    os.mkdir(output_dir)
    
    for image_path in os.listdir(folder_path):
        image = Image.open(os.path.join('segmentation_outputs', image_path)).convert('RGB')        
        sr_image = model.predict(image)        
        output_path = f"{output_dir}/{image_path.split('/')[-1].replace('.png', '_sr.png')}"
        sr_image.save(output_path)
        upscaled_image_paths.append(output_path)
    
    return upscaled_image_paths


image_paths = 'segmentation_outputs'
model_weights_path = 'weights/RealESRGAN_x4.pth'
output_dir = 'upscaled_images'

upscaled_images = upscale_images(image_paths, model_weights_path, output_dir)
print(f"Upscaled images saved at: {upscaled_images}")

"""
