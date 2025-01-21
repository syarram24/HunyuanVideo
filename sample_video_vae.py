import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchmetrics.functional as tm_F
import torch
import os
import torch
import numpy as np

def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    dataset_path = "/mnt/localssd/ffhq-dataset/images1024x1024/00000"
    output_path = "/mnt/localssd/psnr_heatmaps/Flux_ffhq_images_256"
    os.makedirs(output_path, exist_ok=True)

    # Load VAE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    print(hunyuan_video_sampler.vae.config)
    vae = hunyuan_video_sampler.vae
    
    # Get the updated args
    args = hunyuan_video_sampler.args


    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize for VAE model compatibility
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Process first 100 images
    image_files = [f for f in sorted(os.listdir(dataset_path)) if f.endswith(('.jpeg', '.png'))][:1000]

    # Add these before the main processing loop
    psnr_values = []
    successful_images = 0

    for idx, image_file in enumerate(image_files):
        try:
            # Load image
            image_path = os.path.join(dataset_path, image_file)
            image = Image.open(image_path).convert("RGB")
            original_image = transform(image).unsqueeze(0).to(device)

            # VAE reconstruction
            vae.eval()
            with torch.no_grad():
                #latent = vae.encode(original_image).latent_dist.sample()
                print(f'original_image min {original_image.min()} max {original_image.max()}')
                latent = vae.encode(original_image*2.0 - 1.0).latent_dist.sample()
                print(f'latent {latent.shape}')
                latent_scaled = latent #* 0.18215
                reconstructed_image = vae.decode(latent_scaled, return_dict=False)[0]

                print(f'reconstructed_image min {reconstructed_image.min()} max {reconstructed_image.max()}')
                
                # Compute PSNR using torchvision
                torchmetrics_psnr = tm_F.peak_signal_noise_ratio(
                    (original_image), 
                    (reconstructed_image + 1.0)/2.0,
                    data_range=1.0
                )
                print(f'torchvision psnr: {torchmetrics_psnr}')
                
                # Add this line to store PSNR value
                psnr_values.append(torchmetrics_psnr.item())
                successful_images += 1
            # Post-process images
            original_image_np = original_image.squeeze().cpu().numpy().transpose(1, 2, 0)
            original_image_np = np.clip(original_image_np, 0, 1)
            reconstructed_image_np = ((reconstructed_image + 1.0)/2.0).squeeze().cpu().numpy().transpose(1, 2, 0)
            reconstructed_image_np = np.clip(reconstructed_image_np, 0, 1)

            # Compute pixel-wise MSE
            mse = np.mean((original_image_np - reconstructed_image_np) ** 2, axis=2)

            # Compute pixel-wise PSNR (avoid log(0) with epsilon)
            epsilon = 1e-10
            psnr_heatmap = 10 * np.log10(1.0 / (mse + epsilon))
            
            print(f'psnr_heatmap {psnr_heatmap.mean()}')

            # Normalize PSNR heatmap for visualization
            psnr_heatmap_normalized = np.clip(psnr_heatmap / 40.0, 0, 1)

            combined_output_path = os.path.join(output_path, f"combined_{idx:03d}.png")

            # Convert the heatmap to an image
            plt.figure(figsize=(4, 4))
            plt.imshow(psnr_heatmap_normalized, cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(label="Normalized PSNR")
            plt.axis("off")
            heatmap_path = os.path.join(output_path, f"temp_heatmap_{idx:03d}.png")
            plt.savefig(heatmap_path)
            plt.close()
            heatmap_image = Image.open(heatmap_path)

            # Convert numpy arrays to PIL images
            original_pil = Image.fromarray((original_image_np * 255).astype(np.uint8))
            reconstructed_pil = Image.fromarray((reconstructed_image_np * 255).astype(np.uint8))

            # Ensure all images are the same size
            heatmap_image = heatmap_image.resize(original_pil.size)

            # Combine images horizontally
            combined_width = original_pil.width + reconstructed_pil.width + heatmap_image.width
            combined_height = original_pil.height
            combined_image = Image.new("RGB", (combined_width, combined_height))

            # Paste images side-by-side
            combined_image.paste(original_pil, (0, 0))
            combined_image.paste(reconstructed_pil, (original_pil.width, 0))
            combined_image.paste(heatmap_image, (original_pil.width + reconstructed_pil.width, 0))

            # Save combined image
            combined_image.save(combined_output_path)

            # Remove temporary heatmap file
            os.remove(heatmap_path)

            print(f"Processed {idx + 1}/{len(image_files)}: {image_file} and saved combined image")


        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        break
    # Add these after the processing loop
    if successful_images > 0:
        average_psnr = sum(psnr_values) / successful_images
        print(f"\nProcessing complete!")
        print(f"Successfully processed {successful_images} images")
        print(f"Average PSNR: {average_psnr:.2f}")
        print(f"Min PSNR: {min(psnr_values):.2f}")
        print(f"Max PSNR: {max(psnr_values):.2f}")
    

    

if __name__ == "__main__":
    main()
