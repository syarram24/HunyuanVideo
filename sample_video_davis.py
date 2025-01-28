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
    
    dataset_path = "/mnt/localssd/DAVIS/JPEGImages/Full-Resolution" #Full-Resolution
    output_path = "/mnt/localssd/psnr_heatmaps/Hunyuan_davis_video_1080p"
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
    vae = hunyuan_video_sampler.vae #.float()
    print(vae.dtype)
    
    # Get the updated args
    args = hunyuan_video_sampler.args


    transform = transforms.Compose([
        transforms.Resize((540, 960 )),
        transforms.CenterCrop((512, 960)),  # Center crop to 1024x512 before resizing
        #transforms.Resize((960, 540)),  # Resize for VAE model compatibility
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Process first 100 images
    # Get all folders in the dataset path
    folders = [f for f in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, f))]
    
    # Get all image files from the first folder
    first_folder = os.path.join(dataset_path, folders[0])
   

    # Add these before the main processing loop
    psnr_values = []
    successful_videos = 0

    for idx, folder in enumerate(folders[:20]):
        if True:
            # Load image
            # Get all images from current folder
            folder_path = os.path.join(dataset_path, folder)
            image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])[:13]
            
            # Create list to store frames
            frames = []
            
            # Load and transform each image
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                image = Image.open(image_path).convert("RGB")
                print(f'image shape: {image.size}')
                frame = transform(image)
                print(f'frame shape: {frame.shape}')
                frames.append(frame)
            
            # Stack frames into video tensor [1, 3, T, H, W]
            original_video = torch.stack(frames, dim=1).unsqueeze(0)
            original_video = original_video.to(device)
            print(f'original_video shape: {original_video.shape}')

            # Convert input to float16 to match VAE dtype
            original_video = original_video.half()

            # VAE reconstruction
            vae.eval()
            with torch.no_grad():
                print(f'original_video {original_video.shape} min {original_video.min()} max {original_video.max()}')
                latent = vae.encode(original_video*2.0 - 1.0).latent_dist.sample()
                print(f'latent {latent.shape}')
                latent_scaled = latent #* 0.18215
                reconstructed_video = vae.decode(latent_scaled, return_dict=False)[0]

                print(f'reconstructed_video {reconstructed_video.shape} min {reconstructed_video.min()} max {reconstructed_video.max()}')
                
                # Compute PSNR using torchvision
                torchmetrics_psnr = tm_F.peak_signal_noise_ratio(
                    (original_video), 
                    (reconstructed_video + 1.0)/2.0,
                    data_range=1.0
                )
                print(f'torchvision psnr: {torchmetrics_psnr}')
                
                # Add this line to store PSNR value
                psnr_values.append(torchmetrics_psnr.item())
                successful_videos += 1

            # Post-process images
            combined_images = []
            for frame_id in range(original_video.shape[2]):
                original_image = original_video[:, :, frame_id, :, :]
                reconstructed_image = reconstructed_video[:, :, frame_id, :, :]
                original_image_np = original_image.squeeze().cpu().numpy().transpose(1, 2, 0)
                original_image_np = np.clip(original_image_np, 0, 1)
                reconstructed_image_np = ((reconstructed_image + 1.0)/2.0).squeeze().cpu().numpy().transpose(1, 2, 0)
                reconstructed_image_np = np.clip(reconstructed_image_np, 0, 1)

                # Compute pixel-wise MSE
                mse = np.mean((original_image_np - reconstructed_image_np) ** 2, axis=2)

                # Compute pixel-wise PSNR (avoid log(0) with epsilon)
                epsilon = 1e-6
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
                combined_images.append(combined_image)
            
            # Combine all frames vertically
            total_height = sum(img.height for img in combined_images)
            total_width = combined_images[0].width
            final_combined = Image.new("RGB", (total_width, total_height))
            
            y_offset = 0
            for img in combined_images:
                final_combined.paste(img, (0, y_offset))
                y_offset += img.height
                
            # Save final combined image with all frames
            final_combined.save(combined_output_path)

            # Remove temporary heatmap file
            os.remove(heatmap_path)

            print(f"Processed {idx + 1}/{len(folders)}: {folder} and saved combined image")


        # except Exception as e:
        #     print(f"Error processing {folder}: {e}")
        
    # Add these after the processing loop
    if successful_videos > 0:
        average_psnr = sum(psnr_values) / successful_videos
        print(f"\nProcessing complete!")
        print(f"Successfully processed {successful_videos} videos")
        print(f"Average PSNR: {average_psnr:.2f}")
        print(f"Min PSNR: {min(psnr_values):.2f}")
        print(f"Max PSNR: {max(psnr_values):.2f}")
    

    

if __name__ == "__main__":
    main()
