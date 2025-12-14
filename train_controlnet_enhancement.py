#!/usr/bin/env python3
"""
ControlNet Training for Real Estate Photo Enhancement
Optimized for 80GB A100 - Maximum Quality Focus

Architecture:
- Base: Stable Diffusion 2.1 (768 native, scales to 1024)
- ControlNet: Image-to-image enhancement conditioning
- Control signal: Source image (low quality)
- Target: Enhanced image (professional edit)

Max Resolution on 80GB A100:
- 1024x1024 with batch_size=2, grad_accum=4
- Effective batch size: 8
"""

import os
import sys
import json
import argparse
import math
import random
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

# Diffusers imports
try:
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        UNet2DConditionModel,
        ControlNetModel,
    )
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    import accelerate
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install diffusers transformers accelerate peft")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealEstateEnhancementDataset(Dataset):
    """
    Dataset for real estate photo enhancement.

    Pairs:
    - source: Low quality / unedited photo (control signal)
    - target: Professionally enhanced photo (training target)
    """

    def __init__(
        self,
        jsonl_path: str,
        base_dir: str,
        resolution: int = 1024,
        tokenizer=None,
        prompt: str = "high quality professional real estate interior photo, HDR, well lit, sharp details"
    ):
        self.base_dir = Path(base_dir)
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.prompt = prompt

        # Load pairs
        self.pairs = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(self.pairs)} image pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load images
        src_path = self.base_dir / pair['src']
        tar_path = self.base_dir / pair['tar']

        source_image = Image.open(src_path).convert('RGB')
        target_image = Image.open(tar_path).convert('RGB')

        # Resize to training resolution
        source_image = source_image.resize(
            (self.resolution, self.resolution),
            Image.LANCZOS
        )
        target_image = target_image.resize(
            (self.resolution, self.resolution),
            Image.LANCZOS
        )

        # Random horizontal flip (apply to both)
        if random.random() > 0.5:
            source_image = source_image.transpose(Image.FLIP_LEFT_RIGHT)
            target_image = target_image.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to tensors
        source_tensor = torch.from_numpy(
            np.array(source_image).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        target_tensor = torch.from_numpy(
            np.array(target_image).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        # Normalize to [-1, 1] for diffusion
        source_tensor = source_tensor * 2.0 - 1.0
        target_tensor = target_tensor * 2.0 - 1.0

        # Tokenize prompt
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                self.prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = text_inputs.input_ids[0]
        else:
            input_ids = torch.zeros(77, dtype=torch.long)

        return {
            "source": source_tensor,  # Control signal (conditioning image)
            "target": target_tensor,  # Training target
            "input_ids": input_ids,   # Text prompt tokens
        }


def create_controlnet_from_unet(unet: UNet2DConditionModel) -> ControlNetModel:
    """
    Initialize ControlNet from UNet weights.
    This gives better starting point than random initialization.
    """
    controlnet = ControlNetModel.from_unet(unet)
    return controlnet


def train(args):
    """Main training function"""

    print("=" * 80)
    print("CONTROLNET TRAINING FOR REAL ESTATE ENHANCEMENT")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Resolution: {args.resolution}x{args.resolution}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation}")
    print(f"   Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Mixed precision: {args.mixed_precision}")
    print(f"   Output: {args.output_dir}")

    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("\n📂 Loading models...")

    # Use SD 2.1 for better quality at higher resolutions
    model_id = args.pretrained_model

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Initialize ControlNet from UNet
    print("🏗️  Initializing ControlNet from UNet weights...")
    controlnet = create_controlnet_from_unet(unet)

    # Freeze everything except ControlNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    # Count parameters
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable_params:,}")

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        print("   ✅ Gradient checkpointing enabled")

    # Enable xformers if available
    if args.enable_xformers:
        try:
            import xformers
            controlnet.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
            print("   ✅ xFormers attention enabled")
        except ImportError:
            print("   ⚠️  xFormers not available")

    # Create dataset
    print("\n📂 Loading dataset...")
    train_dataset = RealEstateEnhancementDataset(
        jsonl_path=args.train_jsonl,
        base_dir=args.data_dir,
        resolution=args.resolution,
        tokenizer=tokenizer,
        prompt=args.prompt,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Learning rate scheduler
    num_training_steps = args.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare with accelerator
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # Move frozen models to device
    vae.to(accelerator.device, dtype=torch.float32)  # VAE in fp32 for stability
    text_encoder.to(accelerator.device)
    unet.to(accelerator.device)

    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Total steps: {num_training_steps}")
    print(f"   Steps per epoch: {len(train_dataloader)}")

    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.epochs):
        controlnet.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not accelerator.is_main_process
        )

        for batch in progress_bar:
            with accelerator.accumulate(controlnet):
                # Get batch data
                source_images = batch["source"]  # Control signal
                target_images = batch["target"]  # Training target
                input_ids = batch["input_ids"]

                # Encode target images to latents
                with torch.no_grad():
                    latents = vae.encode(target_images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # Prepare control signal (source images as conditioning)
                # For image-to-image, we use the source image directly
                controlnet_image = source_images

                # Get ControlNet output
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict noise with UNet + ControlNet conditioning
                with torch.no_grad():
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                # Calculate loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            epoch_loss += loss.detach().item()
            global_step += 1

            progress_bar.set_postfix({
                "loss": f"{loss.detach().item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
            })

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_dataloader)

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs}: Avg Loss = {avg_epoch_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0 or avg_epoch_loss < best_loss:
                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
                accelerator.save_state(save_path)

                # Save ControlNet separately
                unwrapped_controlnet = accelerator.unwrap_model(controlnet)
                unwrapped_controlnet.save_pretrained(
                    os.path.join(args.output_dir, f"controlnet-{epoch+1}")
                )

                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    unwrapped_controlnet.save_pretrained(
                        os.path.join(args.output_dir, "controlnet-best")
                    )
                    print(f"   ✅ New best model saved! Loss: {best_loss:.4f}")

    # Final save
    if accelerator.is_main_process:
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        unwrapped_controlnet.save_pretrained(
            os.path.join(args.output_dir, "controlnet-final")
        )

        # Save training config
        config = {
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "epochs": args.epochs,
            "lr": args.lr,
            "best_loss": best_loss,
            "pretrained_model": args.pretrained_model,
        }
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Model saved: {args.output_dir}/controlnet-best")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train ControlNet for Real Estate Enhancement")

    # Data
    parser.add_argument("--train_jsonl", type=str, default="train.jsonl")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--prompt", type=str,
                        default="high quality professional real estate interior photo, HDR, well lit, sharp details, magazine quality")

    # Model
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-diffusion-2-1",
                        help="Base SD model (2.1 recommended for quality)")

    # Training - Optimized for 80GB A100
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Training resolution (1024 for 80GB A100)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Memory optimization
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--enable_xformers", action="store_true", default=True)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs_controlnet_1024")
    parser.add_argument("--save_every", type=int, default=10)

    # System
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
