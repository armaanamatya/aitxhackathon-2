"""
Stable Diffusion LoRA Fine-tuning for Real Estate HDR Enhancement

Uses img2img pipeline with LoRA fine-tuning to learn the
professional editing style from paired images.

Key approach:
- Use SD-inpainting or SD-img2img as backbone
- Train LoRA adapters on paired data
- Learn to "denoise" from source → target style
- Very efficient training with LoRA (~1-4M trainable params)

Requires: diffusers, peft, accelerate, transformers
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

# Hugging Face libraries
try:
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        DDPMScheduler,
        StableDiffusionImg2ImgPipeline,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model
    from accelerate import Accelerator
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    DIFFUSERS_AVAILABLE = False
    print(f"Warning: Required libraries not available: {e}")
    print("Install with: pip install diffusers transformers peft accelerate")


class HDRPairedDataset(Dataset):
    """Dataset for SD LoRA training with paired images."""

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        image_size: int = 512,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size

        # Load pairs
        with open(jsonl_path, 'r') as f:
            pairs = [json.loads(line) for line in f]

        # Split
        split_idx = int(len(pairs) * train_ratio)
        if split == "train":
            self.pairs = pairs[:split_idx]
        else:
            self.pairs = pairs[split_idx:]

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])

        print(f"Loaded {len(self.pairs)} {split} samples")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        src_path = self.data_root / pair['src']
        tar_path = self.data_root / pair['tar']

        src_img = Image.open(src_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        src_tensor = self.transform(src_img)
        tar_tensor = self.transform(tar_img)

        return {
            'source': src_tensor,
            'target': tar_tensor,
            'prompt': "a professionally edited real estate photo, bright, clean, natural lighting",
        }


class SDLoRATrainer:
    """
    Trainer for Stable Diffusion LoRA fine-tuning.

    Approach: Learn to denoise source → target latents
    """

    def __init__(
        self,
        data_root: str,
        jsonl_path: str,
        output_dir: str,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        image_size: int = 512,
        batch_size: int = 1,
        gradient_accumulation: int = 4,
        num_epochs: int = 100,
        lr: float = 1e-4,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        num_workers: int = 4,
        save_interval: int = 10,
        sample_interval: int = 5,
        resume_checkpoint: str = None,
    ):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Required libraries not available. Install diffusers, peft, etc.")

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation,
            mixed_precision="fp16",
        )

        self.device = self.accelerator.device
        print(f"Using device: {self.device}")

        # Settings
        self.model_id = model_id
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.lora_rank = lora_rank
        self.save_interval = save_interval
        self.sample_interval = sample_interval

        # Output directories
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        print("Loading datasets...")
        train_dataset = HDRPairedDataset(
            data_root, jsonl_path, image_size, "train"
        )
        val_dataset = HDRPairedDataset(
            data_root, jsonl_path, image_size, "val"
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        # Load SD components
        print(f"Loading Stable Diffusion from {model_id}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Apply LoRA to UNet
        print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})...")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()

        # Move to device
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)

        # Optimizer
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=0.01
        )

        # Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs * len(self.train_loader)
        )

        # Prepare with accelerator
        self.unet, self.optimizer, self.train_loader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.unet, self.optimizer, self.train_loader, self.lr_scheduler
            )

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]

        return prompt_embeds

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.unet.train()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            prompts = batch['prompt']

            with self.accelerator.accumulate(self.unet):
                # Encode source and target to latents
                with torch.no_grad():
                    source_latents = self.vae.encode(source).latent_dist.sample()
                    source_latents = source_latents * self.vae.config.scaling_factor

                    target_latents = self.vae.encode(target).latent_dist.sample()
                    target_latents = target_latents * self.vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(target_latents)

                # Sample timesteps
                bsz = target_latents.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=self.device
                ).long()

                # Add noise to target latents
                noisy_latents = self.noise_scheduler.add_noise(
                    target_latents, noise, timesteps
                )

                # Concatenate source latents as condition (like ControlNet)
                # We'll use a simpler approach: predict noise conditioned on source
                # by using source as initial latent + noise
                conditioned_latents = noisy_latents + 0.1 * (source_latents - target_latents)

                # Encode prompts
                prompt_embeds = self.encode_prompt(prompts[0]).repeat(bsz, 1, 1)

                # Predict noise
                noise_pred = self.unet(
                    conditioned_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                # Loss - predict the noise
                loss = F.mse_loss(noise_pred, noise)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            self.global_step += 1

            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        return {'loss': total_loss / len(self.train_loader)}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save LoRA weights."""
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # Save LoRA weights
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(self.checkpoint_dir / f"lora_epoch_{epoch}")

            if is_best:
                unwrapped_unet.save_pretrained(self.checkpoint_dir / "lora_best")

    @torch.no_grad()
    def generate_samples(self, epoch: int, num_samples: int = 4):
        """Generate sample images."""
        self.unet.eval()

        # Load pipeline with trained LoRA
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            unet=self.accelerator.unwrap_model(self.unet),
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            safety_checker=None,
        ).to(self.device)

        # Get samples from val set
        batch = next(iter(self.val_loader))
        sources = batch['source'][:num_samples]

        for i, src in enumerate(sources):
            # Convert tensor to PIL
            src_pil = transforms.ToPILImage()((src + 1) / 2)

            # Generate
            result = pipe(
                prompt="a professionally edited real estate photo, bright, clean, natural lighting",
                image=src_pil,
                strength=0.5,
                num_inference_steps=30,
            ).images[0]

            result.save(self.sample_dir / f"epoch_{epoch:04d}_sample_{i}.jpg")

    def train(self):
        """Main training loop."""
        print(f"\nStarting SD LoRA training from epoch {self.start_epoch}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print("-" * 50)

        for epoch in range(self.start_epoch, self.num_epochs):
            train_losses = self.train_epoch(epoch)

            # Logging
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_losses['loss']:.4f}")
            print(f"  LR: {self.lr_scheduler.get_last_lr()[0]:.2e}")

            # Save samples
            if (epoch + 1) % self.sample_interval == 0:
                self.generate_samples(epoch)

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)

        print("\nTraining complete!")
        self.save_checkpoint(self.num_epochs - 1, is_best=True)


def main():
    parser = argparse.ArgumentParser(description="Train SD LoRA for HDR Enhancement")

    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--jsonl_path", type=str, default="train.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs_sd_lora")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=5)

    args = parser.parse_args()

    trainer = SDLoRATrainer(
        data_root=args.data_root,
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        image_size=args.image_size,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
    )

    trainer.train()


if __name__ == "__main__":
    main()
