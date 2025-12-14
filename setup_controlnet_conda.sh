#!/bin/bash
# One-line conda setup for ControlNet

conda env create -f controlnet_env.yml -y && \
conda activate controlnet && \
python -c "from diffusers import ControlNetModel; print('✅ Environment ready!')"
