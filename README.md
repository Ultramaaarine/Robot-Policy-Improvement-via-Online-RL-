# Robot-Policy-Improvement-via-Online-RL
This project presents a reinforcement-learning-enhanced diffusion policy framework for robotic manipulation. A diffusion policy is first trained on offline demonstrations and then improved using critic-guided online interaction.
This repo is a user guild for my project. 
This repo is tested on a 24.04 Unbuntu machine
# Quick Start

1. Clone the repository:

```bash
mkdir -p SAC_Diffusion
https://github.com/Ultramaaarine/Robot-Policy-Improvement-via-Online-RL-.git
cd SAC_Diffusion
```

2. Download the CALVIN dataset

    Please note that the dataset requires significant storage space and may take a considerable amount of time to download. For more information, visit the [CALVIN repository](https://github.com/mees/calvin).

   You don't need to perfrom every steps in [CALVIN repository] just download the dataset to dataset folder.

```bash
cd dataset
git clone --recurse-submodules https://github.com/mees/calvin.git
cd dataset
sh download_data.sh D | ABC | ABCD | debug
```
3. create conda environment
```bash
conda create -n sacdiff_venv python = 3.12
```
4. Activate conda environment
```bash
conda activate sacdiff_venv
```
5. Install pytorch

  Attention: if you are in root path "SAC_Diffusion",skip command "cd ..". Make Sure you are in root path!!!
```bash
cd ..
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```
6. Install other packages
```bash
pip install -r requirements.txt
```
7. Run training script

  run DDPM training
```bash
python sac_diffusion/workspaces/ddpm_critic_training.py
```

  run DDIM training
```bash
python sac_diffusion/workspaces/ddim_critic_training.py
```

