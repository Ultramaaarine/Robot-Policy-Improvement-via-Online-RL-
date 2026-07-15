# Robot-Policy-Improvement-via-Online-RL
This project presents a reinforcement-learning-enhanced diffusion policy framework for robotic manipulation. A diffusion policy is first trained on offline demonstrations and then improved using critic-guided online interaction.
This repo is a user guild for my project. 
This repo is tested on a 24.04 Unbuntu machine
# Quick Start

1. Clone the repository:

```bash
git clone https://github.com/你的用户名/SAC-Diffusion.git
cd SAC_Diffusion
```
2. download extracted demontrations

```bash
cd demonstrations
git
```

3. Download the CALVIN dataset (Optional)

```bash
cd dataset
git clone --recurse-submodules https://github.com/mees/calvin.git
$ export CALVIN_ROOT=$(pwd)/calvin
```
4. create conda environment
```bash
conda create -n sacdiff_venv python = 3.12
```
5. Activate conda environment
```bash
conda activate sacdiff_venv
```
6. Install pytorch

  Attention: if you are in root path "SAC_Diffusion",skip command "cd ..". Make Sure you are in root path!!!
```bash
cd ..
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```
7. Install other packages
```bash
pip install -r requirements.txt

8. Run training script
  run DDPM training
```bash
python sac_diffusion/workspaces/ddpm_critic_training.py
```
  run DDIM training
```bash
python sac_diffusion/workspaces/ddim_critic_training.py
```

