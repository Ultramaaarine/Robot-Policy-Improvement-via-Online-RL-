# Robot-Policy-Improvement-via-Online-RL
This project presents a reinforcement-learning-enhanced diffusion policy framework for robotic manipulation. A diffusion policy is first trained on offline demonstrations and then improved using critic-guided online interaction.
This repo is a user guild for my project. 

# Quick Start

1. Clone the repository:

```bash
git clone https://github.com/你的用户名/SAC-Diffusion.git
cd SAC-Diffusion
```
2. download extracted demontrations

```bash
cd demonstrations
git
```

3. Download the CALVIN dataset(Optional)

```bash
cd dataset
git clone --recurse-submodules https://github.com/mees/calvin.git
$ export CALVIN_ROOT=$(pwd)/calvin

