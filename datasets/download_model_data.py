import wandb
import os
run = wandb.init()


artifact = run.use_artifact("fderc_diffusion/Inverse_PF/AlphaFoldPDB:v0")
dir = artifact.download()
os.system('tar -xvzf artifacts/AlphaFoldPDB:v0/AlphaFoldPDB.tar.gz')

artifact = run.use_artifact("fderc_diffusion/Inverse_PF/processed_data:v2")
dir = artifact.download()
os.system('tar -xvzf artifacts/processed_data:v2/processed_data.tar.gz')

artifact = run.use_artifact("fderc_diffusion/Inverse_PF/pretrained_model:v0")
dir = artifact.download()

artifact = run.use_artifact("fderc_diffusion/Inverse_PF/reward_model:v2")
dir = artifact.download()

wandb.finish()
