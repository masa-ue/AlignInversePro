import argparse
import os.path
# import hydra
import random
import string
import datetime
from datetime import date
import logging
import pickle
import wandb
import sys
sys.path.append('~/private/seqft/multiflow')
sys.path.append('../')
# print(sys.path)
from protein_oracle.utils import str2bool
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from multiflow.models import utils as mu
from biotite.sequence.io import fasta
import shutil
from protein_oracle.data_utils import ALPHABET
import pandas as pd
import numpy as np
import torch

import json, time, os, sys, glob
import shutil
import warnings
# import numpy as np
# from torch import optim
from torch.utils.data import DataLoader
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
from concurrent.futures import ProcessPoolExecutor    
from protein_oracle.utils import set_seed
from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
from protein_oracle.model_utils import ProteinMPNNOracle, get_std_opt
from fmif.model_utils import ProteinMPNNFMIF
from fmif.fm_utils import Interpolant, get_likelihood, set_diversity
# import fmif.model_utils as mu
# from fmif.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, set_seed
# from fmif.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNNFMIF
# from fmif.fm_utils import Interpolant, fm_model_step
from tqdm import tqdm
from multiflow.models import folding_model
from types import SimpleNamespace



def cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss, save_path, namename, eval=False):
    with torch.no_grad():
        gen_foldtrue_mpnn_results_list = []
        gen_true_mpnn_results_list = []
        foldtrue_true_mpnn_results_list = []
        run_name = save_path.split('/')
        if run_name[-1] == '':
            run_name = run_name[-2]
        else:
            run_name = run_name[-1]
        
        if eval:
            sc_output_dir = os.path.join('sc_tmp', f"{run_name}_{namename}_sc_output", batch["protein_name"][0][:-4])
            the_pdb_path = os.path.join(pdb_path, batch['WT_name'][0])
            # fold the ground truth sequence
            os.makedirs(os.path.join(sc_output_dir, 'true_seqs'), exist_ok=True)
            true_fasta = fasta.FastaFile()
            true_fasta['true_seq_1'] = "".join([ALPHABET[x] for _ix, x in enumerate(S[0]) if mask_for_loss[0][_ix] == 1])
            true_fasta_path = os.path.join(sc_output_dir, 'true_seqs', 'true.fa')
            true_fasta.write(true_fasta_path)
            true_folded_dir = os.path.join(sc_output_dir, 'true_folded')
            if os.path.exists(true_folded_dir):
                shutil.rmtree(true_folded_dir)
            os.makedirs(true_folded_dir, exist_ok=False)
            true_folded_output = the_folding_model.fold_fasta(true_fasta_path, true_folded_dir)
        
        for _it, ssp in enumerate(S_sp):
            if not eval:
                sc_output_dir = os.path.join('sc_tmp', run_name, 'sc_output', batch["protein_name"][_it][:-4])
            sc_output_dir = os.path.join('sc_tmp', f"{run_name}_{namename}_sc_output", batch["protein_name"][0][:-4],  str(_it))
            os.makedirs(sc_output_dir, exist_ok=True)
            os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)
            codesign_fasta = fasta.FastaFile()
            codesign_fasta['codesign_seq_1'] = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
            codesign_fasta.write(codesign_fasta_path)

            folded_dir = os.path.join(sc_output_dir, 'folded')
            if os.path.exists(folded_dir):
                shutil.rmtree(folded_dir)
            os.makedirs(folded_dir, exist_ok=False)

            folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)
            
            if not eval:
                # fold the ground truth sequence
                os.makedirs(os.path.join(sc_output_dir, 'true_seqs'), exist_ok=True)
                true_fasta = fasta.FastaFile()
                true_fasta['true_seq_1'] = "".join([ALPHABET[x] for _ix, x in enumerate(S[_it]) if mask_for_loss[_it][_ix] == 1])
                true_fasta_path = os.path.join(sc_output_dir, 'true_seqs', 'true.fa')
                true_fasta.write(true_fasta_path)
                true_folded_dir = os.path.join(sc_output_dir, 'true_folded')
                if os.path.exists(true_folded_dir):
                    shutil.rmtree(true_folded_dir)
                os.makedirs(true_folded_dir, exist_ok=False)
                true_folded_output = the_folding_model.fold_fasta(true_fasta_path, true_folded_dir)

            if not eval:
                the_pdb_path = os.path.join(pdb_path, batch['WT_name'][_it])
            # folded generated with folded true
            gen_foldtrue_mpnn_results = mu.process_folded_outputs(os.path.join(true_folded_dir, 'folded_true_seq_1.pdb'), folded_output)
            # folded generated with pdb true
            gen_true_mpnn_results = mu.process_folded_outputs(the_pdb_path, folded_output)
            # folded true with pdb true
            foldtrue_true_mpnn_results = mu.process_folded_outputs(the_pdb_path, true_folded_output)

            if eval:
                seq_revovery = (S_sp[_it] == S[0]).float().mean().item()
            else:
                seq_revovery = (S_sp[_it] == S[_it]).float().mean().item()
            gen_foldtrue_mpnn_results['seq_recovery'] = seq_revovery
            gen_true_mpnn_results['seq_recovery'] = seq_revovery
            gen_foldtrue_mpnn_results_list.append(gen_foldtrue_mpnn_results)
            gen_true_mpnn_results_list.append(gen_true_mpnn_results)
            foldtrue_true_mpnn_results_list.append(foldtrue_true_mpnn_results)

    return gen_foldtrue_mpnn_results_list, gen_true_mpnn_results_list, foldtrue_true_mpnn_results_list


argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argparser.add_argument("--path_for_pdbs", type=str, default="../datasets/AlphaFold_model_PDBs", help="path for loading pdb files") 
argparser.add_argument("--path_for_outputs", type=str, default="../datasets", help="path for loading pdb files") 
argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for") # 200
argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
# argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
# argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
argparser.add_argument("--batch_size", type=int, default=5, help="number of sequences for one batch") # 128
# argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
argparser.add_argument("--num_neighbors", type=int, default=50, help="number of neighbors for the sparse graph")   # 48
argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout") # TODO
argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")   # TODO
# argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
# argparser.add_argument("--debug", type=str2bool, default=False, help="minimal data loading for debugging")
argparser.add_argument("--gradient_norm", type=float, default=1.0, help="clip gradient norm, set to negative to omit clipping")
argparser.add_argument("--mixed_precision", type=str2bool, default=True, help="train with mixed precision")
# argparser.add_argument("--initialize_with_pretrain", type=str2bool, default=False, help="initialize with FMIF weights")
argparser.add_argument("--train_using_diff", type=str2bool, default=False, help="training using difference in dG")
argparser.add_argument("--predict_ddg", type=str2bool, default=True, help="model directly predicts ddG")
argparser.add_argument("--search_schedule", type=str, default="all", help="exponential, linear, or all")
argparser.add_argument("--drop_schedule", type=str, default="", help="exponential, linear...")
argparser.add_argument("--drop_oversample_rate", type=int, default=2)
# TODO
argparser.add_argument("--wandb_name", type=str, default="debug", help="folder name for saving the results")
argparser.add_argument("--lr", type=float, default=1e-4)
argparser.add_argument("--wd", type=float, default=1e-4)

argparser.add_argument("--min_t", type=float, default=1e-2)
argparser.add_argument("--schedule", type=str, default='linear') # other schedule is not implemented
argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
argparser.add_argument("--temp", type=float, default=0.1)
argparser.add_argument("--noise", type=float, default=1.0) # 20.0
argparser.add_argument("--interpolant_type", type=str, default='masking')
argparser.add_argument("--do_purity", type=str2bool, default=False) # True
argparser.add_argument("--num_timesteps", type=int, default=50) # 500
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--eval_every_n_epochs", type=int, default=1)
argparser.add_argument("--num_samples_per_eval", type=int, default=10)

argparser.add_argument("--accum_steps", type=int, default=1)
argparser.add_argument("--truncate_steps", type=int, default=10)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument("--alpha", type=float, default=0.001)
argparser.add_argument("--gumbel_softmax_temp", type=float, default=0.5)

argparser.add_argument("--decoding", type=str, default='original')
argparser.add_argument("--reward_name", type=str, default = 'stability')
argparser.add_argument("--dps_scale", type=float, default=0.0)
argparser.add_argument("--tds_alpha", type=float, default=0.0)
argparser.add_argument("--batchsize", type=int, default=5)
argparser.add_argument("--repeatnum", type=int, default=5)
argparser.add_argument("--depth", type=int, default=3)

args = argparser.parse_args()
set_seed(args.seed, torch.cuda.is_available())

pdb_path = '../datasets/AlphaFold_model_PDBs'
max_len = 75  # Define the maximum length of proteins
dataset = ProteinStructureDataset(pdb_path, max_len) # max_len set to 75 (sequences range from 31 to 74)
loader = DataLoader(dataset, batch_size=1000, shuffle=False)

folder_name = 'log'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

folder_name = 'sc_tmp'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

folder_name = '../.cache'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")


# make a dict of pdb filename: index
for batch in loader:
    pdb_structures = batch[0]
    pdb_filenames = batch[1]
    pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
    break

dpo_dict_path = '../datasets/processed_data'
dpo_train_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict_wt.pkl'), 'rb'))
dpo_train_dataset = ProteinDPODataset(dpo_train_dict, pdb_idx_dict, pdb_structures)
loader_train = DataLoader(dpo_train_dataset, batch_size=1, shuffle=False)
dpo_valid_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict_wt.pkl'), 'rb'))
dpo_valid_dataset = ProteinDPODataset(dpo_valid_dict, pdb_idx_dict, pdb_structures)
loader_valid = DataLoader(dpo_valid_dataset, batch_size=1, shuffle=False)
dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
#keys_to_include = {'XX|run7_0974_0003.pdb', '5JRT.pdb'}
#dpo_test_dict = {k: dpo_test_dict[k] for k in keys_to_include}
dpo_test_dataset = ProteinDPODataset(dpo_test_dict, pdb_idx_dict, pdb_structures)
loader_test = DataLoader(dpo_test_dataset, batch_size=1, shuffle=False)

'''
Now, I choose 50 protein backbones for conditioning
'''
dpo_train50 = {k: dpo_train_dict[k] for k in list(dpo_train_dict)[:50]}
dpo_train_dataset50 = ProteinDPODataset(dpo_train50, pdb_idx_dict, pdb_structures)
loader_train50 = DataLoader(dpo_train_dataset50, batch_size=1, shuffle=False)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


# Load Pre-trained model 

old_fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                    edge_features=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_encoder_layers,
                    k_neighbors=args.num_neighbors,
                    dropout=args.dropout,
                    # augment_eps=args.backbone_noise
                    )
old_fmif_model.to(device)
old_fmif_model.load_state_dict(torch.load('../datasets/artifacts/pretrained_model:v0/epoch300_step447702.pt')['model_state_dict'])
old_fmif_model.finetune_init()

noise_interpolant = Interpolant(args)
noise_interpolant.set_device(device)

# Load reward model 

reward_model = ProteinMPNNOracle(node_features=args.hidden_dim,
                    edge_features=args.hidden_dim,
                    hidden_dim=args.hidden_dim,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_encoder_layers,
                    k_neighbors=args.num_neighbors,
                    dropout=args.dropout,
                    # augment_eps=args.backbone_noise
                    )
reward_model.to(device)
reward_model.load_state_dict(torch.load("../datasets/artifacts/reward_model:v2/epoch5_step17135.pt")['model_state_dict'])
reward_model.finetune_init()
# for param in reward_model.parameters():
#     param.requires_grad = False
reward_model.eval()


folding_cfg = {
    'seq_per_sample': 1,
    'folding_model': 'esmf',
    'own_device': False,
    'pmpnn_path': './ProteinMPNN/',
    'pt_hub_dir': '../.cache/',
    #'colabfold_path': '/data/wangc239/colabfold-conda/bin/colabfold_batch' # for AF2
}
folding_cfg = SimpleNamespace(**folding_cfg)
the_folding_model = folding_model.FoldingModel(folding_cfg)
save_path = os.path.join(args.path_for_outputs, f'eval_{args.wandb_name}')

# result save path
folder_path = f"log/{args.wandb_name}"
if os.path.exists(folder_path):
    warnings.warn(f"Result folder already exists: {folder_path}", RuntimeWarning)
os.makedirs(folder_path, exist_ok=True)
args_dict = vars(args)
with open(os.path.join(folder_path, "args.json"), "w") as f:
    json.dump(args_dict, f, indent=4)

model_to_test_list = [old_fmif_model]
for testing_model in model_to_test_list:

    testing_model.eval()
    print(f'Testing Model... Sampling {args.decoding}')
    repeat_num= args.repeatnum
    batchsize = args.batchsize
    valid_sp_acc, valid_sp_weights = 0., 0.
    gen_foldtrue_mpnn_results_merge = []
    gen_true_mpnn_results_merge = []
    foldtrue_true_mpnn_results_merge = []
    all_model_logl_sum, all_model_logl_mean = [], []
    rewards_eval = []
    rewards = []

    all_reward, all_scRMSD, all_recovery = [], [], []
    all_diversity = []
    all_result_dict = {}
    for _step, batch in enumerate(tqdm(loader_train50)):
        # Load Data # 
        X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
        X = X.repeat(batchsize, 1, 1, 1)
        mask = mask.repeat(batchsize, 1)
        chain_M = chain_M.repeat(batchsize, 1)
        residue_idx = residue_idx.repeat(batchsize, 1)
        chain_encoding_all = chain_encoding_all.repeat(batchsize, 1)
        mask_for_loss = mask*chain_M
 
        if args.reward_name == 'stability':
            new_reward_model  = reward_model 
        elif args.reward_name == 'LDDT':
            from fmif.reward_PLDDT import newreward_model
            new_reward_model = newreward_model(batch, the_folding_model, pdb_path, mask_for_loss, save_path) 
        elif args.reward_name == 'scRMSD':
            from fmif.reward_RMSD_new import newreward_model
            new_reward_model = newreward_model(batch, the_folding_model, pdb_path, mask_for_loss, save_path)
        elif args.reward_name ==  'stability_rosetta':
            from fmif.reward_energy import newreward_model
            new_reward_model = newreward_model(batch, the_folding_model, pdb_path, mask_for_loss, save_path)
        else:
            raise NotImplementedError()

        # Start Sampling#
        if args.decoding == 'dps':
            S_sp, _, _ = noise_interpolant.sample_controlled_DPS(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                guidance_scale=args.dps_scale, reward_model=reward_model)  
        elif args.decoding == 'SMC':
            S_sp, _, _ = noise_interpolant.sample_controlled_SMC(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                reward_model=new_reward_model,reward_name = args.reward_name, alpha=args.tds_alpha, repeats = repeat_num )
        elif args.decoding == 'SVDD': 
             S_sp, _, _ = noise_interpolant.sample_controlled_SVDD(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                reward_model=new_reward_model, reward_name = args.reward_name, repeats = repeat_num )
        elif args.decoding == 'DDBFS':
             S_sp, _, _, rate_svdd = noise_interpolant.sample_controlled_SVDD_BFS(
                 testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                 reward_model=new_reward_model, reward_name=args.reward_name, repeats=repeat_num, depth=args.depth, search_schudule=args.search_schedule, drop_schudule=args.drop_schedule, oversamplerate=args.drop_oversample_rate,
                 return_svdd_rate=True)
        elif args.decoding == 'NestedIS': 
            S_sp, _, _ = noise_interpolant.sample_controlled_NestedIS(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all,
                reward_model= new_reward_model, reward_name = args.reward_name )     
        elif args.decoding == 'original':
            S_sp, _, _ = noise_interpolant.sample(testing_model, X, mask, chain_M, residue_idx, chain_encoding_all)
        #S_sp: 30 times 47, 30 * 47 * 4* 3
        else:
            raise NotImplementedError()

        # Evaluation 
        if args.reward_name == 'stability': 
            final_reward = reward_model(X, S_sp, mask, chain_M, residue_idx, chain_encoding_all) 
            final_reward = final_reward.cpu().detach().numpy()
        elif args.reward_name == 'LDDT':
            final_reward = new_reward_model.cal_reward(S_sp)
            final_reward = final_reward.cpu().detach().numpy()
        elif args.reward_name == 'scRMSD':
            final_reward = new_reward_model.cal_rmsd_reward(S_sp)
        elif args.reward_name == 'stability_rosetta':
            final_reward = new_reward_model.calculate_energy(S_sp)
        else:
            final_reward = reward_model(X, S_sp, mask, chain_M, residue_idx, chain_encoding_all) 
            final_reward = final_reward.cpu().detach().numpy()

        rewards.append(final_reward)

        '''
        For evaluation
        ''' 
        # Calculate recovery rate 
        true_false_sp = (S_sp == S).float()
        mask_for_loss = mask*chain_M
        valid_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
        valid_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()

        hoge = true_false_sp * mask_for_loss
        recovery_r = torch.sum(hoge,1)/torch.sum(mask_for_loss,1)

        # Calculate likelihood
        model_logl_sum, model_logl_mean = get_likelihood(
            testing_model,
            (X, S_sp, mask, chain_M, residue_idx, chain_encoding_all),
            args.num_timesteps,
            device,
            noise_interpolant,
            eps=1e-5,
            mean_value=True
        )
        model_logl_sum = model_logl_sum.detach().cpu().numpy().mean().item()
        model_logl_mean = model_logl_mean.detach().cpu().numpy().mean().item()
        all_model_logl_sum.append(model_logl_sum)
        all_model_logl_mean.append(model_logl_mean)

        # Calculate RMSD 
        gen_foldtrue_mpnn_results_list, gen_true_mpnn_results_list, foldtrue_true_mpnn_results_list = cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss, save_path, args.decoding,  eval=True)
        gen_foldtrue_mpnn_results_merge.append(gen_foldtrue_mpnn_results_list)
        gen_true_mpnn_results_merge.extend(gen_true_mpnn_results_list)
        foldtrue_true_mpnn_results_merge.extend(foldtrue_true_mpnn_results_list)

        # calculate diversity
        cur_diversity = set_diversity(S_sp.detach().cpu().numpy(), mask_for_loss.detach().cpu().numpy())
        all_diversity.append(cur_diversity)

        # print("Reward", np.mean(final_reward) )
        # print("scRMSD", np.mean(np.array(pd.concat(gen_true_mpnn_results_list)['bb_rmsd'])))
        # print("recovery", np.mean(recovery_r.detach().cpu().numpy()))
        cur_reward = np.mean(final_reward).item()
        cur_scRMSD = np.mean(np.array(pd.concat(gen_true_mpnn_results_list)['bb_rmsd'])).item()
        cur_recovery = np.mean(recovery_r.detach().cpu().numpy()).item()
        all_reward.append(cur_reward)
        all_scRMSD.append(cur_scRMSD)
        all_recovery.append(cur_recovery)

        # current protein results
        cur_protein_prefix = f"{args.decoding}_{args.reward_name}_{batch['protein_name'][0][:-4]}"
        cur_result_dict = {
            f'{cur_protein_prefix}_Reward': cur_reward,
            f'{cur_protein_prefix}_scRMSD': cur_scRMSD,
            f'{cur_protein_prefix}_recovery': cur_recovery,
            f'{cur_protein_prefix}_log_likelihood_sum': model_logl_sum,
            f'{cur_protein_prefix}_log_likelihood_mean': model_logl_mean,
            f'{cur_protein_prefix}_diversity': cur_diversity,
            f'{cur_protein_prefix}_svdd_rate': rate_svdd if args.decoding == 'DDBFS' else 0,
        }
        print(cur_result_dict)
        assert not set(all_result_dict.keys()) & set(cur_result_dict.keys())
        all_result_dict.update(cur_result_dict)

        # save per protein result
        df = pd.concat(gen_true_mpnn_results_list)
        # df.to_csv("log/"+args.wandb_name+"_reward_"+ args.decoding  + "_" + args.reward_name + "_" + batch['protein_name'][0][:-4] + ".csv", index=False)
        # np.savez("log/"+args.wandb_name+"_reward_"+ args.decoding  + "_" + args.reward_name + "_" + batch['protein_name'][0][:-4] + ".npz", reward = final_reward) # Save "rewards" for generated samples
        # np.savez("log/"+args.wandb_name+"_recovery_"+ args.decoding  + "_" + args.reward_name + "_" + batch['protein_name'][0][:-4] + ".npz", reward = recovery_r.cpu().data.numpy()) # Save "recovery rates" for generated samples
        # np.savez("log/"+args.wandb_name+"_scRMSD_"+ args.decoding  + "_"+ args.reward_name + "_" + batch['protein_name'][0][:-4] + ".npz", reward = pd.concat(gen_true_mpnn_results_list)['bb_rmsd'])  # Save "scRMSDs" for generated samples
        df.to_csv(os.path.join(folder_path, f"reward_{cur_protein_prefix}.csv"), index=False)
        np.savez(os.path.join(folder_path, f"reward_{cur_protein_prefix}.npz"), reward=final_reward)
        np.savez(os.path.join(folder_path, f"recovery_{cur_protein_prefix}.npz"), reward=recovery_r.cpu().data.numpy())
        np.savez(os.path.join(folder_path, f"scRMSD_{cur_protein_prefix}.npz"), reward=pd.concat(gen_true_mpnn_results_list)['bb_rmsd'])

    # save aggregated result
    all_result_dict['final_Reward'] = sum(all_reward) / len(all_reward)
    all_result_dict['final_scRMSD'] = sum(all_scRMSD) / len(all_scRMSD)
    all_result_dict['final_recovery'] = sum(all_recovery) / len(all_recovery)
    all_result_dict['final_log_likelihood_sum'] = sum(all_model_logl_sum) / len(all_model_logl_sum)
    all_result_dict['final_log_likelihood_mean'] = sum(all_model_logl_mean) / len(all_model_logl_mean)
    all_result_dict['final_diversity'] = sum(all_diversity) / len(all_diversity)
    with open(os.path.join(folder_path, "result_dict.json"), "w") as f:
        json.dump(all_result_dict, f)
