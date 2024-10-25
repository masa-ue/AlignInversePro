# include both sequence recovery and scRMSD calculation
############## THE CODE IS UNFINISHED ################

import argparse
import os.path
# import hydra
import random
import string
import datetime
from datetime import date
import logging
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
from concurrent.futures import ProcessPoolExecutor    
from fmif.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, set_seed
from fmif.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNNFMIF
from fmif.fm_utils import Interpolant, fm_model_step
from tqdm import tqdm
import wandb
from multiflow.models import folding_model
from multiflow.data.residue_constants import restypes
from multiflow.models import utils as mu
from biotite.sequence.io import fasta
from types import SimpleNamespace


def main(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    assert torch.cuda.is_available(), "CUDA is not available"

    # load PDB data
    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 4}
    
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)
    print(len(train), len(valid), len(test)) # 23349 1464 1539

    # train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    # train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    test_set = PDB_dataset(list(test.keys()), loader_pdb, test, params)
    test_loader = torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    fmif_model.to(device)
    fmif_model.load_state_dict(torch.load('/data/wangc239/pmpnn/outputs/jnvGDJFmCj_20240710_145313/model_weights/epoch300_step447702.pt')['model_state_dict'])
    fmif_model.eval()

    noise_interpolant = Interpolant(args)
    noise_interpolant.set_device(device)

    folding_cfg = {
        'seq_per_sample': 1,
        'folding_model': 'esmf',
        'own_device': False,
        'pmpnn_path': './ProteinMPNN/',
        'pt_hub_dir': './.cache/torch/',
        'colabfold_path': '/data/wangc239/colabfold-conda/bin/colabfold_batch' # for AF2
    }
    folding_cfg = SimpleNamespace(**folding_cfg)
    the_folding_model = folding_model.FoldingModel(folding_cfg)


    with ProcessPoolExecutor(max_workers=12) as executor:
        # import time
        # t0 = time.time()
        # q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        pq = queue.Queue(maxsize=3)
        for i in range(3):
            # q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            pq.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        # pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()
        pdb_dict_test = pq.get().result()
        # print(len(pdb_dict_train), len(pdb_dict_valid))
        # print(len(pdb_dict_train[0]['seq_chain_A']), 
        #       len(pdb_dict_train[0]['coords_chain_A']['N_chain_A']),
        #       len(pdb_dict_train[0]['coords_chain_A']['CA_chain_A']),
        #       len(pdb_dict_train[0]['coords_chain_A']['C_chain_A']),
        #       len(pdb_dict_train[0]['coords_chain_A']['O_chain_A']),
        #       len(pdb_dict_train[0]['seq']))
        # dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=args.max_protein_length)
        # print(len(dataset_train), len(dataset_valid))
        
        # loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        loader_test = StructureLoader(dataset_test, batch_size=args.batch_size)
        # print(len(loader_train), len(loader_valid), len(loader_test))
        # t1 = time.time()
        # print('time for loading data:', t1-t0)
        
    with torch.no_grad():
        print(len(loader_valid))
        valid_sp_acc, valid_sp_weights = 0., 0.
        for _, batch in tqdm(enumerate(loader_valid)):
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            S_sp, _, _ = noise_interpolant.sample(fmif_model, X, mask, chain_M, residue_idx, chain_encoding_all)
            # print(S_sp.shape)
            # print(S.shape)
            # print((S_sp == S))
            true_false_sp = (S_sp == S).float()
            mask_for_loss = mask*chain_M
            valid_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
            valid_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            for _it, ssp in enumerate(S_sp):
                # TODO: save the original structure as .pdb file
                # pdb_dict_list = get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000)
                # for i, pdb_data in enumerate(pdb_dict_list):
                #     file_path = f"output_{i+1}.pdb"  # Define the path and filename for the output file
                #     write_pdb(pdb_data, file_path)
                def write_pdb(data, file_path):
                    with open(file_path, 'w') as file:
                        for idx, chain in enumerate(data['visible_list']):
                            coords_chain = data['coords_chain_' + chain]
                            seq_chain = data['seq_chain_' + chain]
                            for res_index, residue in enumerate(seq_chain):
                                file.write(f"ATOM  {res_index + 1:5d}  N   {residue} {chain}{res_index + 1:4d}    {coords_chain['N_chain_' + chain][res_index][0]:8.3f}{coords_chain['N_chain_' + chain][res_index][1]:8.3f}{coords_chain['N_chain_' + chain][res_index][2]:8.3f}  1.00 20.00           N\n")
                                file.write(f"ATOM  {res_index + 1:5d}  CA  {residue} {chain}{res_index + 1:4d}    {coords_chain['CA_chain_' + chain][res_index][0]:8.3f}{coords_chain['CA_chain_' + chain][res_index][1]:8.3f}{coords_chain['CA_chain_' + chain][res_index][2]:8.3f}  1.00 20.00           C\n")
                                file.write(f"ATOM  {res_index + 1:5d}  C   {residue} {chain}{res_index + 1:4d}    {coords_chain['C_chain_' + chain][res_index][0]:8.3f}{coords_chain['C_chain_' + chain][res_index][1]:8.3f}{coords_chain['C_chain_' + chain][res_index][2]:8.3f}  1.00 20.00           C\n")
                                file.write(f"ATOM  {res_index + 1:5d}  O   {residue} {chain}{res_index + 1:4d}    {coords_chain['O_chain_' + chain][res_index][0]:8.3f}{coords_chain['O_chain_' + chain][res_index][1]:8.3f}{coords_chain['O_chain_' + chain][res_index][2]:8.3f}  1.00 20.00           O\n")
                        file.write("TER\nENDMDL\n")

                # save the ssp as .fasta file
                os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)
                codesign_fasta = fasta.FastaFile()
                codesign_fasta['codesign_seq_1'] = "".join([restypes[x] for x in ssp])
                codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
                codesign_fasta.write(codesign_fasta_path)

                folded_dir = os.path.join(sc_output_dir, 'folded')
                if os.path.exists(folded_dir):
                    shutil.rmtree(folded_dir)
                os.makedirs(folded_dir, exist_ok=False)
                
                folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)
                mpnn_results = mu.process_folded_outputs(pdb_path, folded_output)


        validation_sp_accuracy = valid_sp_acc / valid_sp_weights
        validation_sp_accuracy_ = np.format_float_positional(np.float32(validation_sp_accuracy), unique=False, precision=3)
        print('validation_sp_accuracy:', validation_sp_accuracy_)

        print(len(loader_test))
        test_sp_acc, test_sp_weights = 0., 0.
        for _, batch in tqdm(enumerate(loader_test)):
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            S_sp, _, _ = noise_interpolant.sample(fmif_model, X, mask, chain_M, residue_idx, chain_encoding_all)
            true_false_sp = (S_sp == S).float()
            mask_for_loss = mask*chain_M
            test_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
            test_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
        test_sp_accuracy = test_sp_acc / test_sp_weights
        test_sp_accuracy_ = np.format_float_positional(np.float32(test_sp_accuracy), unique=False, precision=3)
        print('test_sp_accuracy:', test_sp_accuracy_)
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="/data/wangc239/pmpnn/raw/pdb_2021aug02", help="path for loading training data") 
    # argparser.add_argument("--path_for_training_data", type=str, default="/data/wangc239/pmpnn/raw/pdb_2021aug02_sample", help="path for loading training data") 
    # TODO: modify the outputs logging system and wandb logging
    # argparser.add_argument("--path_for_outputs", type=str, default="/data/wangc239/pmpnn/outputs", help="path for logs and model weights")
    # argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    # argparser.add_argument("--num_epochs", type=int, default=400, help="number of epochs to train for") # 200
    # argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    # argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")   # 48
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")   # 0.2
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")

    argparser.add_argument("--min_t", type=float, default=1e-2)
    argparser.add_argument("--schedule", type=str, default='linear') # other schedule is not implemented
    argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
    argparser.add_argument("--temp", type=float, default=0.1)
    argparser.add_argument("--noise", type=float, default=1.0) # 20.0
    argparser.add_argument("--interpolant_type", type=str, default='masking')
    argparser.add_argument("--do_purity", type=bool, default=True)
    argparser.add_argument("--num_timesteps", type=int, default=500)
    # argparser.add_argument("--seed", type=int, default=0)
    # argparser.add_argument("--eval_every_n_epochs", type=int, default=20)
 
    args = argparser.parse_args()    
    main(args)
