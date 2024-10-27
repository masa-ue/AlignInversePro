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
from fmif.fm_utils import Interpolant, get_likelihood
# import fmif.model_utils as mu
# from fmif.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, set_seed
# from fmif.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNNFMIF
# from fmif.fm_utils import Interpolant, fm_model_step
from tqdm import tqdm
from multiflow.models import folding_model
from types import SimpleNamespace

import pyrosetta; pyrosetta.init()

from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.core.pack.task.operation import OperateOnResidueSubset
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking, IncludeCurrent
# Set up FastRelax with the desired score function
from pyrosetta import *

init()

from multiprocessing import Pool


def calcualte_energy(pdb, number):
    pose = pose_from_pdb(pdb)
    scorefxn = pyrosetta.create_score_function("ref2015_cart")
    # Create a TaskFactory to control packing
    tf = TaskFactory()
    # Restrict packing to only repack the existing rotamers (no mutations)
    tf.push_back(RestrictToRepacking())
    # Create and apply the packer
    packer = PackRotamersMover(scorefxn, tf.create_task_and_apply_taskoperations(pose))
    packer.apply(pose)
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.apply(pose)
    pose.dump_pdb(pdb)
    return scorefxn(pose)

# Function to run FastRelax in parallel
def run_parallel_fastrelax(pdbs, num_jobs=2):
    """ Run multiple FastRelax jobs in parallel. """
    # Create a multiprocessing pool with num_jobs workers
    with Pool(num_jobs) as pool:
        # Run FastRelax jobs in parallel
        results = pool.starmap(calcualte_energy, [(pdbs[i], i) for i in range(num_jobs)])
    return results


class newreward_model:
    def __init__(self,batch, the_folding_model, pdb_path, mask_for_loss, save_path):
        self.batch = batch 
        self.the_folding_model = the_folding_model
        self.pdb_path = pdb_path 
        self.mask_for_loss = mask_for_loss
        self.save_path = save_path 

    def cal_rmsd_reward(self, S_sp):
        batch = self.batch
        the_folding_model = self.the_folding_model
        pdb_path = self.pdb_path
        mask_for_loss = self.mask_for_loss
        save_path = self.save_path 
        run_name = 'pseudo'

        
        sc_output_dir = os.path.join('/data/ueharam/sc_tmp', run_name, 'pseduo_sc_output', batch["protein_name"][0][:-4])
        the_pdb_path = os.path.join(pdb_path, batch['WT_name'][0])
    
        foldtrue_true_mpnn_results_list = []
        os.makedirs(sc_output_dir, exist_ok=True)
        os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)
        for _it, ssp in enumerate(S_sp):
   
            codesign_fasta = fasta.FastaFile()
            codesign_fasta['codesign_seq_1'] = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
            codesign_fasta.write(codesign_fasta_path)

            folded_dir = os.path.join(sc_output_dir, 'folded')
            if os.path.exists(folded_dir):
                shutil.rmtree(folded_dir)
            os.makedirs(folded_dir, exist_ok=False)

            folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)
            
            # folded generated with pdb true
            foldtrue_true_mpnn_results = mu.process_folded_outputs(the_pdb_path, folded_output)
        
            foldtrue_true_mpnn_results_list.append(foldtrue_true_mpnn_results['bb_rmsd'][0])
       
        return foldtrue_true_mpnn_results_list
    
    def calculate_energy(self, S_sp):
        batch = self.batch
        the_folding_model = self.the_folding_model
        pdb_path = self.pdb_path
        mask_for_loss = self.mask_for_loss
        save_path = self.save_path 
        run_name = 'pseudo'

        
        sc_output_dir = os.path.join('/data/ueharam/sc_tmp', run_name, 'pseduo_sc_output', batch["protein_name"][0][:-4])
        the_pdb_path = os.path.join(pdb_path, batch['WT_name'][0])
    
        energy_list = []
        os.makedirs(sc_output_dir, exist_ok=True)

    
        folded_path_list = [ ]

        for _it, ssp in enumerate(S_sp):
            os.makedirs(os.path.join(sc_output_dir, str(_it)), exist_ok=True)
            os.makedirs(os.path.join(sc_output_dir, str(_it), 'fmif_seqs'), exist_ok=True)

            codesign_fasta = fasta.FastaFile()
            codesign_fasta['codesign_seq_1'] = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            codesign_fasta_path = os.path.join(sc_output_dir, str(_it), 'fmif_seqs', 'codesign.fa')
            codesign_fasta.write(codesign_fasta_path)

            folded_dir = os.path.join(sc_output_dir,  str(_it),'folded')
            if os.path.exists(folded_dir):
                shutil.rmtree(folded_dir)
            os.makedirs(folded_dir, exist_ok=False)
            folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)
            folded_path_list.append(folded_output['folded_path'][0])

        energy_list = run_parallel_fastrelax(folded_path_list, len(S_sp))
 
        return energy_list 