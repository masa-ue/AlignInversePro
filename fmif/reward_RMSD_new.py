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
from openfold.utils.superimposition import superimpose

from multiprocessing import Pool
from multiflow.data import utils as du



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

        
        sc_output_dir = os.path.join('sc_tmp', run_name, 'pseduo_sc_output', batch["protein_name"][0][:-4])
        the_pdb_path = os.path.join(pdb_path, batch['WT_name'][0])
    
        foldtrue_true_mpnn_results_list = []
        os.makedirs(sc_output_dir, exist_ok=True)
        os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)

        sequences = ["".join([ALPHABET[x] for _ix, x in enumerate(ssp)]) for _it, ssp in enumerate(S_sp) ] 
        fold_outputs = the_folding_model.esmf_model_parallel_sturcture(sequences)
        #foldtrue_true_mpnn_results = mu.process_folded_outputs_modify(the_pdb_path)

        all_atoms = fold_outputs['atom37_atom_exists']
        batchsize = all_atoms.shape[0]
        length = all_atoms.shape[1]

        sample_feats = du.parse_pdb_feats('sample', the_pdb_path)
        sample_bb_pos = sample_feats['atom_positions'][:, :3].reshape(-1, 3) #141 times 3 

        def _calc_bb_rmsd(mask, sample_bb_pos, folded_bb_pos):
            aligned_rmsd = superimpose(
                torch.tensor(sample_bb_pos)[None],
                torch.tensor(folded_bb_pos[None]),
                mask[:, None].repeat(1, 3).reshape(-1)
            )
            return aligned_rmsd[1].item()
         
        bb_rmsd_list = []
        for _it in range(batchsize):
            res_mask = torch.ones(length)
            folded_feats  = du.parse_pdb_feats('folded', "log/generated_proteins" + str(_it) + ".pdb")
            folded_bb_pos = folded_feats['atom_positions'][:, :3].reshape(-1, 3) #141 times 3 
            bb_rmsd = _calc_bb_rmsd(res_mask, sample_bb_pos, folded_bb_pos)
            bb_rmsd_list.append(bb_rmsd)
        return bb_rmsd_list
    
   