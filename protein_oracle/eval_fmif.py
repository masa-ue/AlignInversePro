import argparse
import os.path
import random
import string
import datetime
import pickle
from protein_oracle.utils import str2bool
from multiflow.models import utils as mu
from biotite.sequence.io import fasta
import shutil
from protein_oracle.data_utils import ALPHABET
import pandas as pd
import numpy as np


def cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss):
    gen_foldtrue_mpnn_results_list = []
    gen_true_mpnn_results_list = []
    foldtrue_true_mpnn_results_list = []
    for _it, ssp in enumerate(S_sp):
        sc_output_dir = os.path.join('/data/ueharam/sc_tmp', 'sc_output', batch["protein_name"][_it][:-4])
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

        the_pdb_path = os.path.join(pdb_path, batch['WT_name'][_it])
        # folded generated with folded true
        gen_foldtrue_mpnn_results = mu.process_folded_outputs(os.path.join(true_folded_dir, 'folded_true_seq_1.pdb'), folded_output)
        # folded generated with pdb true
        gen_true_mpnn_results = mu.process_folded_outputs(the_pdb_path, folded_output)
        # folded true with pdb true
        foldtrue_true_mpnn_results = mu.process_folded_outputs(the_pdb_path, true_folded_output)

        seq_revovery = (S_sp[_it] == S[_it]).float().mean().item()
        gen_foldtrue_mpnn_results['seq_recovery'] = seq_revovery
        gen_true_mpnn_results['seq_recovery'] = seq_revovery
        gen_foldtrue_mpnn_results_list.append(gen_foldtrue_mpnn_results)
        gen_true_mpnn_results_list.append(gen_true_mpnn_results)
        foldtrue_true_mpnn_results_list.append(foldtrue_true_mpnn_results)

    return gen_foldtrue_mpnn_results_list, gen_true_mpnn_results_list, foldtrue_true_mpnn_results_list


def parse_df(results_df):
    avg_rmsd = results_df['bb_rmsd'].mean()
    success_rate = results_df['bb_rmsd'].apply(lambda x: 1 if x < 2 else 0).mean()
    return np.format_float_positional(avg_rmsd, unique=False, precision=3), np.format_float_positional(success_rate, unique=False, precision=3)


def main(args):
    import os
    import torch
    from torch.utils.data import DataLoader
    import os.path
    from concurrent.futures import ProcessPoolExecutor    
    from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize, ALPHABET
    from fmif.model_utils import ProteinMPNNFMIF
    from fmif.fm_utils import Interpolant
    from tqdm import tqdm
    from multiflow.models import folding_model
    from types import SimpleNamespace

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    pdb_path = args.path_for_pdbs
    max_len = 75  # Define the maximum length of proteins
    dataset = ProteinStructureDataset(pdb_path, max_len) # max_len set to 75 (sequences range from 31 to 74)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    # make a dict of pdb filename: index
    for batch in loader:
        pdb_structures = batch[0]
        pdb_filenames = batch[1]
        pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
        break

    dpo_dict_path = args.path_for_dpo_dicts
    dpo_train_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict_wt.pkl'), 'rb'))
    dpo_train_dataset = ProteinDPODataset(dpo_train_dict, pdb_idx_dict, pdb_structures)
    loader_train = DataLoader(dpo_train_dataset, batch_size=args.batch_size, shuffle=False)
    dpo_valid_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict_wt.pkl'), 'rb'))
    dpo_valid_dataset = ProteinDPODataset(dpo_valid_dict, pdb_idx_dict, pdb_structures)
    loader_valid = DataLoader(dpo_valid_dataset, batch_size=args.batch_size, shuffle=False)
    dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
    dpo_test_dataset = ProteinDPODataset(dpo_test_dict, pdb_idx_dict, pdb_structures)
    loader_test = DataLoader(dpo_test_dataset, batch_size=args.batch_size, shuffle=False)

    fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        # augment_eps=args.backbone_noise
                        )
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

    # def read_fasta(file_path):
    #     with open(file_path, 'r') as file:
    #         # Dictionary to store the sequences
    #         sequences = {}
            
    #         # Temporary variables to hold the current sequence ID and sequence
    #         seq_id = None
    #         seq = []
            
    #         for line in file:
    #             line = line.strip()
    #             if line.startswith(">"):  # Description line
    #                 if seq_id is not None:
    #                     sequences[seq_id] = ''.join(seq)
    #                 seq_id = line[1:]  # Remove the '>' character
    #                 seq = []
    #             else:
    #                 seq.append(line)
            
    #         # Add the last sequence
    #         if seq_id is not None:
    #             sequences[seq_id] = ''.join(seq)
        
    #     return sequences

    # debug_path = '/data/wangc239/multiflow/inference_outputs/multiflow/weights/last/inverse_folding/run_2024-06-28_16-21-49/length_66/7T5U'
    # debug_dataset = ProteinStructureDataset(debug_path, 200) # max_len set to 75 (sequences range from 31 to 74)
    # debug_loader = DataLoader(debug_dataset, batch_size=1000, shuffle=False)
    # for b in debug_loader:
    #     debug_structures = b[0]
    #     debug_filenames = b[1]
    #     debug_pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
    #     break
    # debug_seq = read_fasta('/data/wangc239/multiflow/inference_outputs/multiflow/weights/last/inverse_folding/run_2024-06-28_16-21-49/length_66/7T5U/true_aa.fa')
    # print(debug_seq)
    # batch = {
    #     'structure': debug_structures[0].unsqueeze(0),
    #     'aa_seq': [debug_seq['seq_1']],
    #     'aa_seq_wt': [debug_seq['seq_1']],
    # }
    # X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
    # S_sp, _, _ = noise_interpolant.sample(fmif_model, X, mask, chain_M, residue_idx, chain_encoding_all)
    # mask_for_loss = mask*chain_M
    # # print(S_sp)
    # print('sequence recovery: ', (S_sp[0] == S[0]).float().mean().item())

    # sc_output_dir = os.path.join('/data/ueharam/sc_tmp', 'sc_output', 'debug')
    # os.makedirs(sc_output_dir, exist_ok=True)
    # os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)
    # codesign_fasta = fasta.FastaFile()
    # codesign_fasta['codesign_seq_1'] = "".join([ALPHABET[x] for _ix, x in enumerate(S_sp[0]) if mask_for_loss[0][_ix] == 1])
    # print(codesign_fasta)
    # codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
    # codesign_fasta.write(codesign_fasta_path)

    # folded_dir = os.path.join(sc_output_dir, 'folded')
    # if os.path.exists(folded_dir):
    #     shutil.rmtree(folded_dir)
    # os.makedirs(folded_dir, exist_ok=False)

    # folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)
    # # print(folded_output)
    # # read in the .pdb file in the folded_dir
    # # folded_output = glob.glob(folded_dir + '/*.pdb')

    # # fold the ground truth sequence
    # os.makedirs(os.path.join(sc_output_dir, 'true_seqs'), exist_ok=True)
    # true_fasta = fasta.FastaFile()
    # true_fasta['true_seq_1'] = "".join([ALPHABET[x] for _ix, x in enumerate(S[0]) if mask_for_loss[0][_ix] == 1])
    # print(true_fasta)
    # true_fasta_path = os.path.join(sc_output_dir, 'true_seqs', 'true.fa')
    # true_fasta.write(true_fasta_path)
    # true_folded_dir = os.path.join(sc_output_dir, 'true_folded')
    # if os.path.exists(true_folded_dir):
    #     shutil.rmtree(true_folded_dir)
    # os.makedirs(true_folded_dir, exist_ok=False)
    # true_folded_output = the_folding_model.fold_fasta(true_fasta_path, true_folded_dir)

    # the_pdb_path = os.path.join(debug_path, 'sample.pdb')
    # # mpnn_results = mu.process_folded_outputs(os.path.join(true_folded_dir, 'folded_true_seq_1.pdb'), folded_output)
    # mpnn_results = mu.process_folded_outputs(the_pdb_path, folded_output)
    # print(mpnn_results)

    # import sys
    # sys.exit(0)
        
    with ProcessPoolExecutor(max_workers=12) as executor:

        with torch.no_grad():
            print(len(loader_train))
            train_sp_acc, train_sp_weights = 0., 0.
            gen_foldtrue_mpnn_results_merge = []
            gen_true_mpnn_results_merge = []
            foldtrue_true_mpnn_results_merge = []
            for _, batch in tqdm(enumerate(loader_train)):
                X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
                #X. Shape [128, 72, 4, 3] # S.shape [128, 72]  # mask.shape [128,72]
                S_sp, _, _ = noise_interpolant.sample(fmif_model, X, mask, chain_M, residue_idx, chain_encoding_all)
                true_false_sp = (S_sp == S).float()
                mask_for_loss = mask*chain_M
                train_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
                train_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                # for _it, ssp in enumerate(S_sp):
                #     # if batch["protein_name"][_it][:-4] != '5KPH':
                #     #     continue
                #     # save the ssp as .fasta file
                #     sc_output_dir = os.path.join('/data/ueharam/sc_tmp', 'sc_output', batch["protein_name"][_it][:-4])
                #     os.makedirs(sc_output_dir, exist_ok=True)
                #     os.makedirs(os.path.join(sc_output_dir, 'fmif_seqs'), exist_ok=True)
                #     codesign_fasta = fasta.FastaFile()
                #     codesign_fasta['codesign_seq_1'] = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
                #     codesign_fasta_path = os.path.join(sc_output_dir, 'fmif_seqs', 'codesign.fa')
                #     codesign_fasta.write(codesign_fasta_path)

                #     folded_dir = os.path.join(sc_output_dir, 'folded')
                #     if os.path.exists(folded_dir):
                #         shutil.rmtree(folded_dir)
                #     os.makedirs(folded_dir, exist_ok=False)

                #     folded_output = the_folding_model.fold_fasta(codesign_fasta_path, folded_dir)

                #     # fold the ground truth sequence
                #     os.makedirs(os.path.join(sc_output_dir, 'true_seqs'), exist_ok=True)
                #     true_fasta = fasta.FastaFile()
                #     true_fasta['true_seq_1'] = "".join([ALPHABET[x] for _ix, x in enumerate(S[_it]) if mask_for_loss[_it][_ix] == 1])
                #     true_fasta_path = os.path.join(sc_output_dir, 'true_seqs', 'true.fa')
                #     true_fasta.write(true_fasta_path)
                #     true_folded_dir = os.path.join(sc_output_dir, 'true_folded')
                #     if os.path.exists(true_folded_dir):
                #         shutil.rmtree(true_folded_dir)
                #     os.makedirs(true_folded_dir, exist_ok=False)
                #     true_folded_output = the_folding_model.fold_fasta(true_fasta_path, true_folded_dir)

                #     the_pdb_path = os.path.join(pdb_path, batch['WT_name'][_it])
                #     # folded generated with folded true
                #     gen_foldtrue_mpnn_results = mu.process_folded_outputs(os.path.join(true_folded_dir, 'folded_true_seq_1.pdb'), folded_output)
                #     # folded generated with pdb true
                #     gen_true_mpnn_results = mu.process_folded_outputs(the_pdb_path, folded_output)
                #     # folded true with pdb true
                #     foldtrue_true_mpnn_results = mu.process_folded_outputs(the_pdb_path, true_folded_output)

                #     seq_revovery = (S_sp[_it] == S[_it]).float().mean().item()
                #     gen_foldtrue_mpnn_results['seq_recovery'] = seq_revovery
                #     gen_true_mpnn_results['seq_recovery'] = seq_revovery
                #     gen_foldtrue_mpnn_results_list.append(gen_foldtrue_mpnn_results)
                #     gen_true_mpnn_results_list.append(gen_true_mpnn_results)
                #     foldtrue_true_mpnn_results_list.append(foldtrue_true_mpnn_results)
                gen_foldtrue_mpnn_results_list, gen_true_mpnn_results_list, foldtrue_true_mpnn_results_list = cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss)
                gen_foldtrue_mpnn_results_merge.extend(gen_foldtrue_mpnn_results_list)
                gen_true_mpnn_results_merge.extend(gen_true_mpnn_results_list)
                foldtrue_true_mpnn_results_merge.extend(foldtrue_true_mpnn_results_list)
                print('aaa')

            train_sp_accuracy = train_sp_acc / train_sp_weights
            train_sp_accuracy_ = np.format_float_positional(np.float32(train_sp_accuracy), unique=False, precision=3)

            gen_foldtrue_mpnn_results_merge = pd.concat(gen_foldtrue_mpnn_results_merge)
            gen_true_mpnn_results_merge = pd.concat(gen_true_mpnn_results_merge)
            foldtrue_true_mpnn_results_merge = pd.concat(foldtrue_true_mpnn_results_merge)
            gen_foldtrue_mpnn_results_merge.to_csv('/data/ueharam/sc_tmp/train_gen_foldtrue_mpnn_results.csv')
            gen_true_mpnn_results_merge.to_csv('/data/ueharam/sc_tmp/train_gen_true_mpnn_results.csv')
            foldtrue_true_mpnn_results_merge.to_csv('/data/ueharam/sc_tmp/train_foldtrue_true_mpnn_results.csv')

            train_gen_foldtrue_rmsd, train_gen_foldtrue_success_rate = parse_df(gen_foldtrue_mpnn_results_merge)
            train_gen_true_rmsd, train_gen_true_success_rate = parse_df(gen_true_mpnn_results_merge)
            train_foldtrue_true_rmsd, train_foldtrue_true_success_rate = parse_df(foldtrue_true_mpnn_results_merge)

            print(len(loader_valid))
            valid_sp_acc, valid_sp_weights = 0., 0.
            gen_foldtrue_mpnn_results_merge = []
            gen_true_mpnn_results_merge = []
            foldtrue_true_mpnn_results_merge = []
            for _, batch in tqdm(enumerate(loader_valid)):
                print('aaa')
                X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
                S_sp, _, _ = noise_interpolant.sample(fmif_model, X, mask, chain_M, residue_idx, chain_encoding_all)
                true_false_sp = (S_sp == S).float()
                mask_for_loss = mask*chain_M
                valid_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
                valid_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                gen_foldtrue_mpnn_results_list, gen_true_mpnn_results_list, foldtrue_true_mpnn_results_list = cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss)
                gen_foldtrue_mpnn_results_merge.extend(gen_foldtrue_mpnn_results_list)
                gen_true_mpnn_results_merge.extend(gen_true_mpnn_results_list)
                foldtrue_true_mpnn_results_merge.extend(foldtrue_true_mpnn_results_list)

            validation_sp_accuracy = valid_sp_acc / valid_sp_weights
            validation_sp_accuracy_ = np.format_float_positional(np.float32(validation_sp_accuracy), unique=False, precision=3)

            gen_foldtrue_mpnn_results_merge = pd.concat(gen_foldtrue_mpnn_results_merge)
            gen_true_mpnn_results_merge = pd.concat(gen_true_mpnn_results_merge)
            foldtrue_true_mpnn_results_merge = pd.concat(foldtrue_true_mpnn_results_merge)
            gen_foldtrue_mpnn_results_merge.to_csv('/data/ueharam/sc_tmp/valid_gen_foldtrue_mpnn_results.csv')
            gen_true_mpnn_results_merge.to_csv('/data/ueharam/sc_tmp/valid_gen_true_mpnn_results.csv')
            foldtrue_true_mpnn_results_merge.to_csv('/data/ueharam/sc_tmp/valid_foldtrue_true_mpnn_results.csv')

            valid_gen_foldtrue_rmsd, valid_gen_foldtrue_success_rate = parse_df(gen_foldtrue_mpnn_results_merge)
            valid_gen_true_rmsd, valid_gen_true_success_rate = parse_df(gen_true_mpnn_results_merge)
            valid_foldtrue_true_rmsd, valid_foldtrue_true_success_rate = parse_df(foldtrue_true_mpnn_results_merge)


            print(len(loader_test))
            test_sp_acc, test_sp_weights = 0., 0.
            gen_foldtrue_mpnn_results_merge = []
            gen_true_mpnn_results_merge = []
            foldtrue_true_mpnn_results_merge = []
            for _, batch in tqdm(enumerate(loader_test)):
                print('aaa')
                # X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
                S_sp, _, _ = noise_interpolant.sample(fmif_model, X, mask, chain_M, residue_idx, chain_encoding_all)
                true_false_sp = (S_sp == S).float()
                mask_for_loss = mask*chain_M
                test_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
                test_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                gen_foldtrue_mpnn_results_list, gen_true_mpnn_results_list, foldtrue_true_mpnn_results_list = cal_rmsd(S_sp, S, batch, the_folding_model, pdb_path, mask_for_loss)
                gen_foldtrue_mpnn_results_merge.extend(gen_foldtrue_mpnn_results_list)
                gen_true_mpnn_results_merge.extend(gen_true_mpnn_results_list)
                foldtrue_true_mpnn_results_merge.extend(foldtrue_true_mpnn_results_list)

            test_sp_accuracy = test_sp_acc / test_sp_weights
            test_sp_accuracy_ = np.format_float_positional(np.float32(test_sp_accuracy), unique=False, precision=3)

            gen_foldtrue_mpnn_results_merge = pd.concat(gen_foldtrue_mpnn_results_merge)
            gen_true_mpnn_results_merge = pd.concat(gen_true_mpnn_results_merge)
            foldtrue_true_mpnn_results_merge = pd.concat(foldtrue_true_mpnn_results_merge)
            gen_foldtrue_mpnn_results_merge.to_csv('/data/ueharam/sc_tmp/test_gen_foldtrue_mpnn_results.csv')
            gen_true_mpnn_results_merge.to_csv('/data/ueharam/sc_tmp/test_gen_true_mpnn_results.csv')
            foldtrue_true_mpnn_results_merge.to_csv('/data/ueharam/sc_tmp/test_foldtrue_true_mpnn_results.csv')

            test_gen_foldtrue_rmsd, test_gen_foldtrue_success_rate = parse_df(gen_foldtrue_mpnn_results_merge)
            test_gen_true_rmsd, test_gen_true_success_rate = parse_df(gen_true_mpnn_results_merge)
            test_foldtrue_true_rmsd, test_foldtrue_true_success_rate = parse_df(foldtrue_true_mpnn_results_merge)

        print(f"Train SP accuracy: {train_sp_accuracy_}")
        print(f"Validation SP accuracy: {validation_sp_accuracy_}")
        print(f"Test SP accuracy: {test_sp_accuracy_}")
        print(f"Train gen_foldtrue_rmsd: {train_gen_foldtrue_rmsd}, success rate: {train_gen_foldtrue_success_rate}, gen_true_rmsd: {train_gen_true_rmsd}, success rate: {train_gen_true_success_rate}, foldtrue_true_rmsd: {train_foldtrue_true_rmsd}, success rate: {train_foldtrue_true_success_rate}")
        print(f"Validation gen_foldtrue_rmsd: {valid_gen_foldtrue_rmsd}, success rate: {valid_gen_foldtrue_success_rate}, gen_true_rmsd: {valid_gen_true_rmsd}, success rate: {valid_gen_true_success_rate}, foldtrue_true_rmsd: {valid_foldtrue_true_rmsd}, success rate: {valid_foldtrue_true_success_rate}")
        print(f"Test gen_foldtrue_rmsd: {test_gen_foldtrue_rmsd}, success rate: {test_gen_foldtrue_success_rate}, gen_true_rmsd: {test_gen_true_rmsd}, success rate: {test_gen_true_success_rate}, foldtrue_true_rmsd: {test_foldtrue_true_rmsd}, success rate: {test_foldtrue_true_success_rate}")
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_pdbs", type=str, default="/data/wangc239/proteindpo_data/AlphaFold_model_PDBs", help="path for loading pdb files") 
    argparser.add_argument("--path_for_dpo_dicts", type=str, default="/data/wangc239/proteindpo_data/processed_data", help="path for loading ProteinDPO dict files") 

    argparser.add_argument("--path_for_outputs", type=str, default="/data/wangc239/protein_oracle/outputs", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train for") # 200
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--batch_size", type=int, default=128, help="number of sequences for one batch")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")   # 48
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout") # TODO
    argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")   # TODO
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=str2bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=str2bool, default=True, help="train with mixed precision")
    argparser.add_argument("--initialize_with_pretrain", type=str2bool, default=False, help="initialize with FMIF weights")
    argparser.add_argument("--train_using_diff", type=str2bool, default=False, help="training using difference in dG")

    argparser.add_argument("--wandb_name", type=str, default="debug", help="wandb run name")
    argparser.add_argument("--lr", type=float, default=1e-4)
    argparser.add_argument("--wd", type=float, default=1e-4)

    argparser.add_argument("--min_t", type=float, default=1e-2)
    argparser.add_argument("--schedule", type=str, default='linear') # other schedule is not implemented
    argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
    argparser.add_argument("--temp", type=float, default=0.1)
    argparser.add_argument("--noise", type=float, default=1.0) # 20.0
    argparser.add_argument("--interpolant_type", type=str, default='masking')
    argparser.add_argument("--do_purity", type=str2bool, default=False) # True
    argparser.add_argument("--num_timesteps", type=int, default=50) # 50
    argparser.add_argument("--seed", type=int, default=0)
 
    args = argparser.parse_args()    
    print(args)
    main(args)
