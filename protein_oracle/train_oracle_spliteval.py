##### splitting mutants of in-distribution proteins for evaluation #####

import argparse
import random
import string
import datetime
from datetime import date
import pickle
from protein_oracle.utils import str2bool

runid = ''.join(random.choice(string.ascii_letters) for i in range(10))+ '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

def main(args):
    import time, os
    import warnings
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import os.path
    from concurrent.futures import ProcessPoolExecutor    
    from protein_oracle.utils import set_seed
    from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
    from protein_oracle.model_utils import ProteinMPNNOracle
    from fmif.model_utils import ProteinMPNNFMIF
    from scipy.stats import pearsonr
    from tqdm import tqdm
    import wandb
    warnings.filterwarnings("ignore", category=UserWarning)

    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    if args.debug:
        args.path_for_outputs = '/data/wangc239/protein_oracle/outputs_debug'
    
    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    base_folder = os.path.join(base_folder, runid)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    if base_folder[-1] != '/':
        base_folder += '/'
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)
    
    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')
    
    assert torch.cuda.is_available(), "CUDA is not available"
    # set random seed
    set_seed(args.seed, use_cuda=True)

    # wandb
    if not args.debug:
        wandb.init(entity='wangc239', project='protein_oracle', name=args.wandb_name, dir=base_folder, id=runid)
        curr_date = date.today().strftime("%B %d, %Y")
        wandb.config.update({"curr_date": curr_date}, allow_val_change=True)
        wandb.config.update(args, allow_val_change=True)
    else:
        with open(logfile, 'a') as f:
            f.write("Debug mode, not logging to wandb\n")
    with open (logfile, 'a') as f:
        f.write(f"Run ID: {runid}\n")
        f.write(f"Arguments: {args}\n")

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
    dpo_train_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict_curated.pkl'), 'rb'))
    # dpo_valid_dict_wt = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict_wt.pkl'), 'rb'))
    # dpo_test_dict_wt = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
    dpo_valid_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict.pkl'), 'rb'))
    dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict.pkl'), 'rb'))
    
    # split dpo_train_dict into 80% and 20%
    def split_dict(dpo_dict, split_ratio=0.8):
        keys = list(dpo_dict.keys())
        random.shuffle(keys)
        split_idx = int(len(keys) * split_ratio)
        dpo_dict1 = {k: dpo_dict[k] for k in keys[:split_idx]}
        dpo_dict2 = {k: dpo_dict[k] for k in keys[split_idx:]}
        return dpo_dict1, dpo_dict2
    
    dpo_train_dict_1, dpo_train_dict_2 = split_dict(dpo_train_dict, split_ratio=0.8)
    dpo_valid_dict_1, dpo_valid_dict_2 = split_dict(dpo_valid_dict, split_ratio=0.8)
    dpo_test_dict_1, dpo_test_dict_2 = split_dict(dpo_test_dict, split_ratio=0.8)
    print(len(dpo_train_dict_1), len(dpo_train_dict_2), len(dpo_valid_dict_1), len(dpo_valid_dict_2), len(dpo_test_dict_1), len(dpo_test_dict_2))
    
    if args.include_all:
        dpo_train_dict_complete = {**dpo_train_dict_1, **dpo_valid_dict_1, **dpo_test_dict_1}
    else:
        dpo_train_dict_complete = dpo_train_dict_1

    dpo_train_dataset = ProteinDPODataset(dpo_train_dict_complete, pdb_idx_dict, pdb_structures)
    loader_train = DataLoader(dpo_train_dataset, batch_size=args.batch_size, shuffle=True)

    dpo_train_dataset_2 = ProteinDPODataset(dpo_train_dict_2, pdb_idx_dict, pdb_structures)
    loader_train_2 = DataLoader(dpo_train_dataset_2, batch_size=args.batch_size, shuffle=False)
    dpo_valid_dataset_1 = ProteinDPODataset(dpo_valid_dict_1, pdb_idx_dict, pdb_structures)
    loader_valid_1 = DataLoader(dpo_valid_dataset_1, batch_size=args.batch_size, shuffle=False)
    dpo_valid_dataset_2 = ProteinDPODataset(dpo_valid_dict_2, pdb_idx_dict, pdb_structures)
    loader_valid_2 = DataLoader(dpo_valid_dataset_2, batch_size=args.batch_size, shuffle=False)
    dpo_test_dataset_1 = ProteinDPODataset(dpo_test_dict_1, pdb_idx_dict, pdb_structures)
    loader_test_1 = DataLoader(dpo_test_dataset_1, batch_size=args.batch_size, shuffle=False)
    dpo_test_dataset_2 = ProteinDPODataset(dpo_test_dict_2, pdb_idx_dict, pdb_structures)
    loader_test_2 = DataLoader(dpo_test_dataset_2, batch_size=args.batch_size, shuffle=False)

    # dpo_valid_dataset_wt = ProteinDPODataset(dpo_valid_dict_wt, pdb_idx_dict, pdb_structures)
    # loader_valid_wt = DataLoader(dpo_valid_dataset_wt, batch_size=args.batch_size, shuffle=True)
    # dpo_test_dataset_wt = ProteinDPODataset(dpo_test_dict_wt, pdb_idx_dict, pdb_structures)
    # loader_test_wt = DataLoader(dpo_test_dataset_wt, batch_size=args.batch_size, shuffle=True)


    model = ProteinMPNNOracle(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise
                        )
    model.to(device)

    if args.initialize_with_pretrain:
        fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                            edge_features=args.hidden_dim,
                            hidden_dim=args.hidden_dim,
                            num_encoder_layers=args.num_encoder_layers,
                            num_decoder_layers=args.num_encoder_layers,
                            k_neighbors=args.num_neighbors,
                            dropout=args.dropout,
                            augment_eps=args.backbone_noise
                            )
        fmif_model.to(device)
        fmif_model.load_state_dict(torch.load('/data/wangc239/pmpnn/outputs/jnvGDJFmCj_20240710_145313/model_weights/epoch300_step447702.pt')['model_state_dict'])
    
        for name, param in model.named_parameters():
            if name in fmif_model.state_dict():
                param.data = fmif_model.state_dict()[name].data.clone()

        del fmif_model

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if PATH:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def eval_loader(loader_eval):
        eval_loss = 0.
        eval_dg_true = []
        eval_dg_pred = []
        eval_wtnames = []
        for _, batch in enumerate(loader_eval):
            X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
            dg_ml = batch['dG_ML'].to(dtype=torch.float32, device=device)
            dg_ml_wt = batch['dG_ML_wt'].to(dtype=torch.float32, device=device)
            ddg_ml = dg_ml - dg_ml_wt
            dg_pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            if args.predict_ddg:
                loss_dg = F.mse_loss(dg_pred, ddg_ml)
            else:
                if args.train_using_diff:
                    dg_pred_wt = model(X, S_wt, mask, chain_M, residue_idx, chain_encoding_all)
                    loss_dg = F.mse_loss(dg_pred - dg_pred_wt, dg_ml - dg_ml_wt)
                else:
                    loss_dg = F.mse_loss(dg_pred, dg_ml)
            eval_loss += loss_dg.item()
            if args.predict_ddg:
                eval_dg_true.extend(ddg_ml.cpu().data.numpy())
            else:
                eval_dg_true.extend(dg_ml.cpu().data.numpy())
            eval_dg_pred.extend(dg_pred.cpu().data.numpy())
            eval_wtnames.extend(batch['WT_name'])
        eval_loss = eval_loss / len(loader_eval)
        eval_dg_true = np.array(eval_dg_true)
        eval_dg_pred = np.array(eval_dg_pred)
        eval_wtnames = np.array(eval_wtnames)
        pearson_corrs = []
        for wt_name in np.unique(eval_wtnames):
            idx = np.where(eval_wtnames == wt_name)
            if len(idx[0]) > 1:
                corr, _ = pearsonr(eval_dg_true[idx], eval_dg_pred[idx])
                pearson_corrs.append(corr)
        eval_pearson = np.mean(pearson_corrs)
        eval_loss_ = np.format_float_positional(np.float32(eval_loss), unique=False, precision=3)
        eval_pearson_ = np.format_float_positional(np.float32(eval_pearson), unique=False, precision=3)
        return eval_loss, eval_pearson, eval_loss_, eval_pearson_

    with ProcessPoolExecutor(max_workers=12) as executor:
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_loss = 0.
            train_dg_true = []
            train_dg_pred = []
            train_wtnames = []
            for _, batch in tqdm(enumerate(loader_train)):
                if args.debug and _ > 20:
                    break

                X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
                dg_ml = batch['dG_ML'].to(dtype=torch.float32, device=device)
                dg_ml_wt = batch['dG_ML_wt'].to(dtype=torch.float32, device=device)
                ddg_ml = dg_ml - dg_ml_wt
                optimizer.zero_grad()
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        dg_pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        if args.predict_ddg:
                            loss_dg = F.mse_loss(dg_pred, ddg_ml)
                        else:
                            if args.train_using_diff:
                                dg_pred_wt = model(X, S_wt, mask, chain_M, residue_idx, chain_encoding_all)
                                loss_dg = F.mse_loss(dg_pred - dg_pred_wt, dg_ml - dg_ml_wt)
                            else:
                                loss_dg = F.mse_loss(dg_pred, dg_ml)

                    scaler.scale(loss_dg).backward()
                     
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    dg_pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    if args.predict_ddg:
                        loss_dg = F.mse_loss(dg_pred, ddg_ml)
                    else:
                        if args.train_using_diff:
                            dg_pred_wt = model(X, S_wt, mask, chain_M, residue_idx, chain_encoding_all)
                            loss_dg = F.mse_loss(dg_pred - dg_pred_wt, dg_ml - dg_ml_wt)
                        else:
                            loss_dg = F.mse_loss(dg_pred, dg_ml)
                    loss_dg.backward()

                    optimizer.step()

                train_loss += loss_dg.item()
                total_step += 1
                if args.predict_ddg:
                    train_dg_true.extend(ddg_ml.cpu().data.numpy())
                else:
                    train_dg_true.extend(dg_ml.cpu().data.numpy())
                train_dg_pred.extend(dg_pred.cpu().data.numpy())
                train_wtnames.extend(batch['WT_name'])

            train_loss = train_loss / len(loader_train)
            # calculate average pearson correlation of dG_ML and dg_pred clustered by wt_names
            train_dg_true = np.array(train_dg_true)
            train_dg_pred = np.array(train_dg_pred)
            train_wtnames = np.array(train_wtnames)
            pearson_corrs = []
            for wt_name in np.unique(train_wtnames):
                idx = np.where(train_wtnames == wt_name)
                if len(idx[0]) > 1:
                    corr, _ = pearsonr(train_dg_true[idx], train_dg_pred[idx])
                    pearson_corrs.append(corr)
            train_pearson = np.mean(pearson_corrs)
            train_loss_ = np.format_float_positional(np.float32(train_loss), unique=False, precision=3)
            train_pearson_ = np.format_float_positional(np.float32(train_pearson), unique=False, precision=3)
    
            model.eval()
            with torch.no_grad():
                train_loss_2, train_pearson_2, train_loss_2_, train_pearson_2_ = eval_loader(loader_train_2)
                validation_loss_1, validation_pearson_1, validation_loss_1_, validation_pearson_1_ = eval_loader(loader_valid_1)
                validation_loss_2, validation_pearson_2, validation_loss_2_, validation_pearson_2_ = eval_loader(loader_valid_2)
                test_loss_1, test_pearson_1, test_loss_1_, test_pearson_1_ = eval_loader(loader_test_1)
                test_loss_2, test_pearson_2, test_loss_2_, test_pearson_2_ = eval_loader(loader_test_2)

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            # with open(logfile, 'a') as f:
            #     f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss_}, valid: {validation_loss_}, test: {test_loss_}, train_pearson: {train_pearson_}, valid_pearson: {validation_pearson_}, test_pearson: {test_pearson_}, valid_wt_mse: {valid_wt_mse}, test_wt_mse: {test_wt_mse}\n')
            # print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss_}, valid: {validation_loss_}, test: {test_loss_}, train_pearson: {train_pearson_}, valid_pearson: {validation_pearson_}, test_pearson: {test_pearson_}, valid_wt_mse: {valid_wt_mse}, test_wt_mse: {test_wt_mse}')
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss_}, train2: {train_loss_2_}, valid1: {validation_loss_1_}, valid2: {validation_loss_2_}, test1: {test_loss_1_}, test2: {test_loss_2_}, train_pearson: {train_pearson_}, train2_pearson: {train_pearson_2_}, valid1_pearson: {validation_pearson_1_}, valid2_pearson: {validation_pearson_2_}, test1_pearson: {test_pearson_1_}, test2_pearson: {test_pearson_2_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss_}, train2: {train_loss_2_}, valid1: {validation_loss_1_}, valid2: {validation_loss_2_}, test1: {test_loss_1_}, test2: {test_loss_2_}, train_pearson: {train_pearson_}, train2_pearson: {train_pearson_2_}, valid1_pearson: {validation_pearson_1_}, valid2_pearson: {validation_pearson_2_}, test1_pearson: {test_pearson_1_}, test2_pearson: {test_pearson_2_}')

            if not args.debug:
                # wandb.log({"train_loss": train_loss, "valid_loss": validation_loss, "test_loss": test_loss, "train_pearson": train_pearson, "valid_pearson": validation_pearson, "test_pearson": test_pearson, "valid_wt_mse": valid_wt_mse, "test_wt_mse": test_wt_mse}, step=total_step)
                wandb.log({"train_loss": train_loss, "train2_loss": train_loss_2, "valid1_loss": validation_loss_1, "valid2_loss": validation_loss_2, "test1_loss": test_loss_1, "test2_loss": test_loss_2, "train_pearson": train_pearson, "train2_pearson": train_pearson_2, "valid1_pearson": validation_pearson_1, "valid2_pearson": validation_pearson_2, "test1_pearson": test_pearson_1, "test2_pearson": test_pearson_2}, step=total_step)
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename)

    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_pdbs", type=str, default="/data/wangc239/proteindpo_data/AlphaFold_model_PDBs", help="path for loading pdb files") 
    argparser.add_argument("--path_for_dpo_dicts", type=str, default="/data/wangc239/proteindpo_data/processed_data", help="path for loading ProteinDPO dict files") 

    # TODO: modify the outputs logging system and wandb logging
    argparser.add_argument("--path_for_outputs", type=str, default="/data/wangc239/protein_oracle/outputs", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train for") # 200
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--batch_size", type=int, default=128, help="number of sequences for one batch")   # TODO
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")   # 48
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")   # TODO
    argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")   # TODO
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=str2bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=str2bool, default=True, help="train with mixed precision")
    argparser.add_argument("--initialize_with_pretrain", type=str2bool, default=True, help="initialize with FMIF weights")
    argparser.add_argument("--train_using_diff", type=str2bool, default=False, help="training using difference in dG")
    argparser.add_argument("--predict_ddg", type=str2bool, default=False, help="model directly predicts ddG")
    argparser.add_argument("--include_all", type=str2bool, default=False, help="include valid and test into training, for evaluation oracle")

    argparser.add_argument("--wandb_name", type=str, default="debug", help="wandb run name")
    argparser.add_argument("--lr", type=float, default=1e-4)
    argparser.add_argument("--wd", type=float, default=1e-4)
    argparser.add_argument("--seed", type=int, default=0)
 
    args = argparser.parse_args()    
    print(args)
    main(args)
