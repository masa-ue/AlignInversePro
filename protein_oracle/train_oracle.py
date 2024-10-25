import argparse
import os.path
# import hydra
import random
import string
import datetime
from datetime import date
import logging
import pickle
from protein_oracle.utils import str2bool

runid = ''.join(random.choice(string.ascii_letters) for i in range(10))+ '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
# logger = logging.getLogger(__name__)
# file_handler = logging.FileHandler(f'logs/{runid}.log') ########
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# @hydra.main(version_base=None, config_path="configs", config_name="base.yaml")
def main(args):
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
    from protein_oracle.utils import set_seed
    from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
    from protein_oracle.model_utils import ProteinMPNNOracle, get_std_opt
    from fmif.model_utils import ProteinMPNNFMIF
    # from fmif.utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, set_seed
    # from fmif.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNNFMIF
    # from fmif.fm_utils import Interpolant, fm_model_step
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
        # logger.info("Debug mode, not logging to wandb")
    # logger.info(f"Run ID: {runid}")
    # logger.info(f"Arguments: {args}")
    with open (logfile, 'a') as f:
        f.write(f"Run ID: {runid}\n")
        f.write(f"Arguments: {args}\n")


    # data_path = args.path_for_training_data
    # params = {
    #     "LIST"    : f"{data_path}/list.csv", 
    #     "VAL"     : f"{data_path}/valid_clusters.txt",
    #     "TEST"    : f"{data_path}/test_clusters.txt",
    #     "DIR"     : f"{data_path}",
    #     "DATCUT"  : "2030-Jan-01",
    #     "RESCUT"  : args.rescut, #resolution cutoff for PDBs
    #     "HOMO"    : 0.70 #min seq.id. to detect homo chains
    # }


    # LOAD_PARAM = {'batch_size': 1,
    #               'shuffle': True,
    #               'pin_memory':False,
    #               'num_workers': 4}

   
    # if args.debug:
    #     args.num_examples_per_epoch = 50
    #     args.max_protein_length = 1000
    #     args.batch_size = 1000

    # train, valid, test = build_training_clusters(params, args.debug)
    # print(len(train), len(valid), len(test)) # 23349 1464 1539
     
    # train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    # train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    # valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    # valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    # test_set = PDB_dataset(list(test.keys()), loader_pdb, test, params)
    # test_loader = torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)


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
    dpo_valid_dict_wt = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict_wt.pkl'), 'rb'))
    dpo_test_dict_wt = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
    dpo_valid_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict.pkl'), 'rb'))
    dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict.pkl'), 'rb'))
    print(len(dpo_train_dict), len(dpo_valid_dict_wt), len(dpo_test_dict_wt), len(dpo_valid_dict), len(dpo_test_dict))
    if args.include_all:
        dpo_train_dict_complete = {**dpo_train_dict, **dpo_valid_dict, **dpo_test_dict}
    elif args.include_wt:
        dpo_train_dict_complete = {**dpo_train_dict, **dpo_valid_dict_wt, **dpo_test_dict_wt}
    else:
        dpo_train_dict_complete = dpo_train_dict

    # dpo_train_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict.pkl'), 'rb'))
    dpo_train_dataset = ProteinDPODataset(dpo_train_dict_complete, pdb_idx_dict, pdb_structures)
    loader_train = DataLoader(dpo_train_dataset, batch_size=args.batch_size, shuffle=True)
    dpo_valid_dataset = ProteinDPODataset(dpo_valid_dict, pdb_idx_dict, pdb_structures)
    loader_valid = DataLoader(dpo_valid_dataset, batch_size=args.batch_size, shuffle=False)
    dpo_test_dataset = ProteinDPODataset(dpo_test_dict, pdb_idx_dict, pdb_structures)
    loader_test = DataLoader(dpo_test_dataset, batch_size=args.batch_size, shuffle=False)

    dpo_valid_dataset_wt = ProteinDPODataset(dpo_valid_dict_wt, pdb_idx_dict, pdb_structures)
    loader_valid_wt = DataLoader(dpo_valid_dataset_wt, batch_size=args.batch_size, shuffle=True)
    dpo_test_dataset_wt = ProteinDPODataset(dpo_test_dict_wt, pdb_idx_dict, pdb_structures)
    loader_test_wt = DataLoader(dpo_test_dataset_wt, batch_size=args.batch_size, shuffle=True)


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
            # else:
            #     print(f"Parameter {name} not found in FMIF model")

        del fmif_model

    # noise_interpolant = Interpolant(args)
    # noise_interpolant.set_device(device)


    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    # optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)


    if PATH:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    with ProcessPoolExecutor(max_workers=12) as executor:
        # import time
        # t0 = time.time()
        # q = queue.Queue(maxsize=3)
        # p = queue.Queue(maxsize=3)
        # pq = queue.Queue(maxsize=3)
        # for i in range(3):
        #     q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        #     p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        #     pq.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        # pdb_dict_train = q.get().result()
        # pdb_dict_valid = p.get().result()
        # pdb_dict_test = pq.get().result()
        # # print(len(pdb_dict_train), len(pdb_dict_valid))
       
        # dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        # dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        # dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=args.max_protein_length)
        # # print(len(dataset_train), len(dataset_valid))
        
        # loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        # loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        # loader_test = StructureLoader(dataset_test, batch_size=args.batch_size)
        # print(len(loader_train), len(loader_valid), len(loader_test))
        # t1 = time.time()
        # print('time for loading data:', t1-t0)
        
        # reload_c = 0 
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            # train_sum, train_weights = 0., 0.
            # train_acc = 0.
            train_loss = 0.
            train_dg_true = []
            train_dg_pred = []
            train_wtnames = []
            # if e % args.reload_data_every_n_epochs == 0:
            #     if reload_c != 0:
            #         # t0 = time.time()
            #         pdb_dict_train = q.get().result()
            #         dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
            #         loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
            #         pdb_dict_valid = p.get().result()
            #         dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
            #         loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
            #         pdb_dict_test = pq.get().result()
            #         dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=args.max_protein_length)
            #         loader_test = StructureLoader(dataset_test, batch_size=args.batch_size)

            #         q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            #         p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            #         pq.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            #         # t1 = time.time()
            #         # print('time for loading data:', t1-t0)
            #         # print(len(pdb_dict_train), len(pdb_dict_valid))
            #     reload_c += 1
            for _, batch in tqdm(enumerate(loader_train)):
                if args.debug and _ > 20:
                    break

                # start_batch = time.time()
                X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
                dg_ml = batch['dG_ML'].to(dtype=torch.float32, device=device)
                dg_ml_wt = batch['dG_ML_wt'].to(dtype=torch.float32, device=device)
                ddg_ml = dg_ml - dg_ml_wt
                # noisy_batch = noise_interpolant.corrupt_batch((X, S, mask, chain_M, residue_idx, chain_encoding_all))

                # elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                # mask_for_loss = mask*chain_M
                
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
                        # log_probs = fm_model_step(model, noisy_batch)
                        # log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        # _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
           
                    scaler.scale(loss_dg).backward()
                     
                    # if args.gradient_norm > 0.0:
                    #     total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # log_probs = fm_model_step(model, noisy_batch)
                    # log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    # _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    # loss_av_smoothed.backward()
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

                    # if args.gradient_norm > 0.0:
                    #     total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()
                
                # loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                # train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                # train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                # train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                train_loss += loss_dg.item()
                total_step += 1
                if args.predict_ddg:
                    train_dg_true.extend(ddg_ml.cpu().data.numpy())
                else:
                    train_dg_true.extend(dg_ml.cpu().data.numpy())
                train_dg_pred.extend(dg_pred.cpu().data.numpy())
                train_wtnames.extend(batch['WT_name'])

            
            model.eval()
            with torch.no_grad():
                # validation_sum, validation_weights = 0., 0.
                # validation_acc = 0.
                validation_loss = 0.
                valid_dg_true = []
                valid_dg_pred = []
                valid_wtnames = []
                for _, batch in enumerate(loader_valid):
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
                    validation_loss += loss_dg.item()
                    valid_dg_true.extend(dg_ml.cpu().data.numpy())
                    valid_dg_pred.extend(dg_pred.cpu().data.numpy())
                    valid_wtnames.extend(batch['WT_name'])

                    # X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    # log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    # mask_for_loss = mask*chain_M
                    # loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    
                    # validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    # validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    # validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                valid_wt_mse = 0.
                for _, batch in enumerate(loader_valid_wt):
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
                    valid_wt_mse += loss_dg.item()

                test_loss = 0.
                test_dg_true = []
                test_dg_pred = []
                test_wtnames = []
                for _, batch in enumerate(loader_test):
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
                    # loss_dg = F.mse_loss(dg_pred, dg_ml)
                    test_loss += loss_dg.item()
                    if args.predict_ddg:
                        test_dg_true.extend(ddg_ml.cpu().data.numpy())
                    else:
                        test_dg_true.extend(dg_ml.cpu().data.numpy())
                    test_dg_pred.extend(dg_pred.cpu().data.numpy())
                    test_wtnames.extend(batch['WT_name'])

                test_wt_mse = 0.
                for _, batch in enumerate(loader_test_wt):
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
                    test_wt_mse += loss_dg.item()

            

            # validation_sp_accuracy_ = '-'
            # test_sp_accuracy_ = '-'

            # if (e+1) % args.eval_every_n_epochs == 0:
            #     with torch.no_grad():
            #         print(len(loader_valid))
            #         valid_sp_acc, valid_sp_weights = 0., 0.
            #         for _, batch in tqdm(enumerate(loader_valid)):
            #             X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            #             S_sp, _, _ = noise_interpolant.sample(model, X, mask, chain_M, residue_idx, chain_encoding_all)
            #             # print(S_sp.shape)
            #             # print(S.shape)
            #             # print((S_sp == S))
            #             true_false_sp = (S_sp == S).float()
            #             mask_for_loss = mask*chain_M
            #             valid_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
            #             valid_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            #         validation_sp_accuracy = valid_sp_acc / valid_sp_weights
            #         validation_sp_accuracy_ = np.format_float_positional(np.float32(validation_sp_accuracy), unique=False, precision=3)

            #         print(len(loader_test))
            #         test_sp_acc, test_sp_weights = 0., 0.
            #         for _, batch in tqdm(enumerate(loader_test)):
            #             X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            #             S_sp, _, _ = noise_interpolant.sample(model, X, mask, chain_M, residue_idx, chain_encoding_all)
            #             true_false_sp = (S_sp == S).float()
            #             mask_for_loss = mask*chain_M
            #             test_sp_acc += torch.sum(true_false_sp * mask_for_loss).cpu().data.numpy()
            #             test_sp_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            #         test_sp_accuracy = test_sp_acc / test_sp_weights
            #         test_sp_accuracy_ = np.format_float_positional(np.float32(test_sp_accuracy), unique=False, precision=3)
            
            # train_loss = train_sum / train_weights
            train_loss = train_loss / len(loader_train)
            # train_accuracy = train_acc / train_weights
            # train_perplexity = np.exp(train_loss)
            # validation_loss = validation_sum / validation_weights
            validation_loss = validation_loss / len(loader_valid)
            valid_wt_mse = valid_wt_mse / len(loader_valid_wt)
            # validation_accuracy = validation_acc / validation_weights
            # validation_perplexity = np.exp(validation_loss)
            test_loss = test_loss / len(loader_test)
            test_wt_mse = test_wt_mse / len(loader_test_wt)

            # calculate average pearson correlation of dG_ML and dg_pred clustered by wt_names
            from scipy.stats import pearsonr
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

            valid_dg_true = np.array(valid_dg_true)
            valid_dg_pred = np.array(valid_dg_pred)
            valid_wtnames = np.array(valid_wtnames)
            # print(valid_dg_true.shape, valid_dg_pred.shape, valid_wtnames.shape)
            pearson_corrs = []
            for wt_name in np.unique(valid_wtnames):
                idx = np.where(valid_wtnames == wt_name)
                if len(idx[0]) > 1:
                    corr, _ = pearsonr(valid_dg_true[idx], valid_dg_pred[idx])
                    pearson_corrs.append(corr)
            # print(pearson_corrs)
            validation_pearson = np.mean(pearson_corrs)

            test_dg_true = np.array(test_dg_true)
            test_dg_pred = np.array(test_dg_pred)
            test_wtnames = np.array(test_wtnames)
            pearson_corrs = []
            for wt_name in np.unique(test_wtnames):
                idx = np.where(test_wtnames == wt_name)
                if len(idx[0]) > 1:
                    corr, _ = pearsonr(test_dg_true[idx], test_dg_pred[idx])
                    pearson_corrs.append(corr)
            test_pearson = np.mean(pearson_corrs)

            # train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            # validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            # train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            # validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
            train_loss_ = np.format_float_positional(np.float32(train_loss), unique=False, precision=3)
            validation_loss_ = np.format_float_positional(np.float32(validation_loss), unique=False, precision=3)
            test_loss_ = np.format_float_positional(np.float32(test_loss), unique=False, precision=3)
            train_pearson_ = np.format_float_positional(np.float32(train_pearson), unique=False, precision=3)
            validation_pearson_ = np.format_float_positional(np.float32(validation_pearson), unique=False, precision=3)
            test_pearson_ = np.format_float_positional(np.float32(test_pearson), unique=False, precision=3)
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
            #     f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss}, valid: {validation_loss}, valid_pearson: {validation_pearson}, test: {test_loss}, test_pearson: {test_pearson}\n')
            # print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss}, valid: {validation_loss}, valid_pearson: {validation_pearson}, test: {test_loss}, test_pearson: {test_pearson}')
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss_}, valid: {validation_loss_}, test: {test_loss_}, train_pearson: {train_pearson_}, valid_pearson: {validation_pearson_}, test_pearson: {test_pearson_}, valid_wt_mse: {valid_wt_mse}, test_wt_mse: {test_wt_mse}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss_}, valid: {validation_loss_}, test: {test_loss_}, train_pearson: {train_pearson_}, valid_pearson: {validation_pearson_}, test_pearson: {test_pearson_}, valid_wt_mse: {valid_wt_mse}, test_wt_mse: {test_wt_mse}')
            #     f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, valid_sp_acc: {validation_sp_accuracy_}, test_sp_acc: {test_sp_accuracy_}\n')
            # print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, valid_sp_acc: {validation_sp_accuracy_}, test_sp_acc: {test_sp_accuracy_}')
            
            if not args.debug:
                # wandb.log({"train_perplexity": train_perplexity, "valid_perplexity": validation_perplexity, "train_accuracy": train_accuracy, "valid_accuracy": validation_accuracy}, step=total_step, commit=False)
                # if (e+1) % args.eval_every_n_epochs == 0:
                #     wandb.log({"valid_sp_accuracy": validation_sp_accuracy, "test_sp_accuracy": test_sp_accuracy}, step=total_step)
                # wandb.log({"train_loss": train_loss, "valid_loss": validation_loss, "valid_pearson": validation_pearson, "test_loss": test_loss, "test_pearson": test_pearson}, step=total_step)
                wandb.log({"train_loss": train_loss, "valid_loss": validation_loss, "test_loss": test_loss, "train_pearson": train_pearson, "valid_pearson": validation_pearson, "test_pearson": test_pearson, "valid_wt_mse": valid_wt_mse, "test_wt_mse": test_wt_mse}, step=total_step)
                
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'optimizer_state_dict': optimizer.optimizer.state_dict(),
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
                        # 'optimizer_state_dict': optimizer.optimizer.state_dict(),
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
    # argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    # argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=128, help="number of sequences for one batch")   # TODO
    # argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
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
    argparser.add_argument("--include_wt", type=str2bool, default=False, help="include valid and test wt into training, for predict_dgg=True")
    argparser.add_argument("--include_all", type=str2bool, default=False, help="include valid and test into training, for evaluation oracle")

    argparser.add_argument("--wandb_name", type=str, default="debug", help="wandb run name")
    argparser.add_argument("--lr", type=float, default=1e-4)
    argparser.add_argument("--wd", type=float, default=1e-4)

    # argparser.add_argument("--min_t", type=float, default=1e-2)
    # argparser.add_argument("--schedule", type=str, default='linear') # other schedule is not implemented
    # argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
    # argparser.add_argument("--temp", type=float, default=0.1)
    # argparser.add_argument("--noise", type=float, default=1.0) # 20.0
    # argparser.add_argument("--interpolant_type", type=str, default='masking')
    # argparser.add_argument("--do_purity", type=bool, default=True)
    # argparser.add_argument("--num_timesteps", type=int, default=500)
    argparser.add_argument("--seed", type=int, default=0)
    # argparser.add_argument("--eval_every_n_epochs", type=int, default=20)
 
    args = argparser.parse_args()    
    print(args)
    main(args)
