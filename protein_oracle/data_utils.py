import torch
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser
import os
import numpy as np

MASKED_TOKEN = 'Z'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
ALPHABET_WITH_MASK = ALPHABET + MASKED_TOKEN
MASK_TOKEN_INDEX = ALPHABET_WITH_MASK.index(MASKED_TOKEN)


class ProteinStructureDataset(Dataset):
    def __init__(self, directory, max_len):
        """
        Args:
            directory (string): Directory with all the .pdb files.
            max_len (int): Maximum length of the protein sequences.
        """
        self.directory = directory
        self.max_len = max_len
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.pdb')]
        self.parser = PDBParser(QUIET=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filenames[idx])
        structure = self.parser.get_structure(id=None, file=file_path)
        model = structure[0]  # Assuming only one model per PDB file

        # Extract coordinates for N, CA, C, O atoms for each residue
        coords = []
        for residue in model.get_residues():
            try:
                n = residue['N'].get_coord()
                ca = residue['CA'].get_coord()
                c = residue['C'].get_coord()
                o = residue['O'].get_coord()
                coords.append([n, ca, c, o])
            except KeyError:
                continue  # Skip residues that do not have all atoms

        # Pad the coordinates to max_len
        coords = np.array(coords, dtype=np.float32)
        if len(coords) < self.max_len:
            padding = np.zeros((self.max_len - len(coords), 4, 3), dtype=np.float32)
            coords = np.concatenate((coords, padding), axis=0)
        elif len(coords) > self.max_len:
            print(f"Protein sequence in {self.filenames[idx]} is longer than max_len. Truncating.")
            coords = coords[:self.max_len]

        return torch.tensor(coords), self.filenames[idx]

# Example usage
# directory = '/data/wangc239/proteindpo_data/AlphaFold_model_PDBs'
# max_len = 75  # Define the maximum length of proteins
# dataset = ProteinStructureDataset(directory, max_len) # max_len set to 75 (sequences range from 31 to 74)
# loader = DataLoader(dataset, batch_size=1000, shuffle=False)

# # for batch in loader:
# #     print(batch[0].shape)  # Should print torch.Size([batch_size, max_len, 4, 3]) # torch.Size([862, 75, 4, 3])
# #     print(batch[1])  # Should print the filenames of the PDB files in the batch
# #     break  # For demonstration, break after the first batch

# # make a dict of pdb filename: index
# for batch in loader:
#     pdb_structures = batch[0]
#     pdb_filenames = batch[1]
#     pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
#     break

# print(len(pdb_idx_dict), pdb_structures.shape) # 862 torch.Size([862, 75, 4, 3])


# we can make pdb structure a dictionary

# make a dataset with dpo_train_dict

class ProteinDPODataset(Dataset):
    def __init__(self, dpo_train_dict, pdb_idx_dict, pdb_structure):
        self.dpo_train_dict = dpo_train_dict
        self.protein_list = list(dpo_train_dict.keys())
        self.pdb_idx_dict = pdb_idx_dict
        self.pdb_structure = pdb_structure

    def __len__(self):
        return len(self.protein_list)
    
    def __getitem__(self, idx):
        protein_name = self.protein_list[idx]
        protein_data = self.dpo_train_dict[protein_name]
        protein_structure = self.pdb_structure[self.pdb_idx_dict[protein_data[1]]]
        
        return {
            'protein_name': protein_name,
            'aa_seq': protein_data[0],
            'WT_name': protein_data[1],
            'aa_seq_wt': protein_data[2],
            'dG_ML': protein_data[3],
            # 'ddG_ML': protein_data[4],
            'dG_ML_wt': protein_data[5],
            # 'ddG_ML_wt': protein_data[6],
            'name_wt': protein_data[7],
            'structure': protein_structure,
            }


# import pickle
# dpo_train_dict = pickle.load(open('/data/wangc239/proteindpo_data/processed_data/dpo_train_dict.pkl', 'rb'))
# dpo_train_dataset = ProteinDPODataset(dpo_train_dict, pdb_idx_dict, pdb_structures)
# dpo_train_loader = DataLoader(dpo_train_dataset, batch_size=3, shuffle=True)


# make a featurize function 

def featurize(batch, device):
    B = batch['structure'].shape[0]
    L_max = max([len(x) for x in batch['aa_seq']])
    X = batch['structure'][:, :L_max, :, :].to(dtype=torch.float32, device=device)
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    S_wt = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.int32)
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    for i, seq in enumerate(batch['aa_seq']):
        S[i, :len(seq)] = np.asarray([ALPHABET.index(aa) for aa in seq], dtype=np.int32)
        mask[i, :len(seq)] = 1
        residue_idx[i, :len(seq)] = np.arange(len(seq))
    for i, seq in enumerate(batch['aa_seq_wt']):
        S_wt[i, :len(seq)] = np.asarray([ALPHABET.index(aa) for aa in seq], dtype=np.int32)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    S_wt = torch.from_numpy(S_wt).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    chain_M = mask.clone()
    chain_encoding_all = mask.clone()
    return X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt


# import torch

# def featurize(batch, device):
#     B = batch['structure'].shape[0]
#     L_max = max(len(x) for x in batch['aa_seq'])
#     X = batch['structure'][:, :L_max, :, :].to(dtype=torch.float32, device=device)
    
#     # Initialize tensors directly on the device
#     S = torch.zeros([B, L_max], dtype=torch.long, device=device)
#     S_wt = torch.zeros([B, L_max], dtype=torch.long, device=device)
#     mask = torch.zeros([B, L_max], dtype=torch.float32, device=device)
#     residue_idx = torch.full([B, L_max], -100, dtype=torch.long, device=device)
    
#     # Precompute alphabet index mapping for faster access
#     alphabet_dict = {aa: idx for idx, aa in enumerate(ALPHABET)}
    
#     for i, seq in enumerate(batch['aa_seq']):
#         seq_len = len(seq)
#         # Directly create and assign tensors on the device
#         S[i, :seq_len] = torch.tensor([alphabet_dict[aa] for aa in seq], dtype=torch.long, device=device)
#         mask[i, :seq_len] = 1
#         residue_idx[i, :seq_len] = torch.arange(seq_len, device=device)
    
#     for i, seq in enumerate(batch['aa_seq_wt']):
#         seq_len = len(seq)
#         S_wt[i, :seq_len] = torch.tensor([alphabet_dict[aa] for aa in seq], dtype=torch.long, device=device)
    
#     chain_M = mask.clone()
#     chain_encoding_all = mask.clone()
    
#     return X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt


# for batch in dpo_train_loader:
#     # print(batch['protein_name'])
#     # print(batch['aa_seq'])
#     # print(batch['WT_name'])
#     # print(batch['aa_seq_wt'])
#     # print(batch['dG_ML'])
#     # print(batch['ddG_ML'])
#     # print(batch['dG_ML_wt'])
#     # print(batch['ddG_ML_wt'])
#     # print(batch['name_wt'])
#     # print(batch['structure'])
#     X, S, mask, chain_M, residue_idx, chain_encoding_all = featurize(batch, device='cuda')
#     print(X.shape, S.shape, mask.shape, chain_M.shape, residue_idx.shape, chain_encoding_all.shape)
#     print(X)
#     print(S)
#     print(mask)
#     print(chain_M)
#     print(residue_idx)
#     print(chain_encoding_all)

#     break
