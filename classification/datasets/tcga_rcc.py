import os
import numpy as np
import pandas as pd 

import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
import hydra
import h5py
import random
import functools


def augment(coords, p_augment=0.5, p_hflip=0.5, p_vflip=0.5, p_transpose=0.5):
    """ Use sparse representation due to space. 
        We store the indices of the features and then apply some random transformations to them.
        We then use the indices to get the features from the original array """

    if random.random() < p_augment:
        size = [coords.max().item() + 1, coords.max().item() + 1] # Need a square shape for the permutation matrix to work
        sparse_feats = torch.sparse_coo_tensor(coords.permute(1,0), torch.tensor(range(len(coords))).to(torch.float), size=size).coalesce()

        # Permutation matrix with ones on the antidiagonal
        permutation_matrix = torch.sparse_coo_tensor(torch.tensor([(i, size[0] - i - 1) for i in range(size[0])]).permute(1,0), torch.tensor([1] * size[0]).to(torch.float))

        # Flip horizontally
        if random.random() < p_hflip:
            sparse_feats = torch.sparse.mm(permutation_matrix, sparse_feats).coalesce()

        # Flip vertically
        if random.random() < p_vflip:
            sparse_feats = torch.sparse.mm(sparse_feats, permutation_matrix).coalesce()

        # Flip and mirror
        if random.random() < p_transpose:
            sparse_feats = sparse_feats.transpose(1,0).coalesce()

        augmented_coordinates = sparse_feats.values().to(torch.long)
    else:
        augmented_coordinates = torch.tensor(range(len(coords))).to(torch.long)

    return augmented_coordinates


def sample_features(feats, N):
    if feats.shape[0] < N:
        return feats
    start = random.randint(0, feats.shape[0] - N)
    return feats[start:start+N]


def pad_features(feats, N):
    if feats.shape[0] < N:
        return torch.cat([feats, torch.zeros(N - feats.shape[0], feats.shape[1])])
    else:
        return feats


class TCGA_RCC(Dataset):
    def __init__(self, root_path, split, fold, cfg, num_samples=None):
        self.root_path = root_path
        self.split = split
        self.cfg = cfg
        self.fold = fold
        self.num_samples = None if num_samples == 'None' else num_samples
        self.null_token = cfg.dataset.null_token
        self.dataset = self._load_dataset(fold, split)

    def _load_dataset(self, fold: int, split: str) -> pd.DataFrame:
        dataset = pd.read_csv(os.path.join(self.root_path, 'folds', f'fold_{fold}.csv'))
        dataset = dataset[dataset.split == split]
        return dataset

    def __len__(self):
        if self.split == 'train' and self.num_samples is not None:
            if self.num_samples < 1:
                fraction = int(self.num_samples * len(self.dataset))
                return len(self.dataset[:fraction])
            else:
                return len(self.dataset[:self.num_samples])
        else:
            return len(self.dataset)
    
    
    @functools.cached_property
    def min_num_features(self):
        for index in range(len(self)):
            sample = self.dataset.iloc[index]
            file_name = sample['FileName']

            with h5py.File(os.path.join(self.root_path, 'TCGA_features', file_name), 'r') as f:
                feats = np.array(f['feats'])
                return feats.shape[0]

    def __getitem__(self, index):
        sample = self.dataset.iloc[index]
        file_name = sample['FileName']

        with h5py.File(os.path.join(self.root_path, 'TCGA_features', file_name), 'r') as f:
            feats = np.array(f['feats'])
            augmented = np.array(f['augmented'])
            coords = np.array(f['coords'])

            # Sort on coords (from top left to bottom right)
            sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
            feats = torch.tensor(feats[sorted_indices])
            coords = torch.tensor(coords[sorted_indices])
            augmented = torch.tensor(augmented[sorted_indices])
        
        if self.split == 'train' and self.cfg.dataset.augment.do_augment:
            augmented_coordinates = augment(coords, p_augment=self.cfg.dataset.augment.p_augment, 
                                            p_hflip=self.cfg.dataset.augment.p_hflip, p_vflip=self.cfg.dataset.augment.p_vflip, 
                                            p_transpose=self.cfg.dataset.augment.p_transpose)
            feats = feats[augmented_coordinates]
        
        if self.cfg.model.get('meta', {}).get('self_supervision', {}).get('use', False):
            feats = sample_features(feats, self.cfg.model.meta.self_supervision.num_features)
            feats = pad_features(feats, self.cfg.model.meta.self_supervision.num_features)
            feats = feats.to(torch.float)

            
        
        if sample['subtype'] == 'KIRC':
            label = torch.tensor([1, 0, 0]) 
        if sample['subtype'] == 'KIRP':
            label = torch.tensor([0, 1, 0]) 
        if sample['subtype'] == 'KICH':
            label = torch.tensor([0, 0, 1]) 
        label_names = ['KIRC', 'KIRP', 'KICH']

        return {'feats': feats, 'label': label, 'label_names': label_names, 'coords': coords, 'file_name': file_name}




def pad_sequence(batched_sequence):
    max_seq_length = np.max([item.shape[0] for item in batched_sequence])

    padded_sequence = []
    for sequence in batched_sequence:
        if sequence.shape[0] < max_seq_length:
            padded_sequence.append(torch.cat([sequence, torch.zeros(max_seq_length - sequence.shape[0], sequence.shape[1])]))
        else:
            padded_sequence.append(sequence)
    return padded_sequence


def tcga_rcc_collate(batch):
    label_names = [item['label_names'] for item in batch] 
    file_name = [item['file_name'] for item in batch]
    label = torch.stack([item['label'] for item in batch]).float()
    feats = torch.stack(pad_sequence([item['feats'] for item in batch])).float()
    coords = torch.stack(pad_sequence([item['coords'] for item in batch])).float()
    return {'ehr': torch.ones([1]), 'feats': feats, 'label': label, 'label_names': label_names, 'coords': coords, 'file_name': file_name}



@hydra.main(config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    train_dataset = TCGA_RCC(root_path = '/data/PATHO/', split='train', cfg=cfg)
    sample_ = train_dataset[0]

if __name__ == '__main__':
    run()     
            

        
    
        

