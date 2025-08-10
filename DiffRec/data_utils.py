import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

def create_user_groups(train_data, n_groups=2):
    """
    Create simple binary user groups: active (above median) vs inactive (below median)
    """
    user_interactions = np.array(train_data.sum(axis=1)).flatten()
    median_interactions = np.median(user_interactions)
    
    # Binary classification: 0 = inactive (below median), 1 = active (above median)
    user_groups = (user_interactions > median_interactions).astype(int)
    
    # Calculate statistics
    inactive_count = np.sum(user_groups == 0)
    active_count = np.sum(user_groups == 1)
    
    print(f"📊 Binary User Groups Created:")
    print(f"  Inactive Users (Group 0): {inactive_count} users, ≤{median_interactions:.1f} interactions")
    print(f"  Active Users (Group 1): {active_count} users, >{median_interactions:.1f} interactions")
    
    return user_groups, {
        'inactive_count': inactive_count,
        'active_count': active_count,
        'median_interactions': median_interactions
    }

def data_load(train_path, valid_path, test_path, create_conditions=True):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    # Create user groups and conditions if requested
    user_groups = None
    group_stats = None
    if create_conditions:
        user_groups, group_stats = create_user_groups(train_data)
        
        # Print group statistics (already printed in create_user_groups)
        pass
    
    return train_data, valid_y_data, test_y_data, n_user, n_item, user_groups, group_stats


class DataDiffusion(Dataset):
    def __init__(self, data, user_groups=None):
        self.data = data
        self.user_groups = user_groups
        
    def __getitem__(self, index):
        item = self.data[index]
        if self.user_groups is not None:
            # Return both the interaction data and user group
            user_group = self.user_groups[index]
            # Convert to one-hot encoding (2 groups: 0, 1)
            user_group_onehot = torch.zeros(2, dtype=torch.float32)
            user_group_onehot[user_group] = 1.0
            return item, user_group_onehot
        return item
        
    def __len__(self):
        return len(self.data)
