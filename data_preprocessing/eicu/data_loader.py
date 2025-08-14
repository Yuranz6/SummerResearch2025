import logging
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from .datasets import eICU_Medical_Dataset, eICU_Medical_Dataset_truncated_WO_reload, data_transforms_eicu

def partition_eicu_data_by_hospital(dataset, client_num, min_samples_per_hospital=10, test_size=0.2, 
                                   unseen_hospital_test=False, target_hospital_id=None,
                                   target_hospital_list=None, auto_select_hospitals=True):
    """
    partition eICU data naturally by hospital IDs with adaptive hospital selection
    
    Args:
        dataset: eICU dataset with hospital_ids attribute
        client_num: number of FL clients needed
        min_samples_per_hospital: minimum samples required per hospital 
        test_size: test split ratio within each hospital
        unseen_hospital_test: whether to use target hospital for evaluation
        target_hospital_id: specific target hospital for unseen hospital test
        target_hospital_list: list of specific hospitals to use (FedWeight-style)
        auto_select_hospitals: whether to auto-select hospitals when list not provided
        
    Returns:
        dict_users_train: {client_idx: array of training data indices}
        dict_users_test: {client_idx: array of test data indices}
        dict_users_train_cls_counts: {client_idx: {class: count}}
        excluded_hospital_data: indices of excluded target hospital (if any)
    """
    
    hospital_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        hospital_id = dataset.hospital_ids[idx]
        hospital_to_indices[hospital_id].append(idx)
    
    # Filter
    valid_hospitals = []
    for hospital_id, indices in hospital_to_indices.items():
        if len(indices) >= min_samples_per_hospital:
            valid_hospitals.append(hospital_id)
    
    logging.info(f"Found {len(valid_hospitals)} valid hospitals with >= {min_samples_per_hospital} samples")
    
    hospitals_needed = client_num + 1 if unseen_hospital_test and target_hospital_id is not None else client_num
    
    if target_hospital_list and len(target_hospital_list) > 0:
        valid_target_hospitals = [h for h in target_hospital_list if h in valid_hospitals]
        
        if len(valid_target_hospitals) >= hospitals_needed:
            selected_hospitals = valid_target_hospitals[:hospitals_needed]
        else:
            logging.warning(f"Only {len(valid_target_hospitals)} valid hospitals from target_hospital_list, need {hospitals_needed}")
            selected_hospitals = valid_target_hospitals
            # Fallback to auto-selection for remaining hospitals
            if auto_select_hospitals and len(selected_hospitals) < hospitals_needed:
                hospital_sizes = [(h, len(hospital_to_indices[h])) for h in valid_hospitals if h not in selected_hospitals]
                hospital_sizes.sort(key=lambda x: x[1], reverse=True)
                additional_needed = hospitals_needed - len(selected_hospitals)
                additional_hospitals = [h for h, _ in hospital_sizes[:additional_needed]]
                selected_hospitals.extend(additional_hospitals)
    
    elif auto_select_hospitals:
        hospital_sizes = [(h, len(hospital_to_indices[h])) for h in valid_hospitals]
        hospital_sizes.sort(key=lambda x: x[1], reverse=True)
        
        if len(valid_hospitals) >= hospitals_needed:
            selected_hospitals = [h for h, _ in hospital_sizes[:hospitals_needed]]
            logging.info(f"Selected hospitals: {selected_hospitals}")
        else:
            selected_hospitals = valid_hospitals
    else:
        raise ValueError("Either target_hospital_list must be provided or auto_select_hospitals must be True")
    excluded_hospital_data = None
    if unseen_hospital_test and target_hospital_id is not None:
            
        if target_hospital_id in selected_hospitals:
            excluded_hospital_data = hospital_to_indices[target_hospital_id]
            target_position = selected_hospitals.index(target_hospital_id)
            
            # Remove target hospital from client list (no replacement)
            selected_hospitals.remove(target_hospital_id)
            
            logging.info(f"Excluded hospital {target_hospital_id} for unseen hospital test ({len(excluded_hospital_data)} samples)")
            logging.info(f"Target hospital {target_hospital_id} was at position {target_position} in rankings")
            logging.info(f"Final hospital selection after exclusion: {selected_hospitals}")
        else:
            raise Exception('the target hospital must be in selected hospital list!')
    
    dict_users_train = {}
    dict_users_test = {}
    dict_users_train_cls_counts = {}
    
    for client_idx in range(len(selected_hospitals)):
        hospital_id = selected_hospitals[client_idx]                                                 
        hospital_indices = np.array(hospital_to_indices[hospital_id])
        hospital_targets = dataset.targets[hospital_indices]
        
        train_indices, test_indices = train_test_split(
            hospital_indices,
            test_size=test_size,
            random_state=42,
            stratify=hospital_targets
        )
        
        dict_users_train[client_idx] = train_indices
        dict_users_test[client_idx] = test_indices
        
        train_targets = dataset.targets[train_indices]
        unique_classes, counts = np.unique(train_targets, return_counts=True)
        
        class_counts = {0: 0, 1: 0}
        for cls, count in zip(unique_classes, counts):
            class_counts[int(cls)] = int(count)
            
        dict_users_train_cls_counts[client_idx] = class_counts
        
        logging.info(f"Client {client_idx} (Hospital {hospital_id}): "
                    f"Train={len(train_indices)}, Test={len(test_indices)}, "
                    f"Train class distribution: {class_counts}")
    
    return dict_users_train, dict_users_test, dict_users_train_cls_counts, excluded_hospital_data


def load_partition_eicu_medical(dataset, data_dir, partition_method, partition_alpha,
                               client_number, batch_size, logger, args=None):
    """
    Main entry point for loading and partitioning eICU data (same logic already implemented in loader.py)
    """
    
    if args and hasattr(args, 'medical_task'):
        eICU_Medical_Dataset.task = args.medical_task
    
    train_dataset = eICU_Medical_Dataset(data_dir, train=True, download=False)
    test_dataset = eICU_Medical_Dataset(data_dir, train=False, download=False)
    
    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)
    class_num = 2  # Binary classification
    
    # Partition data by hospital (natural partitioning for medical data)
    if partition_method == "hospital":
        dict_users, dict_users_train_cls_counts = partition_eicu_data_by_hospital(
            train_dataset, client_number
        )
    else:
        logging.warning(f"Partition method '{partition_method}' not optimal for medical data. "
                       f"Using random partitioning.")
        # random partitioning
        indices = np.arange(train_data_num)
        np.random.shuffle(indices)
        dict_users = {}
        dict_users_train_cls_counts = {}
        
        samples_per_client = train_data_num // client_number
        for i in range(client_number):
            start_idx = i * samples_per_client
            if i == client_number - 1:
                dict_users[i] = indices[start_idx:]
            else:
                dict_users[i] = indices[start_idx:start_idx + samples_per_client]
                
            targets = train_dataset.targets[dict_users[i]]
            class_counts = {0: 0, 1: 0}
            for cls in [0, 1]:
                class_counts[cls] = int(np.sum(targets == cls))
            dict_users_train_cls_counts[i] = class_counts
    
    # Create global data loaders
    train_data_global = data.DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False
    )
    test_data_global = data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False
    )
    
    train_data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}
    train_data_local_cls_counts_dict = {}
    
    for client_idx in range(client_number):
        dataidxs = dict_users[client_idx]
        
        train_data_local_num_dict[client_idx] = len(dataidxs)
        
        train_data_local = eICU_Medical_Dataset_truncated_WO_reload(
            data_dir, dataidxs=dataidxs, train=True,
            transform=None, full_dataset=train_dataset
        )
        
        # For test data, all clients use the same test set (following FedFed pattern)
        test_data_local = test_dataset
        
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        train_data_local_cls_counts_dict[client_idx] = dict_users_train_cls_counts[client_idx]
    
    return (train_dataset, test_dataset, train_data_num, test_data_num,
            train_data_global, test_data_global, train_data_local_num_dict,
            train_data_local_dict, test_data_local_dict, class_num,
            train_data_local_cls_counts_dict)