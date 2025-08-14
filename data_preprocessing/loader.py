import logging
import random
import math
import functools
import os

import numpy as np
import torch
import torch.utils.data as data
from .eicu.datasets import eICU_Medical_Dataset, eICU_Medical_Dataset_truncated_WO_reload, data_transforms_eicu


from data_preprocessing.utils.stats import record_net_data_stats

NORMAL_DATASET_LIST = ["eicu"]

class Data_Loader(object):

    full_data_obj_dict = {
        "eicu": eICU_Medical_Dataset,  
    }
    
    sub_data_obj_dict = {
        "eicu": eICU_Medical_Dataset_truncated_WO_reload,  
    }

    transform_dict = {
        "eicu": data_transforms_eicu,  
    }

    num_classes_dict = {
        "eicu": 2,  
    }

    image_resolution_dict = {
        "eicu": 1,  
    }

    def __init__(self, args=None, process_id=0, mode="centralized", task="centralized",
                data_efficient_load=True, dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="hetero", partition_alpha=0.5, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default", other_params={}):

        self.args = args

        # For partition
        self.process_id = process_id
        self.mode = mode
        self.task = task
        self.data_efficient_load = data_efficient_load 
        self.dirichlet_balance = dirichlet_balance
        self.dirichlet_min_p = dirichlet_min_p

        self.dataset = dataset
        self.datadir = datadir
        self.partition_method = partition_method
        self.partition_alpha = partition_alpha
        self.client_number = client_number
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_sampler = data_sampler

        self.augmentation = augmentation
        self.other_params = other_params

        self.resize = resize

        self.init_dataset_obj()

    def load_data(self):
        self.federated_medical_split()
            
        self.other_params["train_cls_local_counts_dict"] = self.train_cls_local_counts_dict
        self.other_params["client_dataidx_map"] = self.client_dataidx_map

        return self.train_data_global_num, self.test_data_global_num, self.train_data_global_dl, self.test_data_global_dl, \
               self.train_data_local_num_dict, self.test_data_local_num_dict, self.test_data_local_dl_dict, self.train_data_local_ori_dict,self.train_targets_local_ori_dict,\
               self.class_num, self.other_params

    def init_dataset_obj(self):
        self.full_data_obj = Data_Loader.full_data_obj_dict[self.dataset]
        self.sub_data_obj = Data_Loader.sub_data_obj_dict[self.dataset]
        logging.info(f"dataset augmentation: {self.augmentation}, resize: {self.resize}")
        self.transform_func = Data_Loader.transform_dict[self.dataset]
        self.class_num = Data_Loader.num_classes_dict[self.dataset]
        self.image_resolution = Data_Loader.image_resolution_dict[self.dataset]

    def get_transform(self, resize, augmentation, dataset_type, image_resolution=32):
        MEAN, STD, train_transform, test_transform = \
            self.transform_func(
                resize=resize, augmentation=augmentation, dataset_type=dataset_type, image_resolution=image_resolution)
        return MEAN, STD, train_transform, test_transform

    def load_full_data(self):
        # no transforms needed for medical data
        MEAN, STD, train_transform, test_transform = 0.0, 1.0, None, None
        
        if hasattr(self.args, 'medical_task'):
            self.full_data_obj.task = self.args.medical_task
            
        train_ds = self.full_data_obj(self.datadir, train=True, download=False, transform=train_transform)
        test_ds = self.full_data_obj(self.datadir, train=False, download=False, transform=test_transform)
        
        return train_ds, test_ds
    

    def load_sub_data(self, client_index, train_ds, test_ds):
        train_dataidxs = self.client_dataidx_map_train[client_index]
        test_dataidxs = self.client_dataidx_map_test[client_index]
        train_data_local_num = len(train_dataidxs)

        # No transforms needed for medical data
        train_ds_local = self.sub_data_obj(self.datadir, dataidxs=train_dataidxs, train=True, transform=None,
                full_dataset=train_ds)

        train_ori_data = train_ds_local.data.numpy() if isinstance(train_ds_local.data, torch.Tensor) else train_ds_local.data
        train_ori_targets = train_ds_local.targets.numpy() if isinstance(train_ds_local.targets, torch.Tensor) else train_ds_local.targets
            
        test_ds_local = self.sub_data_obj(self.datadir, dataidxs=test_dataidxs, train=False, transform=None,
                        full_dataset=train_ds)   

        test_data_local_num = len(test_ds_local)
        return train_ds_local, test_ds_local, train_ori_data, train_ori_targets, train_data_local_num, test_data_local_num

    def get_dataloader(self, train_ds, test_ds, shuffle=True, drop_last=False, train_sampler=None, num_workers=1):
        logging.info(f"shuffle: {shuffle}, drop_last:{drop_last}, train_sampler:{train_sampler} ")
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=shuffle,
                                drop_last=drop_last, sampler=train_sampler, num_workers=num_workers)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.batch_size, shuffle=True,
                                drop_last=False, num_workers=num_workers)
        
        return train_dl, test_dl

    def get_y_train_np(self, train_ds):
        """Extract labels as numpy array from medical dataset"""
        y_train = train_ds.targets
        y_train_np = np.array(y_train)
        return y_train_np

    def federated_medical_split(self):
        """hospital-based partitioning"""
        train_ds, test_ds = self.load_full_data()
        y_train_np = self.get_y_train_np(train_ds)
        
        self.train_data_global_num = y_train_np.shape[0]
        self.test_data_global_num = len(test_ds)
                
        unseen_hospital_test = getattr(self.args, 'unseen_hospital_test', False)
        target_hospital_id = getattr(self.args, 'target_hospital_id', None)
        target_hospital_list = getattr(self.args, 'target_hospital_list', [])
        auto_select_hospitals = getattr(self.args, 'auto_select_hospitals', True)
        
        from .eicu.data_loader import partition_eicu_data_by_hospital
        self.client_dataidx_map_train, self.client_dataidx_map_test, self.train_cls_local_counts_dict, excluded_hospital_data = partition_eicu_data_by_hospital(
            train_ds, self.client_number, unseen_hospital_test=unseen_hospital_test, target_hospital_id=target_hospital_id,
            target_hospital_list=target_hospital_list, auto_select_hospitals=auto_select_hospitals
        )
        self.client_dataidx_map = self.client_dataidx_map_train
        
        logging.info("train_cls_local_counts_dict = " + str(self.train_cls_local_counts_dict))
        
        # global data loaders 
        if excluded_hospital_data is not None:
            self.client_number = len(self.client_dataidx_map_train)
            excluded_hospital_ds = self.sub_data_obj(self.datadir, dataidxs=excluded_hospital_data, 
                                                   train=True, transform=None, full_dataset=train_ds)
            
            # Create training dataset excluding the target hospital
            # Combine indices from all participating clients (excluding target hospital)
            remaining_hospital_indices = []
            for client_idx in range(self.client_number):
                remaining_hospital_indices.extend(self.client_dataidx_map_train[client_idx])
            
            remaining_hospital_ds = self.sub_data_obj(self.datadir, dataidxs=remaining_hospital_indices,
                                                    train=True, transform=None, full_dataset=train_ds)
            
            # Global train: only remaining hospitals (excluding target)
            # Global test: entire excluded hospital (target hospital test)
            self.train_data_global_dl, _ = self.get_dataloader(
                remaining_hospital_ds, excluded_hospital_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers
            )
            self.test_data_global_dl, _ = self.get_dataloader(
                excluded_hospital_ds, excluded_hospital_ds,
                shuffle=False, drop_last=False, train_sampler=None, num_workers=self.num_workers
            )
            logging.info(f"Global training uses {len(remaining_hospital_indices)} samples from {self.client_number} participating hospitals")
            logging.info(f"Using entire excluded hospital {target_hospital_id} as global test set ({len(excluded_hospital_data)} samples)")
        else:
            self.train_data_global_dl, self.test_data_global_dl = self.get_dataloader(
                train_ds, test_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers
            )
        
        self.train_data_local_num_dict = dict()
        self.test_data_local_num_dict = dict()
        self.train_data_local_ori_dict = dict()
        self.train_targets_local_ori_dict = dict()
        self.test_data_local_dl_dict = dict()
        
        # Create local data for each client
        for client_index in range(self.client_number):
            train_data_local, test_data_local, train_ori_data, train_ori_targets, train_data_local_num, test_data_local_num = self.load_sub_data(client_index, train_ds, test_ds)
            
            train_data_local_dl, test_data_local_dl = self.get_dataloader(
                train_data_local, test_data_local,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers
            )
            
            self.train_data_local_num_dict[client_index] = train_data_local_num
            self.test_data_local_num_dict[client_index] = test_data_local_num
            self.test_data_local_dl_dict[client_index] = test_data_local_dl
            self.train_data_local_ori_dict[client_index] = train_ori_data
            self.train_targets_local_ori_dict[client_index] = train_ori_targets
        
        return self.train_data_local_num_dict, self.train_cls_local_counts_dict



