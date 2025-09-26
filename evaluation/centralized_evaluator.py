import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .bootstrap_evaluator import BootstrapEvaluator
from .metrics import MedicalMetrics
from model.build import create_model
from trainers.build import create_trainer
from utils.set import set_random_seed

class CentralizedEvaluator:
    """
    centralized training baseline (gold standard/upper bound)
    """

    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        self.bootstrap_evaluator = BootstrapEvaluator(args, device)

    def _combine_training_data(self, train_data_loaders):
        all_x = []
        all_y = []

        logging.info("Combining training data from all hospitals")

        for client_idx, train_loader in enumerate(train_data_loaders):
            client_x = []
            client_y = []

            for batch_x, batch_y in train_loader:
                client_x.append(batch_x)
                client_y.append(batch_y)

            if len(client_x) > 0:
                client_data = torch.cat(client_x, dim=0)
                client_targets = torch.cat(client_y, dim=0)

                all_x.append(client_data)
                all_y.append(client_targets)

                logging.info(f"  Hospital {client_idx}: {client_data.shape[0]} samples")


        combined_x = torch.cat(all_x, dim=0)
        combined_y = torch.cat(all_y, dim=0)

        logging.info(f"Total centralized training data: {combined_x.shape[0]} samples, {combined_x.shape[1]} features")
        logging.info(f"Class distribution: {combined_y.float().mean():.3f} positive rate")

        combined_dataset = TensorDataset(combined_x, combined_y)
        combined_dataloader = DataLoader(
            combined_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1
        )

        return combined_dataloader

    def _combine_training_data(self, train_data_local_ori_dict, train_targets_local_ori_dict):

        all_x = []
        all_y = []

        logging.info("Combining training data from all hospitals for centralized training")

        for client_idx in sorted(train_data_local_ori_dict.keys()):
            if client_idx in train_data_local_ori_dict and client_idx in train_targets_local_ori_dict:
                train_data = train_data_local_ori_dict[client_idx]
                train_targets = train_targets_local_ori_dict[client_idx]

                train_data_tensor = torch.FloatTensor(train_data)
                train_targets_tensor = torch.LongTensor(train_targets)

                all_x.append(train_data_tensor)
                all_y.append(train_targets_tensor)

                logging.info(f"  Hospital {client_idx}: {train_data_tensor.shape[0]} samples")

        combined_x = torch.cat(all_x, dim=0)
        combined_y = torch.cat(all_y, dim=0)

        logging.info(f"Total centralized training data: {combined_x.shape[0]} samples, {combined_x.shape[1]} features")
        logging.info(f"Class distribution: {combined_y.float().mean():.3f} positive rate")

        combined_dataset = TensorDataset(combined_x, combined_y)
        combined_dataloader = DataLoader(
            combined_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1
        )

        return combined_dataloader

    def train_centralized_model(self, train_data_local_ori_dict, train_targets_local_ori_dict, validation_data, target_hospital_data):
        set_random_seed(self.args.seed)

        centralized_model = create_model(
            self.args,
            model_name=self.args.model,
            output_dim=self.args.model_output_dim,
            device=self.device
        )

        model_trainer = create_trainer(
            self.args, self.device, centralized_model,
            class_num=2,
            client_index=0, role='client'  
        )

        combined_dataloader = self._combine_training_data(train_data_local_ori_dict, train_targets_local_ori_dict)

        centralized_epochs = 300

        logging.info(f"Starting centralized training for {centralized_epochs} epochs")
        logging.info(f"  Batch size: {self.args.batch_size}")
        logging.info(f"  Learning rate: {self.args.lr}")
        logging.info(f"  Total batches per epoch: {len(combined_dataloader)}")

        for epoch in range(centralized_epochs):
            if epoch % 10 == 0 or epoch == centralized_epochs - 1:
                logging.info(f"Centralized training epoch {epoch + 1}/{centralized_epochs}")

            original_fedprox = self.args.fedprox
            self.args.fedprox = False

            model_trainer.train_dataloader(
                epoch, combined_dataloader, self.device
            )

            self.args.fedprox = original_fedprox

        return centralized_model

    def evaluate_with_bootstrap(self, model, target_hospital_data, target_hospital_id):

        logging.info(f"Starting bootstrap evaluation for Centralized on hospital {target_hospital_id}")

        prepared_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )

        results = self.bootstrap_evaluator.run_bootstrap_evaluation(
            model, prepared_data, 'centralized', target_hospital_id
        )

        return results

    def run_complete_evaluation(self, train_data_loaders, target_hospital_data, target_hospital_id, existing_data=None):
        validation_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )

        if existing_data is not None and 'train_data_local_ori_dict' in existing_data:
            centralized_model = self.train_centralized_model(
                existing_data['train_data_local_ori_dict'],
                existing_data['train_targets_local_ori_dict'],
                validation_data, target_hospital_data
            )
        else:
            raise ValueError("existing_data with raw data dictionaries is required for centralized training")


        results = self.evaluate_with_bootstrap(
            centralized_model, target_hospital_data, target_hospital_id
        )

        return centralized_model, results