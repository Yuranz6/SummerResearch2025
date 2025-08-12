import logging
import math
import os
import sys
import random
from abc import  abstractmethod
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.basePS.ps_client_trainer import PSTrainer
from utils.data_utils import optimizer_to
from model.FL_VAE import *
from optim.AdamW import AdamW
from utils.tool import *
from utils.set import *
from data_preprocessing.cifar10.datasets import Dataset_Personalize, Dataset_3Types_ImageData, Dataset_3Types_MedicalData
import torchvision.transforms as transforms
from utils.log_info import log_info
from utils.randaugment4fixmatch import RandAugmentMC, Cutout, RandAugment_no_CutOut 
from loss_fn.build import FocalLoss
from utils.validation import VAEPerformanceValidator, SharedDataDistributionValidator, MixedDatasetValidator
from utils.feature_extraction import FeatureExtractor

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

class Client(PSTrainer):

    def __init__(self, client_index, train_ori_data, train_ori_targets, test_dataloader, train_data_num,
                 test_data_num, train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num):
        super().__init__(client_index, train_ori_data, train_ori_targets, test_dataloader, train_data_num,
                         test_data_num, device, args, model_trainer)
        if args.VAE == True and vae_model is not None:
            logging.info(f"client {self.client_index} VAE Moel set up")
            self.vae_model = vae_model

        self.test_dataloader = test_dataloader
        self.train_ori_data = train_ori_data  
        self.train_ori_targets = train_ori_targets
        self.train_cls_counts_dict = train_cls_counts_dict
        self.dataset_num = dataset_num

        self.local_num_iterations = math.ceil(len(self.train_ori_data) / self.args.batch_size)

# -------------------------VAE optimization tool for different client------------------------#
        self.vae_optimizer =  AdamW([
            {'params': self.vae_model.parameters()}
        ], lr=1.e-3, betas=(0.9, 0.999), weight_decay=1.e-6)
        self._construct_train_ori_dataloader()
        
        if self.args.VAE_adaptive:
            self._set_local_traindata_property()
            logging.info(self.local_traindata_property)
        
        if self.args.VAE_loss == 'focal':
            self.loss = FocalLoss(alpha=0.25, gamma=2.0)
        else: 
            self.loss = F.cross_entropy

# -------------------------Feature extraction setup------------------------#
        # self.feature_extractor = FeatureExtractor(output_dir=f"feature_analysis_client_{self.client_index}")
        # Store hospital ID if available (for visualization)
        self.hospital_id = getattr(args, 'hospital_id', None)
        
        
        
        # experiments for validating VAE's performance and quality of shared features
        self.vae_validator = VAEPerformanceValidator(device=self.device)
        self.shared_data_validator = SharedDataDistributionValidator()
        self.mixed_data_validator = MixedDatasetValidator()
        
        
    def _construct_train_ori_dataloader(self):
        pass
    
    def _attack(self,size, mean, std):  #
        rand = torch.normal(mean=mean, std=std, size=size).to(self.device)
        return rand

    def _set_local_traindata_property(self):
        class_num = len(self.train_cls_counts_dict)
        clas_counts = [ self.train_cls_counts_dict[key] for key in self.train_cls_counts_dict.keys()]
        max_cls_counts = max(clas_counts)
        if self.local_sample_number < self.dataset_num/self.args.client_num_in_total * 0.2:
            self.local_traindata_property = 1 # 1 means quantity skew is very heavy
        elif self.local_sample_number > self.dataset_num/self.args.client_num_in_total * 0.2 and max_cls_counts > self.local_sample_number * 0.7:
            self.local_traindata_property = 2 # 2 means label skew is very heavy
        else:
            self.local_traindata_property = None

# ================================= Phase 1: VAE Train =================================

    # we don't do augmented training for medical data
    # def aug_classifier_train(self, round, epoch, optimizer, aug_trainloader):
    #     self.vae_model.train()
    #     self.vae_model.training = True

    #     for batch_idx, (x, y) in enumerate(aug_trainloader):
    #         x, y, y_b, lam, mixup_index = mixup_data(x, y, alpha=self.args.VAE_alpha)
    #         x, y, y_b = x.to(self.device), y.to(self.device).view(-1, ), y_b.to(self.device).view(-1, )
    #         # x, y = Variable(x), [Variable(y), Variable(y_b)]
    #         x, y = x, [y, y_b]
    #         n_iter = round * self.args.VAE_local_epoch + epoch * len(aug_trainloader) + batch_idx
    #         optimizer.zero_grad()

    #         for name, parameter in self.vae_model.named_parameters():
    #             if 'classifier' not in name:
    #                 parameter.requires_grad = False
    #         out = self.vae_model.get_classifier()(x)

    #         loss = lam * F.cross_entropy(out, y[0]) + (1. - lam) * F.cross_entropy(out, y[1])
    #         loss.backward()
    #         optimizer.step()



    # def mosaic(self, batch_data):
    #     s = 16
    #     yc, xc = 16, 16
    #     if self.args.dataset =='fmnist':
    #         c, w, h = 1, 32, 32
    #     else:
    #         c, w, h = 3, 32, 32
    #     aug_data = torch.zeros((self.args.VAE_aug_batch_size, c, w, h))
    #     CutOut = Cutout(n_holes=1, length=16)
    #     for k in range(self.args.VAE_aug_batch_size):

    #         sample = random.sample(range(batch_data.shape[0]), 4)
    #         img4 = torch.zeros(batch_data[0].shape)

    #         left = random.randint(0, 16)
    #         up = random.randint(0, 16)

    #         for i, index in enumerate(sample):
    #             if i == 0:  # top left
    #                 x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
    #             elif i == 1:  # top right
    #                 x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
    #             elif i == 2:  # bottom left
    #                 x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
    #             elif i == 3:  # bottom right
    #                 x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
    #             img4[:, x1a:x2a, y1a:y2a] = batch_data[index][:, left:left + 16, up:up + 16]
    #         img4 = CutOut(img4)
    #         aug_data[k] = img4
    #     return aug_data

    # def aug_VAE_train(self, round, epoch, optimizer, aug_trainloader):
    #     self.vae_model.train()
    #     self.vae_model.training = True
    #     self.vae_model.requires_grad_(True)

    #     for batch_idx, (x, y) in enumerate(aug_trainloader):
    #         n_iter = round * self.args.VAE_local_epoch + epoch  * len(aug_trainloader) + batch_idx
    #         batch_size = x.size(0)
    #         if batch_size < 4:
    #             break
    #         # using mosaic data train VAE first for get a good initialize
    #         aug_data = self.mosaic(x).to(self.device)
    #         optimizer.zero_grad()

    #         if self.args.VAE_curriculum:
    #             if epoch < 100:
    #                 re = 10 * self.args.VAE_re
    #             elif epoch < 200:
    #                 re = 5 * self.args.VAE_re
    #             else:
    #                 re = self.args.VAE_re
    #         else:
    #             re = self.args.VAE_re

    #         _, _, aug_gx, aug_mu, aug_logvar, _, _, _ = self.vae_model(aug_data)
    #         aug_l1 = F.mse_loss(aug_gx, aug_data)
    #         aug_l3 = -0.5 * torch.sum(1 + aug_logvar - aug_mu.pow(2) - aug_logvar.exp())
    #         aug_l3 /= self.args.VAE_aug_batch_size * 3 * self.args.VAE_z


    #         aug_loss = re * aug_l1 + self.args.VAE_kl * aug_l3

    #         aug_loss.backward()
    #         optimizer.step()
    
    def _create_balanced_vae_dataset(self):
        """
        Create a balanced dataset for VAE training by:
        1. Using all available positive samples
        2. Downsampling negative samples to achieve target ratio
        
        Returns:
            balanced_data: numpy array of balanced features
            balanced_targets: numpy array of balanced labels
        """
        data = self.train_ori_data
        targets = self.train_ori_targets
        target_pos_ratio = self.args.VAE_target_pos_ratio
        
        pos_indices = np.where(targets == 1)[0]
        neg_indices = np.where(targets == 0)[0]
        
        n_pos = len(pos_indices)
        n_neg_available = len(neg_indices)
        
        
        n_neg_needed = int(n_pos * (1 - target_pos_ratio) / target_pos_ratio)
        
        if n_neg_needed <= n_neg_available:
            selected_neg_indices = np.random.choice(neg_indices, size=n_neg_needed, replace=False)
        else:
            selected_neg_indices = neg_indices  
            logging.warning(f"VAE balanced training: Need {n_neg_needed} neg samples but only have {n_neg_available}, using all available")
        
        balanced_indices = np.concatenate([pos_indices, selected_neg_indices])
        np.random.shuffle(balanced_indices)
        
        balanced_data = data[balanced_indices]
        balanced_targets = targets[balanced_indices]
        
        original_pos_ratio = n_pos / (n_pos + n_neg_available)
        final_pos_ratio = n_pos / len(balanced_indices)
        
        logging.info(f"VAE balanced training - Original: {n_pos} pos, {n_neg_available} neg ({original_pos_ratio:.3f} pos ratio)")
        logging.info(f"VAE balanced training - Balanced: {n_pos} pos, {len(selected_neg_indices)} neg ({final_pos_ratio:.3f} pos ratio)")
        
        return balanced_data, balanced_targets
    

    def train_whole_process(self, round, epoch, optimizer, trainloader):
        self.vae_model.train()
        self.vae_model.training = True

        loss_avg = AverageMeter()
        loss_rec = AverageMeter()
        loss_ce = AverageMeter()
        loss_entropy = AverageMeter()
        loss_kl = AverageMeter()
        top1 = AverageMeter()


        logging.info('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))

        for batch_idx, (x, y) in enumerate(trainloader):
            n_iter = round * self.args.VAE_local_epoch + epoch * len(trainloader) + batch_idx
            x, y = x.to(self.device), y.to(self.device)

            batch_size = x.size(0)


            # CAUTION! do we need to adjust reconstruction strength based on # epoch for medical data? (maybe not?)
            if epoch < 10:
                re = 5 * self.args.VAE_re
            elif epoch < 20:
                re = 2 * self.args.VAE_re
            else:
                re = self.args.VAE_re
            

            optimizer.zero_grad()
            out, hi, gx, mu, logvar, rx, rx_noise1, rx_noise2 = self.vae_model(x)
            y = y.float()
            if out.dim() > 1 and out.size(-1) == 1:
                out_binary = out.squeeze(-1)
            else:
                out_binary = out
            
            # focal loss for imbalanced data
            cross_entropy = self.loss(out_binary[:batch_size*2], y.repeat(2)) # classification CE loss on noisy performance sensitive features rx_noise
            x_ce_loss = self.loss(out_binary[batch_size*2:], y) # CE loss on original features x(bn_x)
            l1 = F.mse_loss(gx, x) # reconstruction loss on generated features (performance-robust)
            l2 = cross_entropy # CE loss on noisy performance sensitive features rx_noise
            l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            l3 /= batch_size * self.vae_model.latent_dim # KL 
            
            # sparsity regularization 
            sparsity_loss = torch.mean(torch.abs(gx))  # L1 sparsity penalty on reconstruction
            sparsity_weight = 0.2


            #  TODO: Add explicit compression constraint from Information Bottleneck
            # Implement ||rx||₂² ≤ ρ constraint from Equation (9) as in FedFed paper

            
            if self.args.VAE_adaptive:
                if self.local_traindata_property == 1 :
                    loss = 5 * re * l1 + self.args.VAE_ce * l2 + 0.5 * self.args.VAE_kl * l3 + self.args.VAE_x_ce * x_ce_loss + sparsity_weight * sparsity_loss 
                if self.local_traindata_property == 2 :
                    loss = re * l1 + 5 * self.args.VAE_ce * l2 + 5 * self.args.VAE_kl * l3 + 5 * self.args.VAE_x_ce * x_ce_loss + sparsity_weight * sparsity_loss 
                if self.local_traindata_property == None:
                    loss = re * l1 + self.args.VAE_ce * l2 + self.args.VAE_kl * l3 + self.args.VAE_x_ce * x_ce_loss + sparsity_weight * sparsity_loss 
            else: 
                loss = re * l1 + self.args.VAE_ce * l2 + self.args.VAE_kl * l3 + self.args.VAE_x_ce * x_ce_loss + sparsity_weight * sparsity_loss
            
    
            loss.backward()
            optimizer.step()


            # prec1, prec5, correct, pred, class_acc = accuracy(out[:batch_size].data, y[:batch_size].data, topk=(1, 5))
            loss_avg.update(loss.data.item(), batch_size)
            loss_rec.update(l1.data.item(), batch_size)
            loss_ce.update(cross_entropy.data.item(), batch_size)
            loss_kl.update(l3.data.item(), batch_size)
            # top1.update(prec1.item(), batch_size) # not needed

            log_info('scalar', 'client {index}:loss'.format(index=self.client_index),
                     loss_avg.avg,step=n_iter,record_tool=self.args.record_tool, 
                        wandb_record=self.args.wandb_record)


            if epoch % 10 == 0:
                if (batch_idx + 1) % 30 == 0:
                    logging.info('\r')
                    logging.info(
                        '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Loss_rec: %.4f Loss_ce: %.4f Loss_entropy: %.4f Loss_kl: %.4f Acc@1: %.3f%%'
                        % (epoch, self.args.VAE_local_epoch, batch_idx + 1,
                           len(trainloader), loss_avg.avg, loss_rec.avg, loss_ce.avg, loss_entropy.avg,
                           loss_kl.avg, top1.avg))


    def train_vae_model(self, round):
            '''Entrypoint of VAE training'''
            if self.args.dataset == 'eicu':
                self._train_vae_model_medical(round)
            else:
                self._train_vae_model_image(round)
        


        
    def _train_vae_model_medical(self, round):
        
        # Create balanced dataset if configured
        if hasattr(self.args, 'VAE_target_pos_ratio') and self.args.VAE_target_pos_ratio is not None:
            balanced_data, balanced_targets = self._create_balanced_vae_dataset()
            # Maybe should use the custom medical dataset
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(balanced_data),
                torch.LongTensor(balanced_targets)
            )
        else:
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(self.train_ori_data),
                torch.LongTensor(self.train_ori_targets)
            )
        
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.VAE_batch_size,
            shuffle=True,
            drop_last=True
        )
        
        for epoch in range(self.args.VAE_local_epoch):
            # skip augmentation steps and go directly to main training
            # QUESTION: is it necessary to augment eICU data (similar to mosaic) and pre-train VAE?
            self.train_whole_process(round, epoch, self.vae_optimizer, trainloader)
            
            if epoch % 50 == 0:
                self.test_local_vae(round, epoch, 'local')
                
        if round % 5 == 0: 
            self.validate_vae_performance(round)
            self.debug_feature_separation(round)
            


    def test_local_vae(self, round, epoch, mode):

        self.vae_model.to(self.device)
        self.vae_model.eval()
        
        test_acc_avg = AverageMeter()
        test_loss_avg = AverageMeter()
        # note, for eicu data(extremely imbalanced, 0.54 positive rate), 
        # we want to measure its AUPRC rather than accuracy
        all_preds = []
        all_targets = []
    
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                batch_size = x.size(0)
                
                output = self.vae_model.classifier_test(x)
                
                if output.dim() > 1 and output.size(-1) == 1:
                    output = output.squeeze(-1)
                
                loss = F.binary_cross_entropy_with_logits(output, y.float())
                probs = torch.sigmoid(output)
                
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(y.float().cpu().numpy())
                
                preds = (probs > 0.5).float()
                correct = (preds == y.float()).float().sum()
                accuracy = correct / batch_size * 100
                test_acc_avg.update(accuracy.item(), batch_size)
                
                test_loss_avg.update(loss.data.item(), batch_size)
        
        from sklearn.metrics import average_precision_score
        auprc = average_precision_score(all_targets, all_preds)
        logging.info("| VAE Testing Round %d Epoch %d | Test Loss: %.4f | Test Acc: %.4f | AUPRC: %.4f" %
                    (round, epoch, test_loss_avg.avg, test_acc_avg.avg, auprc))
    
    # ===================================== End of VAE Training =================================


    # ===================================== Phase 1.5: Share data step ==========================
    def generate_data_by_vae(self):
        
        data = self.train_ori_data
        targets = self.train_ori_targets
        if self.args.dataset == 'eicu':
            train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(data),
            torch.LongTensor(targets)
            )
        
            generate_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.VAE_batch_size,
                shuffle=True,
                drop_last=True
            )
        else: 
            generate_transform = transforms.Compose([])
            if self.args.dataset == 'fmnist':
                generate_transform.transforms.append(transforms.Resize(32))
            generate_transform.transforms.append(transforms.ToTensor())
            
            generate_dataset = Dataset_Personalize(data, targets, transform=generate_transform)
            generate_dataloader = torch.utils.data.DataLoader(dataset=generate_dataset, batch_size=self.args.VAE_batch_size,
                                                            shuffle=False, drop_last=False)

        self.vae_model.to(self.device)
        self.vae_model.eval()

        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(generate_dataloader):
                # distribute data to device
                x, y = x.to(self.device), y.to(self.device).view(-1, )
                _, _, gx, _, _, rx, rx_noise1, rx_noise2 = self.vae_model(x)

                batch_size = x.size(0)

                if batch_idx == 0:
                    self.local_share_data1 = rx_noise1
                    self.local_share_data2 = rx_noise2
                    self.local_share_data_y = y
                else:
                    self.local_share_data1 = torch.cat((self.local_share_data1, rx_noise1))
                    self.local_share_data2 = torch.cat((self.local_share_data2, rx_noise2))
                    self.local_share_data_y = torch.cat((self.local_share_data_y, y))
        
        # Extract and save performance-sensitive features after VAE training completion
        # if hasattr(self.args, 'extract_features') and self.args.extract_features:
        #     self.extract_rx_features_for_visualization()




    # got the classifier parameter from the whole VAE model
    def get_generate_model_classifer_para(self):
        return deepcopy(self.vae_model.get_classifier().cpu().state_dict())


    def receive_global_share_data(self, data1, data2, y):
        '''
        data: Tensor [num, C, H, W] shared by server collected all clients generated by VAE
        y: Tenosr [num, ] label corrospond to data
        '''
        self.global_share_data1 = data1.cpu()
        self.global_share_y = y.cpu()
        self.global_share_data2 = data2.cpu()
        
    def sample_iid_data_from_share_dataset(self,share_data1, share_data2, share_y, share_data_mode = 1):
        # balanced sampling: sample equal amount for each class, if have remaining, sample randomly?
        # QUESTION: should do the same for eICU dataset?
        # random.seed(random.randint(0,10000))
        if share_data_mode == 1 and share_data1 is None:
            raise RuntimeError("Not get shared data TYPE1")
        if share_data_mode == 2 and share_data2 is None:
            raise RuntimeError("Not get shared data TYPE2")
        smaple_num = self.local_sample_number
        smaple_num_each_cls = smaple_num // self.args.num_classes
        last = smaple_num - smaple_num_each_cls * self.args.num_classes 
        np_y = np.array(share_y.cpu())
        for label in range(self.args.num_classes):
            indexes = list(np.where(np_y == label)[0])
            sample = random.sample(indexes, smaple_num_each_cls)
            if label == 0:
                if share_data_mode == 1:
                    epoch_data = share_data1[sample]
                elif share_data_mode==2:
                    epoch_data = share_data2[sample]
                epoch_label = share_y[sample]
            else:
                if share_data_mode == 1:
                    epoch_data = torch.cat((epoch_data, share_data1[sample]))
                elif share_data_mode ==2:
                    epoch_data = torch.cat((epoch_data, share_data2[sample]))
                epoch_label = torch.cat((epoch_label, share_y[sample]))

        last_sample =  random.sample(range(len(share_y)), last) # sample randomly from remaining  
        if share_data_mode == 1:
            epoch_data = torch.cat((epoch_data, share_data1[last_sample]))
        elif share_data_mode == 2:
            epoch_data = torch.cat((epoch_data, share_data2[last_sample]))
        epoch_label = torch.cat((epoch_label, share_y[last_sample]))

        # statitics
        unq, unq_cnt = np.unique(np.array(epoch_label.cpu()), return_counts=True)  
        epoch_data_cls_counts_dict = {unq[i]: unq_cnt[i] for i in range(len(unq))}

        return epoch_data, epoch_label
    
    # TODO: double check its correctness
    def _sample_shared_data_proportional(self, share_data1, share_data2, share_y, share_data_mode, pos_proportion=0.1):
            '''Instead of balanced sampling as in original FedFed, we used porportional sampling for both positive and negative samples'''
            share_data = share_data1 if share_data_mode == 1 else share_data2
            target_total = min(self.local_sample_number, len(share_data))
            
            target_pos = int(target_total * pos_proportion)
            target_neg = target_total - target_pos
            
            pos_indices = torch.where(share_y == 1)[0]
            neg_indices = torch.where(share_y == 0)[0]
            
            actual_pos = min(target_pos, len(pos_indices))
            actual_neg = min(target_neg, len(neg_indices))
            
            sampled_indices = []
            
            # Sample positive 
            if actual_pos > 0:
                selected_pos = pos_indices[torch.randperm(len(pos_indices))[:actual_pos]]
                sampled_indices.append(selected_pos)
            
            # Sample negative
            if actual_neg > 0:
                selected_neg = neg_indices[torch.randperm(len(neg_indices))[:actual_neg]]
                sampled_indices.append(selected_neg)
            
            if sampled_indices:
                all_indices = torch.cat(sampled_indices)
                return share_data[all_indices], share_y[all_indices]
            else:
                return torch.empty(0, share_data.shape[1]), torch.empty(0, dtype=share_y.dtype)


    def construct_mix_dataloader(self, share_data1, share_data2, share_y, share_data_mode=1):
        if self.args.dataset == 'eicu':
            return self._construct_mix_dataloader_medical(share_data1, share_data2, share_y, share_data_mode)
        else:
            # Original image implementation 
            return self._construct_mix_dataloader_image(share_data1, share_data2, share_y, share_data_mode)
        
        
    def _construct_mix_dataloader_image(self, share_data1, share_data2, share_y, share_data_mode=1):
         # two dataloader inclue shared data from server and local origin dataloader
        train_ori_transform = transforms.Compose([])
        if self.args.dataset == 'fmnist':
            train_ori_transform.transforms.append(transforms.Resize(32))
        train_ori_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
        if self.args.dataset not in ['fmnist']:
            train_ori_transform.transforms.append(RandAugmentMC(n=3, m=10))
        train_ori_transform.transforms.append(transforms.ToTensor())
        # train_ori_transform.transforms.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))

        train_share_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #Aug_Cutout(),
        ])
        epoch_data1, epoch_label1 = self.sample_iid_data_from_share_dataset(share_data1, share_data2, share_y, share_data_mode=1)
        epoch_data2, epoch_label2 = self.sample_iid_data_from_share_dataset(share_data1, share_data2, share_y, share_data_mode=2)

        train_dataset = Dataset_3Types_ImageData(self.train_ori_data, epoch_data1,epoch_data2,
                                                 self.train_ori_targets,epoch_label1,epoch_label2,
                                                 transform=train_ori_transform,
                                                 share_transform=train_share_transform)
        self.local_train_mixed_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                                  batch_size=32, shuffle=True,
                                                                  drop_last=False)
        
    def _construct_mix_dataloader_medical(self, share_data1, share_data2, share_y, share_data_mode=1):
        """
        Constructs 3-types simultaneous training dataloader for medical data following FedFed principles.
        Uses raw shared data directly, same pattern as image implementation.
        
        Training types:
        1. Original complete features 
        2. Raw rx_noise1 (shared performance-sensitive features + noise type 1)
        3. Raw rx_noise2 (shared performance-sensitive features + noise type 2)
        """
        from torch.utils.data import DataLoader
        
        if share_data1 is not None and share_data2 is not None:
            # Sample shared rx features (raw noise types 1 & 2) proportionally
            rx_noise1_sampled, share_labels1 = self._sample_shared_data_proportional(
                share_data1, share_data2, share_y, share_data_mode=1
            )
            rx_noise2_sampled, share_labels2 = self._sample_shared_data_proportional(
                share_data1, share_data2, share_y, share_data_mode=2
            )
            
            ori_data_tensor = torch.FloatTensor(self.train_ori_data)
            ori_targets_tensor = torch.LongTensor(self.train_ori_targets)
            
            # Create 3-types dataset using raw shared data directly (no reconstruction)
            train_dataset = Dataset_3Types_MedicalData(
                ori_data=ori_data_tensor,           # Original features
                share_data1=rx_noise1_sampled,      # Raw rx_noise1 directly  
                share_data2=rx_noise2_sampled,      # Raw rx_noise2 directly
                ori_targets=ori_targets_tensor,     # targets for type 1
                share_targets1=share_labels1,       # targets for type 2
                share_targets2=share_labels2        # targets for type 3
            )
            
            self.local_train_mixed_dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=False
            )
            
            print(f"Medical 3-types FedFed training: original={len(ori_data_tensor)}, "
                  f"rx_noise1={len(rx_noise1_sampled)}, rx_noise2={len(rx_noise2_sampled)}")
        else:
            local_dataset = TensorDataset(
                torch.FloatTensor(self.train_ori_data),
                torch.LongTensor(self.train_ori_targets)
            )
            self.local_train_mixed_dataloader = DataLoader(
                local_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=False
            )
            print("No shared data available - using original data only")

        return self.local_train_mixed_dataloader 


    

    def get_local_share_data(self, noise_mode):  # noise_mode means get RXnoise2 or RXnoise2
        if self.local_share_data1 is not None and noise_mode == 1:
            return self.local_share_data1, self.local_share_data_y
        elif self.local_share_data2 is not None and noise_mode == 2:
            return self.local_share_data2, self.local_share_data_y
        else:
            raise NotImplementedError
        
    # ===================================== End of Data Sharing ===========================================

    def check_end_epoch(self):
        return (
                    self.client_timer.local_outer_iter_idx > 0 and self.client_timer.local_outer_iter_idx % self.local_num_iterations == 0)


    def move_vae_to_cpu(self):
        if str(next(self.vae_model.parameters()).device) == 'cpu':
            pass
        else:
            self.vae_model = self.vae_model.to('cpu')


    def move_to_cpu(self):
        if str(next(self.trainer.model.parameters()).device) == 'cpu':
            pass
        else:
            self.trainer.model = self.trainer.model.to('cpu')
            # optimizer_to(self.trainer.optimizer, 'cpu')

        if len(list(self.trainer.optimizer.state.values())) > 0:
            optimizer_to(self.trainer.optimizer, 'cpu')

    def move_to_gpu(self, device):
        if str(next(self.trainer.model.parameters()).device) == 'cpu':
            self.trainer.model = self.trainer.model.to(device)
        else:
            pass

        # logging.info(self.trainer.optimizer.state.values())
        if len(list(self.trainer.optimizer.state.values())) > 0:
            optimizer_to(self.trainer.optimizer, device)
    # ===================================== PHASE 2: Federated Training =====================================
    def lr_schedule(self, num_iterations, warmup_epochs):
        epochs = self.client_timer.local_outer_epoch_idx
        iterations = self.client_timer.local_outer_iter_idx
        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.trainer.lr_schedule(epochs)

    def train(self, share_data1, share_data2, share_y,
              round_idx, named_params, params_type='model',
              global_other_params=None, shared_params_for_simulation=None):
        '''
        return:
        @named_params:   all the parameters in model: {parameters_name: parameters_values}
        @params_indexes:  None
        @local_sample_number: the number of traning set in local
        @other_client_params: in FedAvg is {}
        @local_train_tracker_info:
        @local_time_info:  using this by local_time_info['local_time_info'] = {client_index:   , local_comm_round_idx:,   local_outer_epoch_idx:,   ...}
        @shared_params_for_simulation: not using in FedAvg
        '''

        if self.args.instantiate_all:
            self.move_to_gpu(self.device)
        named_params, params_indexes, local_sample_number, other_client_params, \
        shared_params_for_simulation = self.algorithm_on_train(share_data1, share_data2, share_y, round_idx,
                                                               named_params, params_type,
                                                               global_other_params,
                                                               shared_params_for_simulation)
        if self.args.instantiate_all:
            self.move_to_cpu()

        return named_params, params_indexes, local_sample_number, other_client_params, \
                shared_params_for_simulation

    def set_vae_para(self, para_dict):
        self.vae_model.load_state_dict(para_dict)

    def get_vae_para(self):
        return deepcopy(self.vae_model.cpu().state_dict())
    
    def validate_vae_performance(self, round_idx):
        """
        VAE validation based on Information Bottleneck principle
        to assess feature separation quality
        """
        if self.args.dataset != 'eicu':
            logging.info("VAE validation currently only supports medical (eicu) dataset")
            return
            
        logging.info(f"\n{'='*60}")
        logging.info(f"VAE VALIDATION - Client {self.client_index} - Round {round_idx}")
        logging.info(f"{'='*60}")
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.train_ori_data[:5000]),  
            torch.LongTensor(self.train_ori_targets[:5000])
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64, shuffle=False, drop_last=False
        )
        
        # 1. VAE Feature Separation Validation
        validation_results = self.vae_validator.validate_feature_separation(
            self.vae_model, val_dataloader, self.args
        )
        
        # 2. Mixed Dataset Validation (if shared data available)
        if hasattr(self, 'global_share_data1') and self.global_share_data1 is not None:
            # Create mixed dataloader for comparison
            mixed_dataloader = self.construct_mix_dataloader(share_data_mode=1)
            
            # Create local-only dataloader
            local_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(self.train_ori_data),
                torch.LongTensor(self.train_ori_targets)
            )
            local_dataloader = torch.utils.data.DataLoader(
                local_dataset, batch_size=self.args.batch_size, shuffle=True
            )
            
            mixing_results = self.mixed_data_validator.validate_mixing_effects(
                self, local_dataloader, mixed_dataloader, self.test_dataloader
            )
            
            # 3. Shared Data Quality Validation
            shared_data_results = self.shared_data_validator.validate_shared_data_quality(
                self.global_share_data1, self.global_share_y,
                torch.FloatTensor(self.train_ori_data), torch.LongTensor(self.train_ori_targets)
            )
            
            return {
                'vae_separation': validation_results,
                'mixing_effects': mixing_results,
                'shared_data_quality': shared_data_results
            }
        else:
            logging.info("No shared data available yet - skipping shared data validation")
            return {'vae_separation': validation_results}
    
    def validate_client_heterogeneity(self, all_clients_data):
        """
        No longer used 
        """
        client_indices = list(all_clients_data.keys())
        self.shared_data_validator.analyze_client_heterogeneity(all_clients_data, client_indices)
    
    def debug_feature_separation(self, round_idx):
        """
        generated feature debug
        """
        if self.args.dataset != 'eicu':
            return
            
        logging.info(f"\n{'='*60}")
        logging.info(f"FEATURE SEPARATION DEBUG - Client {self.client_index} - Round {round_idx}")
        logging.info(f"{'='*60}")
        
        self.vae_model.eval()
        self.vae_model.to(self.device)
        
        sample_data = torch.FloatTensor(self.train_ori_data[:50])  
        sample_targets = torch.LongTensor(self.train_ori_targets[:50])
        
        with torch.no_grad():
            sample_data = sample_data.to(self.device)
            
            out, hi, xi, mu, logvar, rx, rx_noise1, rx_noise2 = self.vae_model(sample_data)
            
            x_original = sample_data.cpu()
            xi_robust = xi.cpu()  
            rx_sensitive = rx.cpu()  
            
            logging.info(f"Sample shape: {x_original.shape}")
            logging.info(f"Input data range: [{x_original.min():.4f}, {x_original.max():.4f}]")
            logging.info(f"Performance-robust (xi) range: [{xi_robust.min():.4f}, {xi_robust.max():.4f}]")
            logging.info(f"Performance-sensitive (rx) range: [{rx_sensitive.min():.4f}, {rx_sensitive.max():.4f}]")
            
            x_mean = x_original.mean(dim=0)
            xi_mean = xi_robust.mean(dim=0)
            rx_mean = rx_sensitive.mean(dim=0)
            
            logging.info(f"Original features mean: {x_mean[:10]}...")  
            logging.info(f"Performance-robust mean: {xi_mean[:10]}...")
            logging.info(f"Performance-sensitive mean: {rx_mean[:10]}...")
            
            # Check reconstruction quality of xi
            reconstruction_mse = torch.nn.functional.mse_loss(xi_robust, x_original)
            logging.info(f"Reconstruction MSE (xi vs x): {reconstruction_mse.item():.6f}")
            
            # feature norms
            x_norm = torch.norm(x_original, dim=1).mean()
            xi_norm = torch.norm(xi_robust, dim=1).mean()
            rx_norm = torch.norm(rx_sensitive, dim=1).mean()
            
            logging.info(f"L2 norms - Original: {x_norm:.4f}, xi: {xi_norm:.4f}, rx: {rx_norm:.4f}")
            logging.info(f"Compression ratio ||rx||/||x||: {(rx_norm/x_norm):.4f}")
            
            x_squared_norm = (torch.norm(x_original, dim=1) ** 2).mean()
            rx_squared_norm = (torch.norm(rx_sensitive, dim=1) ** 2).mean()
            compression_ratio_squared = rx_squared_norm / x_squared_norm
            logging.info(f"Squared norm compression ratio ||rx||²/||x||²: {compression_ratio_squared:.4f} (target: ≤0.25)")
            
            reconstruction_check = xi_robust + rx_sensitive
            equation_error = torch.nn.functional.mse_loss(reconstruction_check, x_original)
            logging.info(f"Equation check |x - (xi + rx)|: {equation_error.item():.8f}")
            
            logging.info("\n--- Feature Distribution Analysis ---")
            logging.info(f"Original (x) - Mean: {x_original.mean():.4f}, Std: {x_original.std():.4f}")
            logging.info(f"Robust (xi) - Mean: {xi_robust.mean():.4f}, Std: {xi_robust.std():.4f}")
            logging.info(f"Sensitive (rx) - Mean: {rx_sensitive.mean():.4f}, Std: {rx_sensitive.std():.4f}")
            
            x_flat = x_original.flatten()
            xi_flat = xi_robust.flatten()
            rx_flat = rx_sensitive.flatten()
            
            correlation_x_xi = torch.corrcoef(torch.stack([x_flat, xi_flat]))[0, 1]
            correlation_x_rx = torch.corrcoef(torch.stack([x_flat, rx_flat]))[0, 1]
            correlation_xi_rx = torch.corrcoef(torch.stack([xi_flat, rx_flat]))[0, 1]
            
            logging.info(f"\n--- Correlation Analysis ---")
            logging.info(f"Corr(x, xi): {correlation_x_xi:.4f} (should be high - xi reconstructs x)")
            logging.info(f"Corr(x, rx): {correlation_x_rx:.4f} (residual correlation)")
            logging.info(f"Corr(xi, rx): {correlation_xi_rx:.4f} (should be low - orthogonal)")
            
            logging.info(f"\n--- Sample 0 Analysis (Target: {sample_targets[0].item()}) ---")
            sample_0_x = x_original[0]
            sample_0_xi = xi_robust[0]
            sample_0_rx = rx_sensitive[0]
            
            logging.info(f"{'='*60}\n")

    @abstractmethod
    def algorithm_on_train(self, share_data1, share_data2, share_y,round_idx, 
                           named_params, params_type='model',
                           global_other_params=None,
                           shared_params_for_simulation=None):
        named_params, params_indexes, local_sample_number, other_client_params = None, None, None, None
        return named_params, params_indexes, local_sample_number, other_client_params, shared_params_for_simulation