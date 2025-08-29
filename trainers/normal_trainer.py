import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from utils.data_utils import (
    get_named_data,
    get_all_bn_params,
    check_device
)
from trainers.averager import Averager
from utils.set import *
from utils.log_info import *
from utils.tool import *

class NormalTrainer(object):
    def __init__(self, model, device, criterion, optimizer, lr_scheduler, args, **kwargs):
        


        if kwargs['role'] == 'server':
            if "server_index" in kwargs:
                self.server_index = kwargs["server_index"]
            else:
                self.server_index = args.server_index
            self.client_index = None
            self.index = self.server_index

        elif kwargs['role'] == 'client':
            if "client_index" in kwargs:
                self.client_index = kwargs["client_index"]
            else:
                self.client_index = args.client_index
            self.server_index = None
            self.index = self.client_index
        else:
            raise NotImplementedError

        self.role = kwargs['role']

        self.args = args
        self.model = model

        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer


        self.param_groups = self.optimizer.param_groups  

        self.named_parameters = list(self.model.named_parameters())  # tuple [(name,param),(),...,()]
        if len(self.named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                    in sorted(self.named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'noname.%s' % i
                                    for param_group in self.param_groups
                                    for i, v in enumerate(param_group['params'])}

        self.averager = Averager(self.args, self.model) # it doesn't matter

        self.lr_scheduler = lr_scheduler

    def epoch_init(self):
        pass

    def epoch_end(self):
        pass

    def track(self, tracker, summary_n_samples, model, loss, end_of_epoch,
            checkpoint_extra_name="centralized",
            things_to_track=[]):
        pass

    def update_state(self, **kwargs):
        # This should be called begin the training of each epoch.
        self.update_loss_state(**kwargs)


    def get_model_named_modules(self):
        return dict(self.model.cpu().named_modules())


    def get_model(self):
        return self.model


    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        # for name, param in model_parameters.items():
        #     logging.info(f"Getting params as model_parameters: name:{name}, shape: {param.shape}")
        self.model.load_state_dict(model_parameters)



    def set_feature_align_means(self, feature_align_means):
        self.feature_align_means = feature_align_means
        self.align_feature_loss.feature_align_means = feature_align_means

    def get_feature_align_means(self):
        return self.feature_align_means

    def get_model_bn(self):
        all_bn_params = get_all_bn_params(self.model)
        return all_bn_params

# got it Batch_Normalization set
    def set_model_bn(self, all_bn_params):
        # logging.info(f"all_bn_params.keys(): {all_bn_params.keys()}")
        # for name, params in all_bn_params.items():
            # logging.info(f"name:{name}, params.shape: {params.shape}")
        for module_name, module in self.model.named_modules():
            if type(module) is nn.BatchNorm2d:
                # logging.info(f"module_name:{module_name}, params.norm: {module.weight.data.norm()}")
                module.weight.data = all_bn_params[module_name+".weight"] 
                module.bias.data = all_bn_params[module_name+".bias"] 
                module.running_mean = all_bn_params[module_name+".running_mean"] 
                module.running_var = all_bn_params[module_name+".running_var"] 
                module.num_batches_tracked = all_bn_params[module_name+".num_batches_tracked"] 

#  `mode` choices: ['MODEL', 'GRAD', 'MODEL+GRAD']
    def get_model_grads(self):
        named_mode_data = get_named_data(self.model, mode='GRAD', use_cuda=True)
        # logging.info(f"Getting grads as named_grads: {named_grads}")
        return named_mode_data

    def set_grad_params(self, named_grads):
        # pass
        self.model.train()
        self.optimizer.zero_grad()
        for name, parameter in self.model.named_parameters():
            parameter.grad.copy_(named_grads[name].data.to(self.device)) # 把改name的grad复制过来


    def clear_grad_params(self):
        self.optimizer.zero_grad()

    def update_model_with_grad(self):
        self.model.to(self.device)
        self.optimizer.step()

    def get_optim_state(self):
        return self.optimizer.state


    def clear_optim_buffer(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()


    def lr_schedule(self, progress):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(progress)
        else:
            logging.info("No lr scheduler...........")


    def warmup_lr_schedule(self, iterations):
        if self.lr_scheduler is not None:
            self.lr_scheduler.warmup_step(iterations)



    def get_train_batch_data(self, train_dataloader):
        try:
            train_batch_data = self.train_local_iter.next()
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(train_dataloader)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data

    def train_mix_dataloader(self, epoch, trainloader, device, **kwargs):
        if self.args.dataset == 'eicu':
            return self._train_mix_dataloader_medical(epoch, trainloader, device, **kwargs)
        else:
            # Original image training with three data sources
            return self._train_mix_dataloader_image(epoch, trainloader, device, **kwargs)
    
    def train_dataloader(self, epoch, trainloader, device, **kwargs):
        """Training method for standard federated learning without VAE data mixing"""
        self.model.to(device)
        self.model.train()
        self.model.training = True

        loss_avg = AverageMeter()
        acc = AverageMeter()

        if self.args.dataset != 'eicu': # part below is adapted for medical only, img data not adapted
          raise NotImplementedError

        # logging.info('\n=> Training Epoch #%d, LR=%.4f' % (epoch, self.optimizer.param_groups[0]['lr']))
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            self.optimizer.zero_grad()

            out = self.model(x)
            
            if out.dim() > 1 and out.size(-1) == 1:
                out = out.squeeze(-1)
            self.criterion = F.binary_cross_entropy_with_logits

            loss = self.criterion(out, y.float())

            # ========================FedProx=====================#
            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                
                for name, param in self.model.named_parameters():
                    prev_param = previous_model[name].data.to(device)  
                    param_diff_squared = torch.norm(param - prev_param, p=2)**2  # L2 norm squared
                    fed_prox_reg += (self.args.fedprox_mu / 2) * param_diff_squared
                
                loss += fed_prox_reg
            # ========================FedProx=====================#

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(out)
                predictions = (probs > 0.5).float()
                correct = (predictions == y.float()).float().sum()
                accuracy = correct / batch_size * 100
            
            loss_avg.update(loss.data.item(), batch_size)
            acc.update(accuracy.item(), batch_size)
            
            # if (batch_idx + 1) % 30 == 0:
            #     logging.info('| Epoch [%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            #                 (epoch, batch_idx + 1, len(trainloader), loss_avg.avg, acc.avg))


    def _train_mix_dataloader_medical(self, epoch, trainloader, device, **kwargs):
       
        self.model.to(device)
        self.model.train()
        
        loss_avg = AverageMeter()
        acc = AverageMeter()
        
        # logging.info('\n=> Training Epoch #%d, LR=%.4f' % 
        #             (epoch, self.optimizer.param_groups[0]['lr']))
        
        for batch_idx, batch_data in enumerate(trainloader):
            if len(batch_data) == 4: # SINGLE XS VERSION - 2-types augmentation
                x1, x2, y1, y2 = batch_data
                x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
                
                X = torch.cat((x1, x2))  # Only original + rx_noise1
                y = torch.cat((y1, y2))
                batch_size = x1.size(0)  
                
            elif len(batch_data) == 6: # 3-types augmentation (backward compatibility)
                x1, x2, x3, y1, y2, y3 = batch_data
                x1, x2, x3, y1, y2, y3 = x1.to(device), x2.to(device), x3.to(device), \
                                        y1.to(device), y2.to(device), y3.to(device)
                
                X = torch.cat((x1, x2, x3))
                y = torch.cat((y1, y2, y3))
                batch_size = x1.size(0)  
                
            elif len(batch_data) == 2:  # Standard (features, targets) format
                features, targets = batch_data
                X, y = features.to(device), targets.to(device)
                batch_size = X.size(0)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
            
            self.optimizer.zero_grad()
            
            out = self.model(X)

            if out.dim() > 1 and out.size(-1) == 1:
                out = out.squeeze(-1)
            
            self.criterion = F.binary_cross_entropy_with_logits
            loss = self.criterion(out, y.float())
            
            # FedProx regularization
            # if self.args.fedprox:
            #     fed_prox_reg = 0.0
            #     previous_model = kwargs.get("previous_model", {})
            #     for name, param in self.model.named_parameters():
            #         if name in previous_model:
            #             fed_prox_reg += ((self.args.fedprox_mu / 2) * 
            #                             torch.norm((param - previous_model[name].data.to(device)))**2)
            #     loss += fed_prox_reg
            
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                probs = torch.sigmoid(out)
                predictions = (probs > 0.5).float()
                correct = (predictions == y).float().sum()
                accuracy = correct / batch_size * 100
            
            loss_avg.update(loss.data.item(), batch_size)
            acc.update(accuracy.item(), batch_size)
            
            if (batch_idx + 1) % 50 == 0:
                logging.info('| Epoch [%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
                            (epoch, batch_idx + 1, len(trainloader), loss_avg.avg, acc.avg))
        return loss_avg.avg # don't need to return


    # ========================== Test on Server ==========================#
    
    def test_on_server_for_round(self, round, testloader, device):
        return self._test_medical_auprc(round, testloader, device)
        
    
        
    def _test_medical_auprc(self, round, testloader, device):

        import sklearn
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        self.model.to(device)
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(testloader):
                features, targets = features.to(device), targets.to(device)
                
                out = self.model(features)
                targets_float = targets.float()
                
                if out.dim() > 1 and out.size(-1) == 1:
                    out = out.squeeze(-1)
                
                loss = F.binary_cross_entropy_with_logits(out, targets_float)
                total_loss += loss.item() * features.size(0)
                
                probs = torch.sigmoid(out)
                
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(targets_float.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        auprc = average_precision_score(all_targets, all_preds)
        auc_roc = roc_auc_score(all_targets, all_preds)
        
        predictions = (all_preds > 0.5).astype(float)
        accuracy = (predictions == all_targets).mean() * 100
        
        avg_loss = total_loss / len(testloader.dataset)
        
        logging.info('| Test Round: %d | Loss: %.4f | Accuracy: %.2f%% | AUPRC: %.4f | AUC-ROC: %.4f' %
                    (round, avg_loss, accuracy, auprc, auc_roc))
        
        return auprc
    