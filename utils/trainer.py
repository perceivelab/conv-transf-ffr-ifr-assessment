from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils

class Trainer:
    ''' Class to train the classifier '''
    def __init__(self, net, class_weights, optim, enable_multibranch_ffr, enable_multibranch_ifr, multibranch_loss_weight, scaler=None):
        self.enable_multibranch_ffr = enable_multibranch_ffr
        self.enable_multibranch_ifr = enable_multibranch_ifr
        self.multibranch_loss_weight = multibranch_loss_weight
        # Store model
        self.net = net
        # Store optimizer
        self.optim = optim
        # Create Loss
        self.criterion_label = nn.CrossEntropyLoss(weight = class_weights)
        if self.enable_multibranch_ffr:
            self.criterion_ffr = nn.L1Loss(reduction='sum') # reduction='sum' - pay attention to balance total loss in relation of batch size
        if self.enable_multibranch_ifr:
            self.criterion_ifr = nn.L1Loss(reduction='sum') # reduction='sum' - pay attention to balance total loss in relation of batch size
        # CUDA AMP
        self.scaler = scaler

    def forward_batch(self, imgs_3d, labels, FFRs, iFRs, split):
        ''' send a batch to net and backpropagate '''
        def forward_batch_part():
            # Set network mode
            if split == 'train':
                self.net.train()
                torch.set_grad_enabled(True)   
            else:
                self.net.eval()
                torch.set_grad_enabled(False)
            
            if self.scaler is None:
                # foward pass
                out = self.net(imgs_3d)
                
                if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
                    predicted_labels_logits, predicted_FFRs, predicted_iFRs = out
                elif self.enable_multibranch_ffr:
                    predicted_labels_logits, predicted_FFRs = out
                elif self.enable_multibranch_ifr:
                    predicted_labels_logits, predicted_iFRs = out
                else:
                    predicted_labels_logits = out

                # compute loss label
                loss_labels = self.criterion_label(predicted_labels_logits, labels)
                if self.enable_multibranch_ffr or self.enable_multibranch_ifr:
                    loss = self.multibranch_loss_weight['label_weight']*loss_labels
                else:
                    loss = loss_labels

                if self.enable_multibranch_ffr:
                    predicted_FFRs = predicted_FFRs.squeeze(1)
                    # ignore missing target
                    FFRs_corrected = FFRs.clone().detach()
                    if len(FFRs) == 1: # monai fix bug with len list index=1
                        if torch.isnan(FFRs).item():
                            FFRs_corrected = predicted_FFRs.detach().type(FFRs.dtype)
                    else:
                        FFRs_corrected[torch.isnan(FFRs).tolist()] = predicted_FFRs[torch.isnan(FFRs).tolist()].detach().type(FFRs.dtype)
                    # compute loss
                    loss_FFRs = self.criterion_ffr(predicted_FFRs, FFRs_corrected)
                    loss += self.multibranch_loss_weight['FFR_weight']*loss_FFRs

                if self.enable_multibranch_ifr:
                    predicted_iFRs = predicted_iFRs.squeeze(1)
                    # ignore missing target
                    iFRs_corrected = iFRs.clone().detach()
                    if len(iFRs) == 1: # monai fix bug with len list index=1
                        if torch.isnan(iFRs).item():
                            iFRs_corrected = predicted_iFRs.detach().type(iFRs.dtype)
                    else:
                        iFRs_corrected[torch.isnan(iFRs).tolist()] = predicted_iFRs[torch.isnan(iFRs).tolist()].detach().type(iFRs.dtype)
                    # compute loss
                    loss_iFRs = self.criterion_ifr(predicted_iFRs, iFRs_corrected)
                    loss += self.multibranch_loss_weight['iFR_weight']*loss_iFRs
            else:
                with torch.cuda.amp.autocast():
                    # foward pass
                    out = self.net(imgs_3d)

                    if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
                        predicted_labels_logits, predicted_FFRs, predicted_iFRs = out
                    elif self.enable_multibranch_ffr:
                        predicted_labels_logits, predicted_FFRs = out
                    elif self.enable_multibranch_ifr:
                        predicted_labels_logits, predicted_iFRs = out
                    else:
                        predicted_labels_logits = out

                    # compute loss label
                    loss_labels = self.criterion_label(predicted_labels_logits, labels)
                    if self.enable_multibranch_ffr or self.enable_multibranch_ifr:
                        loss = self.multibranch_loss_weight['label_weight']*loss_labels
                    else:
                        loss = loss_labels

                    if self.enable_multibranch_ffr:
                        predicted_FFRs = predicted_FFRs.squeeze(1)
                        # ignore missing target
                        FFRs_corrected = FFRs.clone().detach()
                        if len(FFRs) == 1: # monai fix bug with len list index=1
                            if torch.isnan(FFRs).item():
                                FFRs_corrected = predicted_FFRs.detach().type(FFRs.dtype)
                        else:
                            FFRs_corrected[torch.isnan(FFRs).tolist()] = predicted_FFRs[torch.isnan(FFRs).tolist()].detach().type(FFRs.dtype)
                        # compute loss
                        loss_FFRs = self.criterion_ffr(predicted_FFRs, FFRs_corrected)
                        loss += self.multibranch_loss_weight['FFR_weight']*loss_FFRs

                    if self.enable_multibranch_ifr:
                        predicted_iFRs = predicted_iFRs.squeeze(1)
                        # ignore missing target
                        iFRs_corrected = iFRs.clone().detach()
                        if len(iFRs) == 1: # monai fix bug with len list index=1
                            if torch.isnan(iFRs).item():
                                iFRs_corrected = predicted_iFRs.detach().type(iFRs.dtype)
                        else:
                            iFRs_corrected[torch.isnan(iFRs).tolist()] = predicted_iFRs[torch.isnan(iFRs).tolist()].detach().type(iFRs.dtype)
                        # compute loss
                        loss_iFRs = self.criterion_ifr(predicted_iFRs, iFRs_corrected)
                        loss += self.multibranch_loss_weight['iFR_weight']*loss_iFRs
            
            # calculate label predicted and scores
            _, predicted_labels = torch.max(predicted_labels_logits.data, 1)
            predicted_scores = predicted_labels_logits.data.clone().detach().cpu()

            if split == 'train':
                #zero the gradient
                self.optim.zero_grad()

                # backpropagate
                if self.scaler is None:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()
            
            if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
                return loss, predicted_labels, predicted_scores, loss_labels, loss_FFRs, loss_iFRs, predicted_FFRs, predicted_iFRs
            elif self.enable_multibranch_ffr:
                return loss, predicted_labels, predicted_scores, loss_labels, loss_FFRs, predicted_FFRs
            elif self.enable_multibranch_ifr:
                return loss, predicted_labels, predicted_scores, loss_labels, loss_iFRs, predicted_iFRs
            else:
                return loss, predicted_labels, predicted_scores

        if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
            loss, predicted_labels, predicted_scores, loss_labels, loss_FFRs, loss_iFRs, predicted_FFRs, predicted_iFRs = forward_batch_part()
        elif self.enable_multibranch_ffr:
            loss, predicted_labels, predicted_scores, loss_labels, loss_FFRs, predicted_FFRs = forward_batch_part()
        elif self.enable_multibranch_ifr:
            loss, predicted_labels, predicted_scores, loss_labels, loss_iFRs, predicted_iFRs = forward_batch_part()
        else:
            loss, predicted_labels, predicted_scores = forward_batch_part()

        if not isinstance(self.optim,torch.optim.LBFGS):
            if split == 'train':
                # update weights (and scaler if exists)
                if self.scaler is None:
                    self.optim.step()
                else:
                    self.scaler.step(self.optim)
                    self.scaler.update()
        else:
            if split == 'train':
                # update weights (and scaler if exists)
                if self.scaler is None:
                    self.optim.step(forward_batch_part)
                else:
                    self.scaler.step(self.optim, forward_batch_part)
                    self.scaler.update()
        
        # metrics
        metrics = {}
        metrics['loss'] = loss.item()
        
        if self.enable_multibranch_ffr or self.enable_multibranch_ifr:
            metrics['loss_label'] = loss_labels.item()
        if self.enable_multibranch_ffr:
            metrics['loss_FFR'] = loss_FFRs.item()
        if self.enable_multibranch_ifr:
            metrics['loss_iFR'] = loss_iFRs.item()

        if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
            predicted = predicted_labels, predicted_scores, predicted_FFRs, predicted_iFRs
        elif self.enable_multibranch_ffr:
            predicted = predicted_labels, predicted_scores, predicted_FFRs
        elif self.enable_multibranch_ifr:
            predicted = predicted_labels, predicted_scores, predicted_iFRs
        else:
            predicted = predicted_labels, predicted_scores

        return metrics, predicted

    def forward_batch_testing(net, imgs_3d, FFRs, iFRs, enable_multibranch_ffr, enable_multibranch_ifr, scaler=None):
        ''' send a batch to net and backpropagate '''
        # Set network mode
        net.eval()
        torch.set_grad_enabled(False)
        
        if scaler is None:
            # foward pass
            out = net(imgs_3d)

            if enable_multibranch_ffr and enable_multibranch_ifr:
                    predicted_labels_logits, predicted_FFRs, predicted_iFRs = out
            elif enable_multibranch_ffr:
                predicted_labels_logits, predicted_FFRs = out
            elif enable_multibranch_ifr:
                predicted_labels_logits, predicted_iFRs = out
            else:
                predicted_labels_logits = out
            
            if enable_multibranch_ffr:
                predicted_FFRs = predicted_FFRs.squeeze(1)
                # ignore missing target
                FFRs_corrected = FFRs.clone().detach()
                FFRs_corrected[torch.isnan(FFRs).tolist()] = predicted_FFRs[torch.isnan(FFRs).tolist()].detach().type(FFRs.dtype)
            
            if enable_multibranch_ifr:
                predicted_iFRs = predicted_iFRs.squeeze(1)
                # ignore missing target
                iFRs_corrected = iFRs.clone().detach()
                iFRs_corrected[torch.isnan(iFRs).tolist()] = predicted_iFRs[torch.isnan(iFRs).tolist()].detach().type(iFRs.dtype)
        else:
            with torch.cuda.amp.autocast():
                # foward pass
                out = net(imgs_3d)
                if enable_multibranch_ffr and enable_multibranch_ifr:
                    predicted_labels_logits, predicted_FFRs, predicted_iFRs = out
                elif enable_multibranch_ffr:
                    predicted_labels_logits, predicted_FFRs = out
                elif enable_multibranch_ifr:
                    predicted_labels_logits, predicted_iFRs = out
                else:
                    predicted_labels_logits = out
                
                if enable_multibranch_ffr:
                    predicted_FFRs = predicted_FFRs.squeeze(1)
                    # ignore missing target
                    FFRs_corrected = FFRs.clone().detach()
                    FFRs_corrected[torch.isnan(FFRs).tolist()] = predicted_FFRs[torch.isnan(FFRs).tolist()].detach().type(FFRs.dtype)
                
                if enable_multibranch_ifr:
                    predicted_iFRs = predicted_iFRs.squeeze(1)
                    # ignore missing target
                    iFRs_corrected = iFRs.clone().detach()
                    iFRs_corrected[torch.isnan(iFRs).tolist()] = predicted_iFRs[torch.isnan(iFRs).tolist()].detach().type(iFRs.dtype)

        # calculate label predicted and scores
        _, predicted_labels = torch.max(predicted_labels_logits.data, 1)
        predicted_scores = predicted_labels_logits.data.clone().detach().cpu()
        
        if enable_multibranch_ffr and enable_multibranch_ifr:
            predicted = predicted_labels, predicted_scores, predicted_FFRs, predicted_iFRs
        elif enable_multibranch_ffr:
            predicted = predicted_labels, predicted_scores, predicted_FFRs
        elif enable_multibranch_ifr:
            predicted = predicted_labels, predicted_scores, predicted_iFRs
        else:
            predicted = predicted_labels, predicted_scores

        return predicted