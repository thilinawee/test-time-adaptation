"""
Builds upon: https://github.com/Jhyun17/DeYO/blob/main/methods/deyo.py
Corresponding paper: https://openreview.net/pdf?id=9w3iw8wDuE
"""

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
from einops import rearrange

from methods.base import TTAMethod
from utils.losses import Entropy
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class DEYO_LA_APX(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.reweight_ent = cfg.DEYO.REWEIGHT_ENT
        self.reweight_plpd = cfg.DEYO.REWEIGHT_PLPD

        self.plpd_threshold = cfg.DEYO.PLPD
        self.deyo_margin = cfg.DEYO.MARGIN * math.log(num_classes)
        self.margin_e0 = cfg.EATA.MARGIN_E0 * math.log(num_classes)

        self.aug_type = cfg.DEYO.AUG_TYPE
        self.occlusion_size = cfg.DEYO.OCCLUSION_SIZE
        self.row_start = cfg.DEYO.ROW_START
        self.column_start = cfg.DEYO.COLUMN_START
        self.patch_len = cfg.DEYO.PATCH_LEN

        self.ent = Entropy()

        # logit adjustment related parameters
        self.TAU = cfg.LOGIT_ADJUST.TAU
        self.EPSILON = cfg.LOGIT_ADJUST.EPSILON
        self.ALPHA = cfg.LOGIT_ADJUST.ALPHA

        self._prior = None
        self.minibatch_count = 0 # to track number of forward passes

        self.loss_calculation = self.select_loss_function(cfg)

    def select_loss_function(self, cfg):
        if cfg.LOGIT_ADJUST.TYPE == "la":
            return self.loss_la
        else:
            raise ValueError(f"Unknown logit adjustment type: {cfg.LOGIT_ADJUST.TYPE}")

    def loss_la(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        # adjust logits
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            class_counts = torch.bincount(predictions, minlength=outputs.shape[1]).float().clamp(min=self.EPSILON)
            sqrt_counts = torch.sqrt(class_counts)
            batch_prior = sqrt_counts / sqrt_counts.sum()
            
            if self._prior is None:
                self._prior = batch_prior
            else:
                self._prior = self.ALPHA * self._prior + (1 - self.ALPHA) * batch_prior
            
            self._prior = self._prior.clamp(min=self.EPSILON)
            self._prior = self._prior / self._prior.sum()
        
        # Adjust logits
        if self.TAU > 0 and self.minibatch_count >= self.WARMUP_STEPS:
            if self.minibatch_count == self.WARMUP_STEPS:
                print("Warmup Completed. Applying Logit Adjustment.") 
            outputs = outputs + self.TAU * torch.log(self._prior.clone())

        entropys = self.ent(outputs)
        filter_ids_1 = torch.where((entropys < self.deyo_margin))
        entropys = entropys[filter_ids_1]
        if len(entropys) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return outputs, loss

        x_prime = imgs_test[filter_ids_1]
        x_prime = x_prime.detach()
        if self.aug_type == 'occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size, self.occlusion_size)
            x_prime[:, :, self.row_start:self.row_start + self.occlusion_size, self.column_start:self.column_start + self.occlusion_size] = occlusion_window
        elif self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(((imgs_test.shape[-1] // self.patch_len) * self.patch_len, (imgs_test.shape[-1] // self.patch_len) * self.patch_len))
            resize_o = torchvision.transforms.Resize((imgs_test.shape[-1], imgs_test.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self.patch_len, ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self.patch_len, ps2=self.patch_len)
            x_prime = resize_o(x_prime)
        elif self.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=imgs_test.shape[-1], ps2=imgs_test.shape[-1])

        with torch.no_grad():
            outputs_prime = self.model(x_prime)
            # adjust logits
            if self.TAU > 0:
                outputs_prime = outputs_prime + self.TAU *  torch.log(self._prior.clone())

        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)

        cls1 = prob_outputs.argmax(dim=1)

        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
        plpd = plpd.reshape(-1)

        filter_ids_2 = torch.where(plpd > self.plpd_threshold)
        entropys = entropys[filter_ids_2]
        if len(entropys) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return outputs, loss

        plpd = plpd[filter_ids_2]

        if self.reweight_ent or self.reweight_plpd:
            coeff = (float(self.reweight_ent) * (1. / (torch.exp(((entropys.clone().detach()) - self.margin_e0)))) +
                     float(self.reweight_plpd) * (1. / (torch.exp(-1. * plpd.clone().detach())))
                     )
            entropys = entropys.mul(coeff)

        loss = entropys.mean(0)
        self.minibatch_count += 1
        return outputs, loss        

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            # update model only if not all instances have been filtered
            if loss is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            # update model only if not all instances have been filtered
            if loss is not None:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names
    
    def configure_model(self):
        """Configure model for use with DeYO."""
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what DeYO updates
        self.model.requires_grad_(False)
        # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def reset(self):
        """Reset the model and optimizer state to the initial source state"""
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self._prior = None
        self.minibatch_count = 0