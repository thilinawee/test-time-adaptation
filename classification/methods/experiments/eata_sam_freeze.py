"""
Builds upon: https://github.com/mr-eggplant/EATA
Corresponding paper: https://arxiv.org/abs/2204.02610
"""

import os
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from datasets.data_loading import get_source_loader
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy

logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class EATA_SAM_FREEZE(TTAMethod):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """

    """
    We Freeze certain layers of the model during the adaptation similar to SAR
    Instead of using SGD, we use SAM optimizer for the adaptation
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        logger.info("EATA_SAM_FREEZE Initialized.")
        self.num_samples_update_1 = 0  # number of samples after first filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after second filtering, exclude both unreliable and redundant samples
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)   # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = cfg.EATA.D_MARGIN   # hyperparameter \epsilon for cosine similarity thresholding (Eqn. 5)

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)
        self.fisher_alpha = cfg.EATA.FISHER_ALPHA  # trade-off \beta for two losses (Eqn. 8)

        # setup loss function
        self.softmax_entropy = Entropy()

        if self.fisher_alpha > 0.0 and self.cfg.SOURCE.NUM_SAMPLES > 0:
            # compute fisher informatrix
            batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
            _, fisher_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                                 adaptation=cfg.MODEL.ADAPTATION,
                                                 preprocess=model.model_preprocess,
                                                 data_root_dir=cfg.DATA_DIR,
                                                 batch_size=batch_size_src,
                                                 ckpt_path=cfg.MODEL.CKPT_PATH,
                                                 num_samples=cfg.SOURCE.NUM_SAMPLES,    # number of samples for ewc reg.
                                                 percentage=cfg.SOURCE.PERCENTAGE,
                                                 workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
                                                 use_clip=cfg.MODEL.USE_CLIP)
            ewc_optimizer = torch.optim.SGD(self.params, 0.001)
            self.fishers = {} # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
            train_loss_fn = nn.CrossEntropyLoss().to(self.device)
            for iter_, batch in enumerate(fisher_loader, start=1):
                images = batch[0].to(self.device, non_blocking=True)
                outputs = self.model(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + self.fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        self.fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            logger.info("Finished computing the fisher matrices...")
            del ewc_optimizer
        else:
            logger.info("Not using EWC regularization. EATA decays to ETA!")
            self.fishers = None

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x[0]

        ######################## BEGIN FIRST LOSS CALCULATION ########################
        outputs = self.model(imgs_test)
        entropys = self.softmax_entropy(outputs)

        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]

        # filter redundant samples
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            # DO NOT UPDATE PROBABILITIES HERE. UPDATE AT THE END OF SECOND LOSS CALCULATION

        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))

        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)

        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
            loss += ewc_loss

        ######################## END FIRST LOSS CALCULATION ########################
        perform_update = len(entropys) != 0

        if not perform_update:
            # filter redundant samples, update model probabilities and return outputs
            if self.current_model_probs is not None:
                updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))

            self.current_model_probs = updated_probs
            return outputs

        # first step of SAM
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        ######################## BEGIN SECOND LOSS CALCULATION ########################
        outputs = self.model(imgs_test)
        entropys = self.softmax_entropy(outputs)

        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]

        # filter redundant samples
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))

        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)

        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
            loss += ewc_loss
            
        ######################## END SECOND LOSS CALCULATION ########################
        loss.backward()
        self.optimizer.second_step(zero_grad=True)
        
        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += entropys.size(0)
        self.current_model_probs = updated_probs

        return outputs

    # @torch.enable_grad()
    # def forward_and_adapt(self, x):
    #     if self.mixed_precision and self.device == "cuda":
    #         with torch.cuda.amp.autocast():
    #             outputs, loss, perform_update = self.loss_calculation(x)
    #         # update model only if not all instances have been filtered
    #         if perform_update:
    #             self.scaler.scale(loss).backward()
    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()
    #         self.optimizer.zero_grad()
    #     else:
    #         outputs, loss, perform_update = self.loss_calculation(x)
    #         # update model only if not all instances have been filtered
    #         if perform_update:
    #             loss.backward()
    #             self.optimizer.step()
    #         self.optimizer.zero_grad()
    #     return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.current_model_probs = None

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():

            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if self.freeze_layers == []:
                if 'layer4' in nm:
                    continue
                if 'blocks.9' in nm:
                    continue
                if 'blocks.10' in nm:
                    continue
                if 'blocks.11' in nm:
                    continue
            else:
                if any([f'{layer}' in nm for layer in self.freeze_layers]):
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

        logger.info(f"EATA_SAM_FREEZE Collected Parameters: {names}")
        return params, names

    def configure_model(self):
        """Configure model for use with eata."""
        # train mode, because eata optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what eata updates
        self.model.requires_grad_(False)
        # configure norm for eata updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def setup_optimizer(self):
        architecture_name = self.cfg.MODEL.ARCH.lower().replace("-", "_")
        if "vit_" in architecture_name or "swin_" in architecture_name:
            logger.info("Overwriting learning rate for transformers, using a learning rate of 0.001.")
            return SAM(self.params, torch.optim.SGD, lr=0.001, momentum=self.cfg.OPTIM.MOMENTUM)
        else:
            return SAM(self.params, torch.optim.SGD, lr=self.cfg.OPTIM.LR, momentum=self.cfg.OPTIM.MOMENTUM)
        

def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


class SAM(torch.optim.Optimizer):
    # from https://github.com/davda54/sam
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
