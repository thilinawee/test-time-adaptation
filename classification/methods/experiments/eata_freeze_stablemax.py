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
import wandb

from methods.base import TTAMethod
from datasets.data_loading import get_source_loader
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
from conf import GlobalVar

logger = logging.getLogger(__name__)

def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )

def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))

def stablemax(x, dim=-1):
    s_x = s(x)
    return s_x/torch.sum(s_x, dim=dim, keepdim=True)

class StableMaxEntropy(nn.Module):
    def __init__(self):
        super(StableMaxEntropy, self).__init__()

    def __call__(self, logits):

        prob = stablemax(logits)
        return -(prob * torch.log(prob + 1e-30)).sum(1)


@ADAPTATION_REGISTRY.register()
class EATA_FREEZE_STABLEMAX(TTAMethod):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """

    """
    We Freeze certain layers of the model during the adaptation similar to SAR
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.global_var = GlobalVar()

        wandb.define_metric(f"softmax_entropy/*", step_metric = "custom_step")
        wandb.define_metric(f"stablemax_entropy/*", step_metric = "custom_step") 

        self.num_samples_update_1 = 0  # number of samples after first filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after second filtering, exclude both unreliable and redundant samples
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)   # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = cfg.EATA.D_MARGIN   # hyperparameter \epsilon for cosine similarity thresholding (Eqn. 5)

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)
        self.fisher_alpha = cfg.EATA.FISHER_ALPHA  # trade-off \beta for two losses (Eqn. 8)

        # setup loss function
        self.stablemax_entropy = StableMaxEntropy()
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

    def loss_calculation(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        do_softmax = True

        imgs_test = x[0]
        outputs = self.model(imgs_test)
        entropys = self.softmax_entropy(outputs) if do_softmax else self.stablemax_entropy(outputs)
        # entropys = self.stablemax_entropy(outputs)

        if self.cfg.DEBUG:
            stablemax_entropys = self.stablemax_entropy(outputs)
            softmax_entropys = self.softmax_entropy(outputs)
            wandb.log({f"softmax_entropy/{self.global_var.corruption_name}": softmax_entropys.mean().item(), "custom_step": self.global_var.adaptation_step})
            wandb.log({f"stablemax_entropy/{self.global_var.corruption_name}": stablemax_entropys.mean().item(), "custom_step": self.global_var.adaptation_step}) 

        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]

        if self.global_var.adaptation_step % 50 == 0:
            logger.info(f"Step: {self.global_var.adaptation_step}, Entropy after filter 1: {entropys}")

        # filter redundant samples
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1) if do_softmax else stablemax(outputs[filter_ids_1]), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1) if do_softmax else stablemax(outputs[filter_ids_1][filter_ids_2]))
        else:
            updated_probs = update_model_probs(self.current_model_probs,  outputs[filter_ids_1].softmax(1) if do_softmax else stablemax(outputs[filter_ids_1]))


        if self.global_var.adaptation_step % 50 == 0:
            logger.info(f"Step: {self.global_var.adaptation_step}, Entropy after filter 2: {entropys}")

        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))


        if self.global_var.adaptation_step % 50 == 0:
            logger.info(f"Step: {self.global_var.adaptation_step}, Coefficient: {coeff}")

        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)

        if self.cfg.DEBUG:
            wandb.log({f"loss/{self.global_var.corruption_name}": loss.item(), "custom_step": self.global_var.adaptation_step})
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # if x[ids1][ids2].size(0) != 0:
        #     loss = self.softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
            loss += ewc_loss

        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.num_samples_update_2 += entropys.size(0)
        self.current_model_probs = updated_probs
        perform_update = len(entropys) != 0
        return outputs, loss, perform_update

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss, perform_update = self.loss_calculation(x)
            # update model only if not all instances have been filtered
            if perform_update:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss, perform_update = self.loss_calculation(x)
            # update model only if not all instances have been filtered
            if perform_update:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs

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
            
            # @TODO: check why following items are skipped
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue
            
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        logger.info(f"EATA_FREEZE Collected Parameters: {names}")
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

        return OrthoGrad(self.params, torch.optim.SGD, lr=self.cfg.OPTIM.LR, momentum=self.cfg.OPTIM.MOMENTUM)
        

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



class OrthoGrad(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls=torch.optim.SGD, **base_optimizer_args):
        """
        A wrapper optimizer that projects gradients to be orthogonal
        to the current parameters before performing an update.

        Args:
            params (iterable): Iterable of parameters to optimize.
            base_optimizer_cls (Optimizer class): The base optimizer class
                (e.g., torch.optim.SGD, torch.optim.AdamW).
            **base_optimizer_args: Arguments for the base optimizer.
                For example, lr=1e-3, weight_decay=1e-2, etc.
        """
        # Minimal defaults for OrthoGrad itself (nothing special needed).
        defaults = {}
        super().__init__(params, defaults)

        # Create the wrapped/base optimizer using *our* param_groups.
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_args)

    @staticmethod
    def _orthogonalize_gradients(params):
        """
        Projects the gradient g to be orthogonal to the current weights w.

        g_orth = g - ( (w·g)/(w·w + eps) ) * w

        And then re-scales g_orth to have the same norm as g.
        """
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    w = p.view(-1)
                    g = p.grad.view(-1)

                    w_norm_sq = torch.dot(w, w) + 1e-30
                    proj = torch.dot(w, g) / w_norm_sq
                    g_orth = g - proj * w

                    g_norm = g.norm(2)
                    g_orth_norm = g_orth.norm(2) + 1e-30
                    g_orth_scaled = g_orth * (g_norm / g_orth_norm)

                    p.grad.copy_(g_orth_scaled.view_as(p.grad))

    def step(self, closure=None):
        for group in self.param_groups:
            self._orthogonalize_gradients(group['params'])

        return self.base_optimizer.step(closure)