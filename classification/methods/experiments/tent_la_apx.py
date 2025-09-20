"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""
import torch
import torch.nn as nn

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
import torch.nn.functional as F
import wandb


@ADAPTATION_REGISTRY.register()
class TENT_LA_APX(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.init_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.init_params[name] = param.clone().detach()
                
        self.TAU = cfg.LOGIT_ADJUST.TAU
        self.EPSILON = cfg.LOGIT_ADJUST.EPSILON
        self.CONFIDENCE_THREASHOLD = cfg.LOGIT_ADJUST.CONFIDENCE_THREASHOLD
        self.ENTROPY_THREASHOLD = cfg.LOGIT_ADJUST.ENTROPY_THREASHOLD

        self._prior = None 
        self.ALPHA = cfg.LOGIT_ADJUST.ALPHA
        self.WARMUP_STEPS = cfg.LOGIT_ADJUST.WARMUP_STEPS

        self.loss = None
        self.minibatch_count = 0 # to track number of forward passes
        # setup loss function
        self.softmax_entropy = Entropy()

        self.loss_calculation = self.select_loss_function(cfg)

        print("Hyperparameters:")
        print(f"TAU: {self.TAU}")
        print(f"EPSILON: {self.EPSILON}")
        print(f"ALPHA: {self.ALPHA}")

    def select_loss_function(self, cfg):
        if cfg.LOGIT_ADJUST.TYPE == "la":
            return self.loss_la
        else:
            raise ValueError(f"Unknown logit adjustment type: {cfg.LOGIT_ADJUST.TYPE}")
   
    def loss_la(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        
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
        if self.TAU > 0 and self.minibatch_count > self.WARMUP_STEPS:
            if self.minibatch_count == self.WARMUP_STEPS + 1:
                print("Warmup Completed. Applying Logit Adjustment.") 
            outputs = outputs + self.TAU * torch.log(self._prior.clone())
        
        loss = self.softmax_entropy(outputs).mean(0)

        self.loss = loss
        self.minibatch_count += 1

        return outputs, loss        

    def log_class_prior(self):
        """Log the current estimate of the class prior."""
        if self._prior is not None:
            prior_dict = {f"prior_class_{i}": prob.item() for i, prob in enumerate(self._prior)}
            wandb.log(prior_dict)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
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

    def reset(self):
        """Reset the model and optimizer state to the initial source state"""
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self._prior = None
        self.minibatch_count = 0
