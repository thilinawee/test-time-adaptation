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
        elif cfg.LOGIT_ADJUST.TYPE == "softmax_prior_la":
            return self.loss_filtered_la
        elif cfg.LOGIT_ADJUST.TYPE == "reg_la":
            return self.loss_la_with_regularization
        elif cfg.LOGIT_ADJUST.TYPE == "prediction_prior_la":
            return self.loss_prior_from_predictions
        elif cfg.LOGIT_ADJUST.TYPE == "prediction_prior_sqrt_la":
            return self.loss_prior_from_predictions_sqrt
        elif cfg.LOGIT_ADJUST.TYPE == "prediction_prior_sqrt_confidence_filter_la":
            return self.loss_prior_from_predictions_sqrt_confidence_filter
        elif cfg.LOGIT_ADJUST.TYPE == "global_prior_ema_la":
            return self.loss_global_prior_ema
        elif cfg.LOGIT_ADJUST.TYPE == "global_prior_la":
            return self.loss_global_prior
        elif cfg.LOGIT_ADJUST.TYPE == "confidence_diff_filter":
            return self.loss_confidence_diff_filter
        elif cfg.LOGIT_ADJUST.TYPE == "confidence_diff_prediction_prior_sqrt_la":
            return self.loss_confidence_diff_prediction_prior_sqrt_la
        elif cfg.LOGIT_ADJUST.TYPE == "prediction_prior_softmax_normalize_la":
            return self.loss_prediction_prior_softmax_normalize
        else:
            raise ValueError(f"Unknown logit adjustment type: {cfg.LOGIT_ADJUST.TYPE}")

    def loss_la(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            batch_prior = probs.mean(dim=0).clamp(min=self.EPSILON)
            batch_prior = batch_prior / batch_prior.sum()
            self._prior = self.ALPHA * self._prior + (1 - self.ALPHA) * batch_prior
            self._prior = self._prior.clamp(min=self.EPSILON)
            self._prior = self._prior / self._prior.sum()
        # adjust logits
        if self.TAU > 0:
            outputs = outputs + self.TAU *  torch.log(self._prior.clone())

        loss = self.softmax_entropy(outputs).mean(0)
        return outputs, loss
    
    def loss_la_with_regularization(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
                
            if self._prior is not None:
                # Calculate batch prior using only low entropy samples
                batch_prior = probs.mean(dim=0).clamp(min=self.EPSILON)
                batch_prior = batch_prior / batch_prior.sum()
                self._prior = self.ALPHA * self._prior + (1 - self.ALPHA) * batch_prior
                self._prior = self._prior.clamp(min=self.EPSILON)
                self._prior = self._prior / self._prior.sum()
            else:
                self._prior = torch.zeros(outputs.shape[1], device=outputs.device)
                batch_prior = probs.mean(dim=0).clamp(min=self.EPSILON)
                batch_prior = batch_prior / batch_prior.sum()
                self._prior += batch_prior

        # Adjust logits
        if self.TAU > 0:
            outputs = outputs + self.TAU * torch.log(self._prior.clone())

        # locotta regularization
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                cosine_weight = (F.cosine_similarity(param.flatten().clone().detach(), self.init_params[name].flatten(), dim=0))
                reg_loss +=  torch.sum((param - self.init_params[name]) ** 2).mul(cosine_weight)
                

        loss = self.softmax_entropy(outputs).mean(0)
        loss +=  reg_loss

        return outputs, loss        
    
    def loss_filtered_la(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            
            # Calculate entropy for each sample
            sample_entropies = -torch.sum(probs * torch.log(probs + self.EPSILON), dim=1)
            
            # Define entropy threshold
            entropy_threshold = torch.log(torch.tensor(outputs.shape[1], device=outputs.device)) * 1.0
            
            # Create mask for low entropy samples
            low_entropy_mask = sample_entropies < entropy_threshold
            
            # Count the number of low entropy samples
            num_filtered_samples = low_entropy_mask.sum().item()
            
            # Only proceed if we have low entropy samples
            if num_filtered_samples > 0:
                # Filter to only low entropy samples
                low_entropy_probs = probs[low_entropy_mask]
                
                if self._prior is not None:
                    # Calculate batch prior using only low entropy samples
                    batch_prior = low_entropy_probs.mean(dim=0).clamp(min=self.EPSILON)
                    batch_prior = batch_prior / batch_prior.sum()
                    self._prior = self.ALPHA * self._prior + (1 - self.ALPHA) * batch_prior
                    self._prior = self._prior.clamp(min=self.EPSILON)
                    self._prior = self._prior / self._prior.sum()
                else:
                    self._prior = torch.zeros(outputs.shape[1], device=outputs.device)
                    batch_prior = low_entropy_probs.mean(dim=0).clamp(min=self.EPSILON)
                    batch_prior = batch_prior / batch_prior.sum()
                    self._prior += batch_prior
            
            # If no low entropy samples available, keep existing prior unchanged
            elif self._prior is None:
                self._prior = torch.ones(outputs.shape[1], device=outputs.device) / outputs.shape[1]

        # Adjust logits
        if self.TAU > 0:
            outputs = outputs + self.TAU * torch.log(self._prior.clone())

        loss = self.softmax_entropy(outputs).mean(0)
        
        # Print the number of filtered samples for prior update
        print(f'Number of filtered samples for prior update: {num_filtered_samples}')
        return outputs, loss
    
    def loss_prior_from_predictions(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)

            # Calculate entropy for each sample
            sample_entropies = -torch.sum(probs * torch.log(probs + self.EPSILON), dim=1)
            
            # Define entropy threshold
            entropy_threshold = torch.log(torch.tensor(outputs.shape[1], device=outputs.device)) * 1.0
            
            # Create mask for low entropy samples
            low_entropy_mask = sample_entropies < entropy_threshold
            
            # Count the number of low entropy samples
            num_filtered_samples = low_entropy_mask.sum().item()
            
            # Only proceed if we have low entropy samples
            if num_filtered_samples > 0:
                # Filter to only low entropy samples
                low_entropy_probs = probs[low_entropy_mask]
                
                if self._prior is None:
                    self._prior = torch.zeros(outputs.shape[1], device=outputs.device)
               
                class_counts = torch.bincount(predictions[low_entropy_mask], minlength=outputs.shape[1]).float().clamp(min=self.EPSILON)
                batch_prior = class_counts / class_counts.sum()
                if self._prior is None:
                    self.prior += batch_prior
                else:
                    self._prior = self.ALPHA * self._prior + (1 - self.ALPHA) * batch_prior

                self._prior = self._prior.clamp(min=self.EPSILON)
                self._prior = self._prior / self._prior.sum()
            
        # Adjust logits
        if self.TAU > 0:
            outputs = outputs + self.TAU * torch.log(self._prior.clone())

        loss = self.softmax_entropy(outputs).mean(0)
        
        # Print the number of filtered samples for prior update
        print(f'Number of filtered samples for prior update: {num_filtered_samples}')
        return outputs, loss
    
    def loss_prior_from_predictions_sqrt(self, x):
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
        if self.TAU > 0:
            outputs = outputs + self.TAU * torch.log(self._prior.clone())
        
        loss = self.softmax_entropy(outputs).mean(0)
        
        return outputs, loss
    
    def loss_prior_from_predictions_sqrt_confidence_filter(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)

            high_confidence_mask = probs.max(dim=1).values > self.CONFIDENCE_THREASHOLD
            num_filtered_samples = high_confidence_mask.sum().item()

            # Only proceed if we have low entropy samples
            if num_filtered_samples > 0:
                
                if self._prior is None:
                    self._prior = torch.zeros(outputs.shape[1], device=outputs.device)
               
                class_counts = torch.bincount(predictions[high_confidence_mask], minlength=outputs.shape[1]).float().clamp(min=self.EPSILON)
                sqrt_counts = torch.sqrt(class_counts)
                batch_prior = sqrt_counts / sqrt_counts.sum()
                if self._prior is None:
                    self.prior += batch_prior
                else:
                    self._prior = self.ALPHA * self._prior + (1 - self.ALPHA) * batch_prior

                self._prior = self._prior.clamp(min=self.EPSILON)
                self._prior = self._prior / self._prior.sum()
            
        # Adjust logits
        if self.TAU > 0:
            outputs = outputs + self.TAU * torch.log(self._prior.clone())

        loss = self.softmax_entropy(outputs).mean(0)
        
        # Print the number of filtered samples for prior update
        print(f'Number of filtered samples for prior update: {num_filtered_samples}')
        return outputs, loss

    def loss_global_prior_ema(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            # Initialize global bin count if it doesn't exist
            if not hasattr(self, '_global_bin_count'):
                self._global_bin_count = torch.zeros(outputs.shape[1], device=outputs.device)
            
            # Update global bin count with current batch predictions
            batch_counts = torch.bincount(predictions, minlength=outputs.shape[1]).float()
            self._global_bin_count += batch_counts
            
            # Create current batch prior from global bin count
            clamped_counts = self._global_bin_count.clamp(min=self.EPSILON)
            sqrt_counts = torch.sqrt(clamped_counts)
            current_prior = sqrt_counts / sqrt_counts.sum()
            # current_prior = clamped_counts / clamped_counts.sum()
            
            # Apply EMA to update the prior
            if not hasattr(self, '_prior') or self._prior is None:
                self._prior = current_prior.clone()
            else:
                self._prior = self.ALPHA * self._prior + (1 - self.ALPHA) * current_prior
            
            # Ensure prior is properly normalized and clamped
            self._prior = self._prior.clamp(min=self.EPSILON)
            self._prior = self._prior / self._prior.sum()
        
        # Adjust logits
        if self.TAU > 0:
            outputs = outputs + self.TAU * torch.log(self._prior.clone())
        
        loss = self.softmax_entropy(outputs).mean(0)
        
        return outputs, loss
    
    def loss_global_prior(self, x):         
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            # Initialize global bin count if it doesn't exist
            if not hasattr(self, '_global_bin_count'):
                self._global_bin_count = torch.zeros(outputs.shape[1], device=outputs.device)
            
            # Update global bin count with current batch predictions
            batch_counts = torch.bincount(predictions, minlength=outputs.shape[1]).float()
            self._global_bin_count += batch_counts
            
            # Create prior from global bin count
            # Clamp to avoid division by zero and normalize
            clamped_counts = self._global_bin_count.clamp(min=self.EPSILON)
            # sqrt_counts = torch.sqrt(clamped_counts)
            # self._prior = sqrt_counts / sqrt_counts.sum()
            self._prior = clamped_counts / clamped_counts.sum()
            
            # Ensure prior is properly normalized and clamped
            self._prior = self._prior.clamp(min=self.EPSILON)
            self._prior = self._prior / self._prior.sum()
        
        # Adjust logits
        if self.TAU > 0:
            outputs = outputs + self.TAU * torch.log(self._prior.clone())
        
        loss = self.softmax_entropy(outputs).mean(0)
        
        return outputs, loss
    def loss_prediction_prior_softmax_normalize(self, x):
        imgs_test = x[0]         
        outputs = self.model(imgs_test)          

        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)              
            
            # Initialize prior if it doesn't exist
            if self._prior is None:
                self._prior = torch.zeros(outputs.shape[1], device=outputs.device)
            
            # Calculate class counts from all samples (no entropy filtering)
            class_counts = torch.bincount(predictions, minlength=outputs.shape[1]).float().clamp(min=self.EPSILON)
            batch_prior = torch.softmax(class_counts, dim=0)
            
            # Update prior with exponential moving average
            self._prior = self.ALPHA * self._prior + (1 - self.ALPHA) * batch_prior
            
            # Normalize prior using softmax for better numerical stability
            self._prior = torch.softmax(self._prior, dim=0)

        # Adjust logits with prior
        if self.TAU > 0 and self._prior is not None:
            outputs = outputs + self.TAU * torch.log(self._prior.clone() + self.EPSILON)

        loss = self.softmax_entropy(outputs).mean(0)
        return outputs, loss

    def loss_confidence_diff_filter(self, x):
        imgs_test = x[0]
        with torch.no_grad():
            original_outputs = self.original_model(imgs_test)
        outputs = self.model(imgs_test)
        soft_orig = F.softmax(original_outputs, dim=1)
        c_o = torch.argmax(soft_orig, dim=1)
        orig_conf = soft_orig.gather(1, c_o.unsqueeze(1)).squeeze(1)
        soft_adapt = F.softmax(outputs, dim=1)
        adapt_conf = soft_adapt.gather(1, c_o.unsqueeze(1)).squeeze(1)
        mask = (adapt_conf >= orig_conf).float()
        ent = self.softmax_entropy(outputs)
        loss = (ent * mask).mean(0)
        mean_soft = soft_adapt.mean(0, keepdim=True)
        loss -= 0.25 * self.softmax_entropy(mean_soft).mean(0)
        return outputs, loss        

    def loss_confidence_diff_prediction_prior_sqrt_la(self, x):

        imgs_test = x[0]
        
        with torch.no_grad():
            original_outputs = self.original_model(imgs_test)
        
        outputs = self.model(imgs_test)
        predictions = torch.argmax(outputs, dim=1)
        
        soft_orig = F.softmax(original_outputs, dim=1)
        c_o = torch.argmax(soft_orig, dim=1)
        orig_conf = soft_orig.gather(1, c_o.unsqueeze(1)).squeeze(1)
        
        soft_adapt = F.softmax(outputs, dim=1)
        adapt_conf = soft_adapt.gather(1, c_o.unsqueeze(1)).squeeze(1)
        
        confidence_mask = adapt_conf >= orig_conf
        num_filtered_samples = confidence_mask.sum().item()
        
        if num_filtered_samples > 0:
            with torch.no_grad():
                # Initialize prior if it doesn't exist
                if self._prior is None:
                    self._prior = torch.zeros(outputs.shape[1], device=outputs.device)
                
                class_counts = torch.bincount(predictions[confidence_mask],
                                               minlength=outputs.shape[1]).float().clamp(min=self.EPSILON)
                sqrt_counts = torch.sqrt(class_counts)
                batch_prior = sqrt_counts / sqrt_counts.sum()
                
                # Fixed: Use exponential moving average correctly
                self._prior = self.ALPHA * self._prior + (1 - self.ALPHA) * batch_prior
                
                # Ensure prior is normalized and has minimum values
                self._prior = self._prior.clamp(min=self.EPSILON)
                self._prior = self._prior / self._prior.sum()
        
        # Apply prior adjustment to logits only if prior exists and TAU > 0
        if self._prior is not None and self.TAU > 0:
            outputs = outputs + self.TAU * torch.log(self._prior.clone())
        
        # Apply confidence mask to outputs before calculating entropy loss

        filtered_outputs = outputs[confidence_mask]
        loss = self.softmax_entropy(filtered_outputs).mean(0)


        return outputs, loss

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
