import math
import torch
from torch.nn import CrossEntropyLoss

import inspect
import transformers
from transformers import EvalPrediction
from transformers.optimization import get_scheduler

ID2LABEL = ['Sorting',
            'Data structures',
            'Complete search',
            'Greedy algorithms',
            'Dynamic programming',
            'Amortized analysis',
            'Range queries',
            'Bit manipulation']


class Trainer(transformers.Trainer):
    """Use CosineAnnealingLR from pytorch 
    """
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps)
            if getattr(self.args, 'use_cosine_anneal_with_warmup', False):
                lr_max=1
                lr_min=1e-1
                cosine_anneal_with_warmup = lambda cur_iter: max(cur_iter / num_warmup_steps, 1e-9) if cur_iter < num_warmup_steps else \
                    (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_iter-num_warmup_steps)/(num_training_steps-num_warmup_steps)*math.pi)))
                
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer=self.optimizer if optimizer is None else optimizer, 
                    lr_lambda=cosine_anneal_with_warmup,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler


class Skill_Trainer(Trainer):
    """Skill_Trainer class is based on the Trainer class from HuggingFace
    Designed specifically for modeling certain algorithmic skill
    """
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids", "scores"] + self.label_names))
            
    def compute_loss(self, model, batch, return_outputs=False):
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        scores = batch.get("scores")
        
        outputs = model(input_ids)
        loss = am_scored_loss(labels, outputs.logits, scores)
        return (loss, outputs) if return_outputs else loss


def am_scored_loss(labels, logits,  scores, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = labels[..., 1:].contiguous()
    shift_scores = scores[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    skill_loss_per_sample = (loss.view(shift_logits.size(0), shift_logits.size(1)) * shift_scores).mean(axis=1)
    # Calculate total_loss = clm_loss + skill_loss(am-scored loss)
    total_loss = alpha * (loss_per_sample + skill_loss_per_sample)
    loss = total_loss.mean()
    return loss


def compute_ppl(p: EvalPrediction):
    lm_logits = p.predictions
    labels = p.label_ids
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    ppl = torch.exp(loss)
    return {"perplexity": ppl.item()}