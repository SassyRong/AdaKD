import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

def simple_forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def forward_kl(logits, teacher_logits, no_model_batch, ratio=None, return_per_token_loss=False):
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    prod_probs =  torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    if return_per_token_loss:
        return x
    if ratio == None:
        distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    else:
        distil_loss = torch.sum(x * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(logits, teacher_logits, no_model_batch, ratio=None, return_per_token_loss=False):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    if return_per_token_loss:
        return -x
    mask = (no_model_batch["label"] != -100).int()
    if ratio == None:
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    else:
        distil_loss = -torch.sum(x * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def symmetric_kl(logits, teacher_logits, logits_smooth, teacher_logits_smooth, no_model_batch, lam=0.9, return_per_token_loss=False):
    if return_per_token_loss:
        for_kl = forward_kl(logits_smooth, teacher_logits_smooth, no_model_batch, return_per_token_loss=True)
        rev_kl = reverse_kl(logits, teacher_logits, no_model_batch, return_per_token_loss=True)
        return 1 * for_kl + 1 * rev_kl
    
    for_kl = forward_kl(logits_smooth, teacher_logits_smooth, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = 1 * for_kl + 1 * rev_kl
    return distil_loss


def js_distance(logits, teacher_logits, no_model_batch, lam=0.9, return_per_token_loss=False):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    mixed_probs = mixed_probs.clamp(min=1e-8)

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    y = torch.sum(prod_probs, dim=-1).view(-1)
    if return_per_token_loss:
        return lam * (-x) + (1-lam) * (-y)
    distil_loss += (1-lam) * -torch.sum(y * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    
def tv_distance(logits, teacher_logits, no_model_batch, return_per_token_loss=False):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    if return_per_token_loss:
        return x
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1, return_per_token_loss=False):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_probs = mixed_probs.clamp(min=1e-8)
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    if return_per_token_loss:
        return -x    
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1, return_per_token_loss=False):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    mixed_probs = mixed_probs.clamp(min=1e-8)
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    if return_per_token_loss:
        return -x    
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def get_ratio(logits, teacher_logits, mu=0.5):
    # [B, L, V]
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)
    
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()

    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)

    errors = torch.abs(re_teacher_probs - re_student_probs)

    cum_sum = torch.cumsum(re_teacher_probs, dim=-1) # B,L,V
    mask = cum_sum > mu
    mask[:,:,0]=False

    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)

    return s1/(s1+s2), s2/(s1+s2)

def adaptive_kl(logits, teacher_logits, no_model_batch, return_per_token_loss=False):
    h_ratio, l_ratio = get_ratio(logits, teacher_logits)
    if return_per_token_loss:
        fkl_per_token = forward_kl(logits, teacher_logits, no_model_batch, return_per_token_loss=True)
        rkl_per_token = reverse_kl(logits, teacher_logits, no_model_batch, return_per_token_loss=True)
        x = fkl_per_token * h_ratio.view(-1) + rkl_per_token * l_ratio.view(-1) 
        return x
        
    distil_loss = forward_kl(logits, teacher_logits, no_model_batch, h_ratio) + reverse_kl(logits, teacher_logits, no_model_batch, l_ratio)
    return distil_loss


def ab_div(logits, teacher_logits, no_model_batch, alpha=0.2, beta=0.7, return_per_token_loss=False):
    """
    Calculate D^{(alpha, beta)} divergence for student (logits) and teacher (teacher_logits) distributions.

    Args:
        logits: Tensor of student logits (B x S x D).
        teacher_logits: Tensor of teacher logits (B x S x D).
        no_model_batch: Dictionary containing auxiliary data (e.g., labels, mask).
        alpha: The alpha parameter in the divergence.
        beta: The beta parameter in the divergence.

    Returns:
        ab_loss: The alpha-beta divergence loss.
    """
    # Compute teacher and student probabilities
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)  # Shape: (B, S, D)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # Shape: (B, S, D)
    
    log_teacher_probs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_student_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

    # Create inf_mask to handle infinite logits
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    # Special case when alpha = 0 and beta = 0
    if alpha == 0 and beta == 0:
        log_diff = torch.log(student_probs) - torch.log(teacher_probs)  # Shape: (B, S, D)
        log_diff = torch.masked_fill(log_diff, inf_mask, 0)  # Handle infinities
        divergence = 0.5 * torch.sum(log_diff ** 2, dim=-1)  # Shape: (B, S)
    elif alpha == 0 and beta != 0:
        # Case where alpha = 0
        q_beta = torch.pow(student_probs, beta)  # Shape: (B, S, D)
        p_beta = torch.pow(teacher_probs, beta)
        likeli_ratio = q_beta / p_beta
        likeli_ratio = torch.masked_fill(likeli_ratio, torch.isnan(likeli_ratio), 0)
        divergence = (1 / beta) * torch.sum(
            q_beta * torch.log(likeli_ratio) - q_beta + p_beta,
            dim=-1,
        )
    elif beta == 0 and alpha != 0:
        # Case where beta = 0
        p_alpha = torch.pow(teacher_probs, alpha)  # Shape: (B, S, D)
        p_alpha = torch.masked_fill(p_alpha, inf_mask, 0)
        q_alpha = torch.pow(student_probs, alpha)
        q_alpha = torch.masked_fill(q_alpha, inf_mask, 0)
        likeli_ratio = p_alpha / q_alpha
        likeli_ratio = torch.masked_fill(likeli_ratio, torch.isnan(likeli_ratio), 0)
        divergence = (1 / alpha) * torch.sum(
            p_alpha * torch.log(likeli_ratio) - p_alpha + q_alpha,
            dim=-1,
        )
    elif alpha + beta == 0:
        # Case where alpha + beta = 0
        p_alpha = torch.pow(teacher_probs, alpha)  # Shape: (B, S, D)
        q_alpha = torch.pow(student_probs, alpha)  # Shape: (B, S, D)
        p_alpha = torch.masked_fill(p_alpha, inf_mask, 0)
        q_alpha = torch.masked_fill(q_alpha, inf_mask, 0)
        divergence = torch.sum(
            (1 / alpha) * (torch.log(q_alpha / p_alpha) + (p_alpha / q_alpha) - 1),
            dim=-1
        )
    else:
        # General case
        p_alpha = torch.exp(alpha * log_teacher_probs)
        q_beta = torch.exp(beta * log_student_probs)
        term1 = torch.masked_fill(p_alpha * q_beta, inf_mask, 0)

        term2 = torch.masked_fill((alpha / (alpha + beta)) * torch.exp((alpha + beta) * log_teacher_probs), inf_mask, 0)

        term3 = torch.masked_fill((beta / (alpha + beta)) * torch.exp((alpha + beta) * log_student_probs), inf_mask, 0)
    
        divergence = -torch.sum(term1 - term2 - term3, dim=-1) / (alpha * beta)
    
        if return_per_token_loss:
            return divergence.view(-1)


    mask = (no_model_batch["label"] != -100).int()  # Shape: (B, S)

    # Apply the mask first to ignore padding positions
    masked_divergence = divergence * mask.float()  # Shape: (B, S), element-wise mask
    

    # Sum the divergence over the sequence length (S), resulting in shape (B,)
    x = torch.sum(masked_divergence, dim=-1)  # Sum over the sequence dimension (S)

    # Compute the ab_loss by summing the masked loss and normalizing by the number of valid positions
    ab_loss = torch.sum(x) / torch.sum(mask.float())  # Normalize by the total number of valid tokens

    return ab_loss


def bdkd(logits, teacher_logits, no_model_batch):
    def entropy(logits):
        probs = F.softmax(logits, dim=-1)  
        log_probs = torch.log(probs + 1e-9)  
        return -torch.sum(probs * log_probs, dim=-1)  

    entropy_student = entropy(logits)  # (B, S)
    entropy_teacher = entropy(teacher_logits)  # (B, S)

    weight_student = torch.where(entropy_student > entropy_teacher, 3.0, 1.0)  
    weight_teacher = torch.where(entropy_teacher > entropy_student, 3.0, 1.0)  

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss1 = -torch.sum(x * mask.view(-1) * weight_teacher.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss2 = -torch.sum(x * mask.view(-1) * weight_student.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss1 + distil_loss2

def alphanet(logits, teacher_logits, no_model_batch, alpha, beta):
    loss1 = ab_div(logits, teacher_logits, no_model_batch, alpha, 1 - alpha)
    loss2 = ab_div(logits, teacher_logits, no_model_batch, beta, 1 - beta)
    if loss1 > loss2:
        return loss1
    return loss2

class LogTokUCalculator:
    '''
    A class to calculate the LogTokU uncertainty loss.
    Based on the LogTokU paper:
    Huan Ma, et al. "Estimating LLM Uncertainty with Logits"
    https://github.com/MaHuanAAA/logtoku
    '''
    def calculate_au(self, logits, topk):
        # logits: [B, L, V]
        if topk < 1 or topk > logits.size(-1):
            raise ValueError("topk must be between 1 and the vocabulary size.")
        top_logits, _ = torch.topk(logits, topk, dim=-1)
        top_logits_min = torch.min(top_logits, dim=-1, keepdim=True).values
        alpha = top_logits - top_logits_min + 1e-8  # Add a small constant to avoid division by zero
        alpha_0 = torch.sum(alpha, dim=-1, keepdim=True)  # Add a small constant to avoid division by zero
        psi_alpha_k_plus_1 = torch.digamma(alpha + 1)
        psi_alpha_0_plus_1 = torch.digamma(alpha_0 + 1)
        results= -(alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
        au = torch.sum(results, dim=-1)
        au_norm = (au - au.min())/ (au.max() - au.min())
        return au_norm
    
    def calculate_eu(self, logits, topk):
        if topk < 1 or topk > logits.size(-1):
            raise ValueError("topk must be between 1 and the vocabulary size.")
        top_logits, _ = torch.topk(logits, topk, dim=-1)
        top_logits_min = torch.min(top_logits, dim=-1, keepdim=True).values
        evidence = top_logits - top_logits_min + 1e-8 
        total_evidence = torch.sum(evidence, dim=-1)
        eu = topk / (total_evidence + topk)
        eu_norm = (eu - eu.min()) / (eu.max() - eu.min())
        return eu_norm

def calculate_nmtkd_score(
    logits: torch.Tensor, 
    teacher_logits: torch.Tensor, 
    topk: int
) -> torch.Tensor:
    """
    Calculates the score from the NMT-KD paper.
    https://github.com/songmzhang/NMT-KD/tree/main
    Args:
        logits: Student model logits [N, V].
        teacher_logits: Teacher model logits [N, V].
        topk: The 'k' for top-k selection.
        
    Returns:
        hr_score: A tensor of shape [N] where each element is the L_hr score
                  for the corresponding token.
    """
    vocab_size = logits.size(-1)
    k = min(topk, vocab_size)
    student_probs = F.softmax(logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    teacher_topk_vals, teacher_topk_indices = torch.topk(teacher_probs, k, dim=-1)
    student_topk_vals, student_topk_indices = torch.topk(student_probs, k, dim=-1)
    t1_indices = teacher_topk_indices[:, :1]
    p_t1 = torch.gather(student_probs, dim=-1, index=t1_indices)
    p_tu = torch.gather(student_probs, dim=-1, index=teacher_topk_indices)   
    top1_diff = p_tu - p_t1
    loss_top1_score = torch.sum(F.relu(top1_diff), dim=-1)
    p_sv = student_topk_vals.unsqueeze(1)
    p_tu_expanded = p_tu.unsqueeze(2)   
    q_sv = torch.gather(teacher_probs, dim=-1, index=student_topk_indices)
    q_sv_expanded = q_sv.unsqueeze(1)
    q_tu_expanded = teacher_topk_vals.unsqueeze(2)
    teacher_preference_mask = q_tu_expanded > q_sv_expanded
    student_prob_diff = p_sv - p_tu_expanded
    loss_topk_terms = F.relu(teacher_preference_mask * student_prob_diff)
    loss_topk_score = torch.sum(loss_topk_terms, dim=(-1, -2))
    hr_score = (loss_top1_score + loss_topk_score).detach()
    return hr_score
    
    
def _compute_token_scores(
    rule: str,
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    student_log: torch.Tensor,
    teacher_log: torch.Tensor,
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    topk: int
) -> torch.Tensor:
    if rule in ("all", "random"):
        return torch.ones(student_probs.size(0), device=logits.device)
    if rule == "prob":
        return 1 - student_probs.max(dim=-1).values
    if rule == "entropy":
        return -(student_probs * student_log).sum(dim=-1)
    if rule == "ce":
        return F.cross_entropy(logits, labels, reduction="none")
    if rule == "ce_entropy":
        ce = F.cross_entropy(logits, labels, reduction="none")
        ent = -(student_probs * student_log).sum(dim=-1)
        return ce * ent
    if rule == "forwardkl":
        return (teacher_probs * (teacher_log - student_log)).sum(dim=-1)
    if rule == "reversekl":
        return (student_probs * (student_log - teacher_log)).sum(dim=-1)
    if rule == "logtoku":
        calc = LogTokUCalculator()
        return calc.calculate_au(logits, topk) * calc.calculate_eu(logits, topk)
    if rule == "jsd":
        m = (student_probs + teacher_probs) / 2
        log_m = torch.log(m + 1e-8)
        js_s = (student_probs * (student_log - log_m)).sum(dim=-1)
        js_t = (teacher_probs * (teacher_log - log_m)).sum(dim=-1)
        return 0.5 * (js_s + js_t)
    if rule == "sensitivity_reverse":
        ent_s = -(student_probs * student_log).sum(dim=-1)
        gap = (student_probs * (student_log - teacher_log)).sum(dim=-1)
        return gap.clamp_min(1e-8) / ent_s.clamp_min(1e-8)
    if rule == "gt_prob":
        t_gt = teacher_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        s_gt = student_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (1 - t_gt) * -torch.log(s_gt + 1e-8)
    if rule == "nmtkd":
        return calculate_nmtkd_score(logits, teacher_logits, topk)
    if rule == "hellinger":
        diff = torch.sqrt(student_probs) - torch.sqrt(teacher_probs)
        return torch.sqrt((diff.pow(2)).sum(dim=-1)) / torch.sqrt(torch.tensor(2.0, device=logits.device))
    if rule == "bhattacharyya":
        bc = torch.sum(torch.sqrt(student_probs * teacher_probs), dim=-1)
        return -torch.log(bc + 1e-8)
    raise ValueError(f"Unknown rule: {rule}")

def _select_tokens(scores: torch.Tensor, rule: str, num_selected: int) -> torch.Tensor:
    n = scores.size(0)
    if rule == "all":
        return torch.arange(n, device=scores.device)
    if rule == "random":
        return torch.randperm(n, device=scores.device)[:num_selected]
    _, idx = torch.topk(scores, k=min(num_selected, n), largest=True)
    return idx


def calculate_distillation_weight(
    logits,
    teacher_logits,
    no_model_batch,
    rule: str = "logtoku",
    selection_ratio: float = 0.5,
    topk: int = 5,
    t_base: float = 1.0,
    t_scale: float = 0.5,
    t_direction: bool = True
):
    """
    Calculate the selection mask and adaptive temperature for distillation.
    
    Args:
        logits: [B, L, V]
        teacher_logits: [B, L, V]
        no_model_batch: dict, contains "label"
        rule: str, the rule to calculate the score ("prob", "entropy", "forwardkl", "reversekl", "logtoku", etc.)
        selection_ratio: float, the ratio of tokens to select
        topk: int, the number of top logits to consider for "topk" rule
        t_base: float, base temperature for scaling
        t_scale: float, sensitivity/scaling factor for temperature
        t_direction: bool, whether to soften the temperature (True) or harden it (False)
        
    Returns:
        selected_indices: torch.Tensor, a tensor of indices for selected tokens [num_selected]
        temperature: torch.Tensor, the calculated temperature for each token [B, L]
    """
    b, l, v = logits.shape
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).float()
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).float()

    labels_flat = no_model_batch["label"].view(-1)
    valid_mask = (labels_flat != -100)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        return (torch.empty(0, dtype=torch.long, device=logits.device),
                torch.empty(0, dtype=torch.float32, device=logits.device))

    flat_student = logits.view(-1, v)[valid_indices]
    flat_teacher = teacher_logits.view(-1, v)[valid_indices]
    labels_valid = labels_flat[valid_indices]

    student_probs = F.softmax(flat_student, dim=-1, dtype=torch.float32)
    teacher_probs = F.softmax(flat_teacher, dim=-1, dtype=torch.float32)
    student_log = F.log_softmax(flat_student, dim=-1, dtype=torch.float32)
    teacher_log = F.log_softmax(flat_teacher, dim=-1, dtype=torch.float32)

    num_valid = valid_indices.size(0)
    num_selected = max(1, int(num_valid * selection_ratio))

    scores = _compute_token_scores(
        rule, student_probs, teacher_probs, student_log, teacher_log,
        flat_student, flat_teacher, labels_valid, topk
    )

    sel_local = _select_tokens(scores, rule, num_selected)
    selected_indices = valid_indices[sel_local]
    selected_scores = scores[sel_local]
    median_val = torch.median(selected_scores)
    rel = torch.clamp(selected_scores / median_val, min=1e-8)
    scale = torch.tanh(torch.log(rel))
    if t_direction:
        temperature_valid = t_base * torch.exp(t_scale * scale)
    else:
        temperature_valid = t_base * torch.exp(-t_scale * scale)
    return selected_indices, temperature_valid

DISTILLATION_FUNCTIONS = {
    "fkd": forward_kl,
    "rkd": reverse_kl,
    "jsd": js_distance,
    "tvd": tv_distance,
    "abkd": ab_div,
    "srkd": skewed_reverse_kl,
    "sfkd": skewed_forward_kl,
}
def AdaKD(
    logits, 
    teacher_logits, 
    no_model_batch, 
    loss_fn_name="reverse_kl",
    rule="logtoku",
    selection_ratio=0.5, 
    topk=5, 
    adaptive_temperature=False, 
    review_ratio=0.0, 
    temperature_base=1.0, 
    temperature_scale=0.3, 
    temperature_direction=True,
    # ----
    should_log_stats=False,
    global_step=0,
    log_file_path='logs/temp_entropy_stats.csv',
    # ----
    **loss_fn_kwargs
):
    """
    Calculate the AdaKD loss.
    Args:
        logits: [B, L, V]
        teacher_logits: [B, L, V]
        no_model_batch: dict, contains "label"
        rule: str, the rule to calculate the mask
        selection_ratio: float, the ratio of tokens to select
        topk: int, the number of top logits to consider for "topk" rule
        adaptive_temperature: bool, whether to use adaptive temperature scaling
        review_ratio: float, the ratio of tokens to review
    Returns:
        distil_loss: torch.Tensor, the calculated loss
    """
    if adaptive_temperature and should_log_stats:
        log_temperature_entropy_stats(
            logits=logits,
            teacher_logits=teacher_logits,
            no_model_batch=no_model_batch,
            global_step=global_step,
            log_file_path=log_file_path,
            sample_ratio=selection_ratio,
            temperature_base=temperature_base,
            temperature_scale=temperature_scale,
            temperature_direction=temperature_direction
        )
    if not (0 <= selection_ratio <= 1):
        raise ValueError("selection_ratio must be between 0 and 1.")
    if np.random.uniform(0, 1) < review_ratio:
        rule = "all"

    sel_idx, sel_temp = calculate_distillation_weight(
        logits,
        teacher_logits,
        no_model_batch,
        rule=rule,
        selection_ratio=selection_ratio,
        topk=topk,
        t_base=temperature_base,
        t_scale=temperature_scale,
        t_direction=temperature_direction
    )
    n = sel_idx.numel()
    if n == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    flat_s = logits.view(-1, logits.size(-1))[sel_idx]
    flat_t = teacher_logits.view(-1, teacher_logits.size(-1))[sel_idx]
    if adaptive_temperature and n > 0:
        t_view = sel_temp.unsqueeze(-1)
        flat_s = flat_s / t_view
        flat_t = flat_t / t_view

    pb_s = flat_s.unsqueeze(1)
    pb_t = flat_t.unsqueeze(1)
    pseudo_labels = torch.zeros(n, 1, dtype=torch.long, device=logits.device)
    pseudo_batch = {"label": pseudo_labels}

    if loss_fn_name not in DISTILLATION_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
    per_token_loss = DISTILLATION_FUNCTIONS[loss_fn_name](
        pb_s, pb_t, pseudo_batch, return_per_token_loss=True, **loss_fn_kwargs
    )
    weight = (sel_temp.view(-1) ** 2) if adaptive_temperature else 1.0
    return (per_token_loss * weight).mean()    

def calculate_gradient_conflict(
    logits,
    teacher_logits,
    no_model_batch,
    difficulty_rule: str = "reversekl",
    loss_fn_name: str = "rkd",
    topk: int = 5,
    sample_ratio: float = 0.5,
):
    """
    Calculate gradient conflict between logits and teacher logits
    """
    b, l, v = logits.shape
    logits = logits.detach().clone().requires_grad_(True)
    labels_flat = no_model_batch["label"].view(-1)
    valid_mask = labels_flat != -100
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid_indices.numel() < 4:
        return {}

    flat_s = logits.view(-1, v)[valid_indices]
    flat_t = teacher_logits.view(-1, v)[valid_indices]
    labels_valid = labels_flat[valid_indices]

    s_probs = F.softmax(flat_s, dim=-1, dtype=torch.float32)
    t_probs = F.softmax(flat_t, dim=-1, dtype=torch.float32)
    s_log = F.log_softmax(flat_s, dim=-1, dtype=torch.float32)
    t_log = F.log_softmax(flat_t, dim=-1, dtype=torch.float32)

    scores = _compute_token_scores(
        difficulty_rule, s_probs, t_probs, s_log, t_log,
        flat_s, flat_t, labels_valid, topk
    )

    num_valid = scores.size(0)
    cutoff = int(num_valid * sample_ratio)
    cutoff = max(cutoff, 1)
    sorted_scores, sorted_idx = torch.sort(scores, descending=True)
    top_sel = sorted_idx[:cutoff]
    mid_split = cutoff // 2
    q1 = top_sel[:mid_split]
    q2 = top_sel[mid_split:]
    q3 = sorted_idx[cutoff:]

    groups = {
        "Q1_Hard": q1,
        "Q2_Mid": q2,
        "Q3_Easy": q3
    }

    distill_fn = DISTILLATION_FUNCTIONS.get(loss_fn_name)
    sft_fn = torch.nn.CrossEntropyLoss()

    full_no_model = {"label": labels_valid}
    total_distill = distill_fn(flat_s, flat_t, full_no_model)
    g_distill_full = torch.autograd.grad(total_distill, logits, retain_graph=True)[0].view(-1)

    total_sft = sft_fn(flat_s, labels_valid)
    g_sft_full = torch.autograd.grad(total_sft, logits, retain_graph=True)[0].view(-1)

    results = {}
    norm_parts = {}

    for name, idx in groups.items():
        if idx.numel() == 0:
            continue
        g_logits = flat_s[idx]
        g_teacher = flat_t[idx]
        g_labels = labels_valid[idx]
        g_batch = {"label": g_labels}

        d_loss = distill_fn(g_logits, g_teacher, g_batch)
        s_loss = sft_fn(g_logits, g_labels)

        g_d = torch.autograd.grad(d_loss, logits, retain_graph=True)[0].view(-1)
        g_s = torch.autograd.grad(s_loss, logits, retain_graph=True)[0].view(-1)

        norm_parts[name] = torch.linalg.norm(g_d).item()
        results[name] = {
            "conflict_vs_sft": F.cosine_similarity(g_d, g_s, dim=0, eps=1e-8).item(),
            "contribution_to_total": F.cosine_similarity(g_d, g_distill_full, dim=0, eps=1e-8).item()
        }

    total_norm = sum(norm_parts.values()) or 1e-9
    for name, val in norm_parts.items():
        results[name]["gradient_norm_percentage"] = val / total_norm * 100
    return results

def log_temperature_entropy_stats(
    logits,
    teacher_logits,
    no_model_batch,
    global_step,
    log_file_path,
    sample_ratio=0.8,
    temperature_base=1.0,
    temperature_scale=0.5,
    temperature_direction=False,
):
    """
    
    Args:
        logits:
        teacher_logits: 
        no_model_batch: 
        global_step: 
        log_file_path: 
        sample_ratio: 
        temperature_base: 
        temperature_scale: 
        temperature_direction: 
    """

    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    logits = logits.detach()
    teacher_logits = teacher_logits.detach()
    labels_flat = no_model_batch["label"].view(-1)
    valid_mask = labels_flat != -100
    vidx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if vidx.numel() < 10:
        return
    flat_s = logits.view(-1, logits.size(-1))[vidx]
    flat_t = teacher_logits.view(-1, teacher_logits.size(-1))[vidx]

    s_probs = F.softmax(flat_s, dim=-1, dtype=torch.float32)
    t_probs = F.softmax(flat_t, dim=-1, dtype=torch.float32)
    diff = torch.sqrt(s_probs) - torch.sqrt(t_probs)
    hell = torch.sqrt((diff.pow(2)).sum(dim=-1)) / np.sqrt(2.0)

    n = hell.numel()
    keep = int(n * sample_ratio)
    if keep == 0:
        return
    sorted_scores, order = torch.sort(hell, descending=True)
    kept_scores = sorted_scores[:keep]
    kept_logits = flat_s[order[:keep]]
    med = torch.median(kept_scores)
    hard_mask = kept_scores > med
    easy_mask = ~hard_mask

    def _entropy(x):
        p = F.softmax(x, dim=-1, dtype=torch.float32).clamp_min(1e-9)
        return -(p * torch.log(p)).sum(dim=-1)

    rows = []
    for group, mask in (("Hard", hard_mask), ("Easy", easy_mask)):
        if mask.sum() == 0:
            continue
        g_scores = kept_scores[mask]
        g_logits = kept_logits[mask]
        ent_before = _entropy(g_logits)
        rel = torch.clamp(g_scores / med, min=1e-8)
        scale = torch.tanh(torch.log(rel))
        temp = temperature_base * torch.exp(
            (temperature_scale if temperature_direction else -temperature_scale) * scale
        )
        logits_after = g_logits / temp.unsqueeze(-1)
        ent_after = _entropy(logits_after)
        for i in range(g_scores.size(0)):
            rows.append({
                "global_step": global_step,
                "token_group": group,
                "temperature": temp[i].item(),
                "entropy_before": ent_before[i].item(),
                "entropy_after": ent_after[i].item()
            })
    if not rows:
        return
    file_exists = os.path.isfile(log_file_path)
    import csv
    with open(log_file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_step", "token_group", "temperature", "entropy_before", "entropy_after"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)