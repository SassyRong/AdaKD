from .losses import forward_kl, reverse_kl, symmetric_kl, js_distance, tv_distance, adaptive_kl, sinkhorn_kl
from .losses import skewed_forward_kl, skewed_reverse_kl
from .losses import AdaKD, calculate_distillation_weight, calculate_gradient_conflict, analyze_idts_gradient_effect
from .losses import ab_div
from .temperature import Global_T, CosineScheduler