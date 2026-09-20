"""
Unit tests for katago.train.metrics_pytorch.policy_distill_loss — the opt-in
temperature / Decoupled-KD generalization of the main policy cross-entropy.

Checks:
  1. Baseline equivalence: temperature=1, DKD off  ==  plain cross_entropy (exact).
  2. Temperature identity: if the student matches the teacher (logits = log target),
     the KD loss equals T^2 * H(teacher_T) for any T (no spurious gradient signal).
  3. DKD decomposition (Zhao 2022): at T=1, TCKD + p_nontarget^teacher * NCKD == KL(teacher||student),
     the exact identity classic KD is a special case of. Validates the TCKD/NCKD split.
  4. Shape: returns one scalar loss per sample.
"""

import torch

from katago.train.metrics_pytorch import cross_entropy, policy_distill_loss


def _random_case(n=16, c=362, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, c, generator=g, dtype=torch.float64)
    # A soft, normalized teacher distribution (like MCTS visit fractions).
    target = torch.softmax(torch.randn(n, c, generator=g, dtype=torch.float64) * 1.5, dim=1)
    return logits, target


def test_baseline_equivalence():
    logits, target = _random_case(seed=1)
    base = cross_entropy(logits, target, dim=1)
    got = policy_distill_loss(logits, target, temperature=1.0, dkd_alpha=None, dkd_beta=None, dim=1)
    assert torch.allclose(got, base, atol=1e-12), "T=1, DKD off must equal cross_entropy exactly"


def test_temperature_matches_entropy_when_student_equals_teacher():
    _, target = _random_case(seed=2)
    eps = 1.0e-30
    logits = torch.log(target.clamp_min(eps))  # student == teacher at any temperature
    for T in (0.5, 2.0, 4.0):
        teacher_T = torch.softmax(torch.log(target.clamp_min(eps)) / T, dim=1)
        entropy_T = -torch.sum(teacher_T * torch.log(teacher_T.clamp_min(eps)), dim=1)
        got = policy_distill_loss(logits, target, temperature=T, dkd_alpha=None, dkd_beta=None, dim=1)
        assert torch.allclose(got, (T * T) * entropy_T, atol=1e-9), f"temperature KD identity failed at T={T}"


def test_dkd_decomposition_reconstructs_kl():
    logits, target = _random_case(seed=3)
    # Direct KL(teacher || student) = cross_entropy - H(teacher).
    eps = 1.0e-30
    log_student = torch.log_softmax(logits, dim=1)
    kl = torch.sum(target * (torch.log(target.clamp_min(eps)) - log_student), dim=1)

    # At T=1: alpha=1,beta=0 -> pure TCKD;  alpha=0,beta=1 -> pure NCKD.
    tckd = policy_distill_loss(logits, target, temperature=1.0, dkd_alpha=1.0, dkd_beta=0.0, dim=1)
    nckd = policy_distill_loss(logits, target, temperature=1.0, dkd_alpha=0.0, dkd_beta=1.0, dim=1)

    p_target = target.max(dim=1).values            # teacher prob of its argmax move
    p_nontarget = 1.0 - p_target                   # the classic-KD coupling weight
    reconstructed = tckd + p_nontarget * nckd
    assert torch.allclose(reconstructed, kl, atol=1e-9), "TCKD + p_nt*NCKD must equal KL(teacher||student)"


def test_returns_one_loss_per_sample():
    logits, target = _random_case(n=7, seed=4)
    for kwargs in (
        dict(temperature=1.0, dkd_alpha=None, dkd_beta=None),
        dict(temperature=2.0, dkd_alpha=None, dkd_beta=None),
        dict(temperature=2.0, dkd_alpha=1.0, dkd_beta=4.0),
    ):
        out = policy_distill_loss(logits, target, dim=1, **kwargs)
        assert out.shape == (7,)
        assert torch.isfinite(out).all()
