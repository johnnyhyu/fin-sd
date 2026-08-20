"""
loss_script.py — computes token-averaged KL divergence between two logit tensors.

KL(P_hint ‖ P_nohint) measures how much information is lost going from the
hint-conditioned distribution to the unhinted one. Minimising this loss pulls
the model's un-hinted behaviour toward its hint-conditioned behaviour.

Direction chosen:
    P  = softmax(logits_with_hint)    — teacher / target (detached, no grad)
    Q  = softmax(logits_without_hint) — student (gradient flows through this)

    KL(P ‖ Q) = Σ_v  P(v) * [log P(v) − log Q(v)]

Using torch.nn.functional.kl_div with reduction="batchmean" gives the
per-token mean (equivalent to averaging over the N completion positions).

Alongside the KL family (forward / reverse / adaptive / top-k / EXOPD variants),
run_length_penalty implements the token-budget term (config.LENGTH_PENALTY) —
the one objective here that is not a divergence between two distributions.
"""
import torch
import torch.nn.functional as F


# Stand-in for -inf when dropping padded top-k entries from a renormalised
# support. It must be FINITE. Masking with float("-inf") gives the right forward
# value but a NaN gradient: at a padded entry the renormalised logs are both -inf,
# so the log-difference is (-inf) − (-inf) = NaN, and while masked_fill hides that
# in the forward, the backward still multiplies the zeroed upstream gradient by
# that NaN (0 · NaN = NaN) and feeds it into the row's logsumexp — which sums
# across the row and poisons every entry in it, then the item's whole logit row,
# then (through the shared batch backward) the entire parameter gradient. AdamW
# writes NaN weights from there and the run is dead, with nothing in the logs.
# exp(-1e9 − logsumexp) underflows to exactly 0.0 in float32, so the padded
# entries still carry no mass and the renormalisation is unchanged.
_MASK_NEG = -1e9


def _mean_over_valid_positions(
    term: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Sum each position's top-k term, then average over the POPULATED positions.

    Shared by the three teacher-top-k losses (the vLLM probing path). A position
    where the server reported no alternatives has an all-zero mask row and
    contributes exactly 0 to the sum; averaging over all N would then scale both
    the reported loss and the gradient by the fraction of such positions, so the
    effective step size would track a server-side reporting artifact rather than
    the objective. Average over the rows that carry a teacher distribution instead.

    Args:
        term: Float tensor (N, k) — the per-entry KL integrand, already zeroed at
              masked entries by the caller.
        mask: Optional float tensor (N, k) — 1 for valid entries, 0 for padding.

    Returns:
        Scalar tensor; gradients flow through `term`.
    """
    per_pos = term.sum(dim=-1)                                       # (N,)
    if mask is None:
        return per_pos.mean()
    valid = (mask.to(device=term.device, dtype=torch.float).sum(dim=-1) > 0).float()
    n_valid = valid.sum()
    if float(n_valid) == 0.0:
        raise ValueError(
            "Every teacher top-k row is masked out: vLLM returned no alternatives "
            "at any position, so there is no teacher distribution to distil."
        )
    return (per_pos * valid).sum() / n_valid


@torch.no_grad()
def token_entropy(logits: torch.Tensor) -> float:
    """Mean per-token Shannon entropy (nats) of softmax(logits).

    A training diagnostic (config.LOG_ENTROPY), not part of any objective — how
    peaked the per-position distribution is, averaged over the N completion
    tokens. Detached and returned as a plain float, ready for wandb.log.

    Args:
        logits: Float tensor (N, V) — the per-position logits to score.

    Returns:
        Scalar Python float: −Σ_v p(v)·log p(v) averaged over the N positions.
    """
    log_p = F.log_softmax(logits.float(), dim=-1)   # (N, V)
    return float(-(log_p.exp() * log_p).sum(dim=-1).mean())


def run_loss(
    logits_with_hint: torch.Tensor,
    logits_without_hint: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(P_hint ‖ P_nohint) averaged over completion tokens.

    Args:
        logits_with_hint:    Float tensor of shape (N, V) — teacher logits.
                             Expected to have no gradient (from probing's no-grad pass).
        logits_without_hint: Float tensor of shape (N, V) — student logits.
                             Must carry a gradient for backprop to work.

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if logits_with_hint.shape != logits_without_hint.shape:
        raise ValueError(
            f"Logit shapes must match. Got {logits_with_hint.shape} vs {logits_without_hint.shape}."
        )

    # Teacher distribution — treated as a fixed constant target.
    p = F.softmax(logits_with_hint.detach().float(), dim=-1)   # (N, V)

    # Log-probabilities for the student — gradients flow through here.
    log_q = F.log_softmax(logits_without_hint.float(), dim=-1)  # (N, V)

    # F.kl_div(input=log_q, target=p) computes Σ p*(log p − log_q) / N
    # reduction="batchmean" divides by the batch size (N here = number of tokens)
    loss: torch.Tensor = F.kl_div(log_q, p, reduction="batchmean")
    return loss


def run_reverse_loss(
    logits_with_hint: torch.Tensor,
    logits_without_hint: torch.Tensor,
) -> torch.Tensor:
    """Compute reverse KL(Q_student ‖ P_teacher) averaged over completion tokens.

    The mode-seeking counterpart to run_loss, used by the on-policy
    student-rollout path (config.STUDENT_ROLLOUT). The completion tokens are
    sampled from the student's own (without-hint) distribution, and this loss
    pulls that student distribution toward the teacher's:

        KL(Q ‖ P) = Σ_v  Q(v) * [log Q(v) − log P(v)]

    Q = softmax(logits_without_hint) is the student (gradient flows through both
    the Q weighting and log Q — i.e. the entropy term); P = softmax(logits_with_hint)
    is the teacher, detached. This variant sums over the full vocab; see
    run_reverse_topk_loss for the mode-seeking partial sum restricted to the
    teacher's top-k token ids.

    Args:
        logits_with_hint:    Float tensor (N, V) — teacher logits, no grad needed.
        logits_without_hint: Float tensor (N, V) — student logits, must carry grad.

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if logits_with_hint.shape != logits_without_hint.shape:
        raise ValueError(
            f"Logit shapes must match. Got {logits_with_hint.shape} vs {logits_without_hint.shape}."
        )

    # Student log-probs — gradients flow through here (both q and log q below).
    log_q = F.log_softmax(logits_without_hint.float(), dim=-1)          # (N, V)
    # Teacher log-probs — fixed target, no grad.
    log_p = F.log_softmax(logits_with_hint.detach().float(), dim=-1)    # (N, V)

    q = log_q.exp()                                                     # (N, V)
    # Sum over vocab, mean over the N completion positions.
    loss: torch.Tensor = (q * (log_q - log_p)).sum(dim=-1).mean()
    return loss


def run_adaptive_loss(
    logits_with_hint: torch.Tensor,
    logits_without_hint: torch.Tensor,
    forward_weight: float,
) -> torch.Tensor:
    """Weighted blend of forward KL(P ‖ Q) and reverse KL(Q ‖ P) over the full vocab.

    The adaptive-KL objective (config.ADAPTIVE_KL): rather than committing to one
    direction, compute both and return

        L = forward_weight · KL(P ‖ Q) + (1 − forward_weight) · KL(Q ‖ P)

    where P = softmax(logits_with_hint) is the teacher (detached) and
    Q = softmax(logits_without_hint) is the student (gradient flows through both
    the Q weighting and log Q). config.adaptive_kl_forward_weight anneals
    forward_weight from 1.0 (pure mass-covering forward KL, == run_loss) to 0.0
    (pure mode-seeking reverse KL, == run_reverse_loss) across training epochs.
    The full-vocab counterpart to run_adaptive_topk_loss.

    Args:
        logits_with_hint:    Float tensor (N, V) — teacher logits, no grad needed.
        logits_without_hint: Float tensor (N, V) — student logits, must carry grad.
        forward_weight:      Weight on forward KL in [0, 1]; reverse gets 1 − it.

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if logits_with_hint.shape != logits_without_hint.shape:
        raise ValueError(
            f"Logit shapes must match. Got {logits_with_hint.shape} vs {logits_without_hint.shape}."
        )

    # Student log-probs — gradients flow through here (both q and log q below).
    log_q = F.log_softmax(logits_without_hint.float(), dim=-1)          # (N, V)
    # Teacher log-probs — fixed target, no grad.
    log_p = F.log_softmax(logits_with_hint.detach().float(), dim=-1)    # (N, V)

    p = log_p.exp()                                                     # (N, V)
    q = log_q.exp()                                                     # (N, V)
    forward = (p * (log_p - log_q)).sum(dim=-1).mean()   # KL(P ‖ Q), mass-covering
    reverse = (q * (log_q - log_p)).sum(dim=-1).mean()   # KL(Q ‖ P), mode-seeking
    return forward_weight * forward + (1.0 - forward_weight) * reverse


def run_reverse_topk_loss(
    logits_with_hint: torch.Tensor,
    logits_without_hint: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Reverse KL(Q_student ‖ P_teacher) over the teacher's renormalised top-k.

    The top-k counterpart to run_reverse_loss, for the student-rollout path
    (config.STUDENT_ROLLOUT with STUDENT_ROLLOUT_BIASED). Instead of summing the
    reverse-KL integrand over the full vocab, it restricts both distributions to
    the teacher's top-k token ids per position and *renormalises* each over that
    support, so Q̃ and P̃ are proper distributions over the top-k and the result
    is a true KL divergence between them:

        Q̃(v) = Q(v) / Σ_{u ∈ topk} Q(u),   P̃(v) = P(v) / Σ_{u ∈ topk} P(u)
        L = (1/N) Σ_i Σ_{v ∈ topk_i}  Q̃(v) · [log Q̃(v) − log P̃(v)]

    The top-k ids are selected from the teacher's per-position distribution.
    Q = softmax(logits_without_hint) is the student (gradient flows through the
    Q̃ weighting and log Q̃) and P = softmax(logits_with_hint) is the teacher
    (detached). Both are first computed over the full vocab, then restricted and
    renormalised over the teacher's top-k support.

    Args:
        logits_with_hint:    Float tensor (N, V) — teacher logits, no grad needed.
        logits_without_hint: Float tensor (N, V) — student logits, must carry grad.
        k:                   Number of teacher top-k ids to sum over (clamped to V).

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if logits_with_hint.shape != logits_without_hint.shape:
        raise ValueError(
            f"Logit shapes must match. Got {logits_with_hint.shape} vs {logits_without_hint.shape}."
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    # Student log-probs — gradients flow through here (both q and log q below).
    log_q = F.log_softmax(logits_without_hint.float(), dim=-1)          # (N, V)
    # Teacher log-probs — fixed target, no grad.
    log_p = F.log_softmax(logits_with_hint.detach().float(), dim=-1)    # (N, V)

    # Teacher's top-k token ids per position define the summation support.
    k = min(k, log_p.shape[-1])
    topk_ids = log_p.topk(k, dim=-1).indices                           # (N, k)

    log_q_k = log_q.gather(-1, topk_ids)                               # (N, k)
    log_p_k = log_p.gather(-1, topk_ids)                               # (N, k)

    # Renormalise both distributions over the top-k support so each is a proper
    # distribution and L is a true KL divergence (not a partial sum).
    log_q_k = log_q_k - torch.logsumexp(log_q_k, dim=-1, keepdim=True)  # (N, k)
    log_p_k = log_p_k - torch.logsumexp(log_p_k, dim=-1, keepdim=True)  # (N, k)
    q_k = log_q_k.exp()                                                # (N, k)

    # Sum over the teacher's top-k support, mean over the N completion positions.
    loss: torch.Tensor = (q_k * (log_q_k - log_p_k)).sum(dim=-1).mean()
    return loss


def run_adaptive_topk_loss(
    logits_with_hint: torch.Tensor,
    logits_without_hint: torch.Tensor,
    k: int,
    forward_weight: float,
) -> torch.Tensor:
    """Weighted blend of forward and reverse KL over the teacher's renormalised top-k.

    The top-k counterpart to run_adaptive_loss (config.ADAPTIVE_KL with a top-k
    flag — STUDENT_ROLLOUT_BIASED / TEACHER_ROLLOUT_BIASED). Both distributions are
    restricted to the teacher's top-k token ids per position and renormalised over
    that support (so P̃ and Q̃ are proper distributions and each direction is a
    true KL), then blended:

        L = forward_weight · KL(P̃ ‖ Q̃) + (1 − forward_weight) · KL(Q̃ ‖ P̃)

    forward_weight=1 recovers the renormalised top-k forward KL and 0 recovers
    run_reverse_topk_loss. Q = softmax(logits_without_hint) is the student
    (gradient flows through the Q̃ weighting and log Q̃); P = softmax(logits_with_hint)
    is the teacher (detached).

    Args:
        logits_with_hint:    Float tensor (N, V) — teacher logits, no grad needed.
        logits_without_hint: Float tensor (N, V) — student logits, must carry grad.
        k:                   Number of teacher top-k ids to sum over (clamped to V).
        forward_weight:      Weight on forward KL in [0, 1]; reverse gets 1 − it.

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if logits_with_hint.shape != logits_without_hint.shape:
        raise ValueError(
            f"Logit shapes must match. Got {logits_with_hint.shape} vs {logits_without_hint.shape}."
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    # Student log-probs — gradients flow through here (both q and log q below).
    log_q = F.log_softmax(logits_without_hint.float(), dim=-1)          # (N, V)
    # Teacher log-probs — fixed target, no grad.
    log_p = F.log_softmax(logits_with_hint.detach().float(), dim=-1)    # (N, V)

    # Teacher's top-k token ids per position define the summation support.
    k = min(k, log_p.shape[-1])
    topk_ids = log_p.topk(k, dim=-1).indices                           # (N, k)

    log_q_k = log_q.gather(-1, topk_ids)                               # (N, k)
    log_p_k = log_p.gather(-1, topk_ids)                               # (N, k)

    # Renormalise both distributions over the top-k support so each direction is a
    # true KL divergence (not a partial sum).
    log_q_k = log_q_k - torch.logsumexp(log_q_k, dim=-1, keepdim=True)  # (N, k)
    log_p_k = log_p_k - torch.logsumexp(log_p_k, dim=-1, keepdim=True)  # (N, k)
    q_k = log_q_k.exp()                                                # (N, k)
    p_k = log_p_k.exp()                                                # (N, k)

    forward = (p_k * (log_p_k - log_q_k)).sum(dim=-1).mean()   # KL(P̃ ‖ Q̃)
    reverse = (q_k * (log_q_k - log_p_k)).sum(dim=-1).mean()   # KL(Q̃ ‖ P̃)
    return forward_weight * forward + (1.0 - forward_weight) * reverse


def run_reverse_topk_loss_from_teacher(
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Biased reverse KL(Q_student ‖ P_teacher) over the teacher's renormalised top-k.

    The reverse-direction counterpart to run_topk_loss, for the teacher-rollout
    path when config.TEACHER_ROLLOUT_REVERSE + TEACHER_ROLLOUT_BIASED are set on the
    default vLLM top-k probing path (config.PROBE_TOPK_VIA_VLLM). It computes the
    same renormalised top-k reverse KL as run_reverse_topk_loss, but takes the
    teacher distribution as the vLLM top-k representation (ids + their full-vocab
    log-probs) instead of slicing it out of full teacher logits — because that path
    never materialises the teacher's full distribution.

        Q̃(v) = Q(v) / Σ_{u ∈ topk} Q(u),   P̃(v) = P(v) / Σ_{u ∈ topk} P(u)
        L = (1/N) Σ_i Σ_{v ∈ topk_i}  Q̃(v) · [log Q̃(v) − log P̃(v)]

    The top-k support is the teacher's top-k ids per position. log P̃ comes from the
    teacher's reported log-probs at those ids (renormalised over the masked top-k);
    log Q̃ is gathered from the student's full-vocab log-softmax at the same ids
    (renormalised likewise), so gradients flow only through those k columns.

    Args:
        teacher_topk_ids:      Long tensor (N, k) — teacher top-k token ids, no grad.
        teacher_topk_logprobs: Float tensor (N, k) — teacher log P at those ids, no grad.
        student_logits:        Float tensor (N, V) — student logits, must carry grad.
        mask:                  Optional float tensor (N, k) — 1 for valid entries,
                               0 for padding (positions with < k alternatives).
                               Padded entries are dropped from the renormalised
                               support so they contribute nothing.

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError(
            f"teacher_topk_ids {teacher_topk_ids.shape} and teacher_topk_logprobs "
            f"{teacher_topk_logprobs.shape} must match."
        )
    if teacher_topk_ids.shape[0] != student_logits.shape[0]:
        raise ValueError(
            f"Position count mismatch: teacher has {teacher_topk_ids.shape[0]} rows, "
            f"student_logits has {student_logits.shape[0]}."
        )

    device = student_logits.device
    ids = teacher_topk_ids.to(device=device, dtype=torch.long)                    # (N, k)
    # Teacher log P at the top-k ids — fixed target, no grad.
    log_p_k = teacher_topk_logprobs.to(device=device, dtype=torch.float).detach()  # (N, k)

    # Student log q at the teacher's top-k ids (gathered from the full-vocab
    # log_softmax without materialising the (N, V) fp32 intermediate). Grad flows
    # through here (both the Q̃ weighting and log Q̃ below).
    logits_f = student_logits.float()                                            # (N, V)
    log_q_k = logits_f.gather(-1, ids) - torch.logsumexp(logits_f, dim=-1, keepdim=True)  # (N, k)

    # Drop padded entries from the support: push them far below the real entries
    # before renormalising so they fall out of both logsumexp totals (and hence
    # carry zero mass). _MASK_NEG rather than -inf — see its comment.
    mask_f = None
    if mask is not None:
        mask_f = mask.to(device=device, dtype=torch.float)                        # (N, k)
        invalid = mask_f == 0
        log_q_k = log_q_k.masked_fill(invalid, _MASK_NEG)
        log_p_k = log_p_k.masked_fill(invalid, _MASK_NEG)

    # Renormalise both over the (masked) top-k support so each is a proper
    # distribution and L is a true KL divergence (not a partial sum).
    log_q_k = log_q_k - torch.logsumexp(log_q_k, dim=-1, keepdim=True)            # (N, k)
    log_p_k = log_p_k - torch.logsumexp(log_p_k, dim=-1, keepdim=True)            # (N, k)
    q_k = log_q_k.exp()                                                           # (N, k)

    term = q_k * (log_q_k - log_p_k)                                             # (N, k)
    if mask_f is not None:
        # Multiply, don't masked_fill: both are exact zeros in the forward, but
        # multiplication also gives the padded entries a finite (zero) gradient.
        term = term * mask_f
    # Sum over the teacher's top-k support, mean over the POPULATED positions.
    loss: torch.Tensor = _mean_over_valid_positions(term, mask)
    return loss


def run_adaptive_topk_loss_from_teacher(
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    student_logits: torch.Tensor,
    forward_weight: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted blend of forward and reverse KL over the teacher's renormalised top-k.

    The adaptive-KL (config.ADAPTIVE_KL) counterpart to
    run_reverse_topk_loss_from_teacher, for the default vLLM top-k probing path
    (config.PROBE_TOPK_VIA_VLLM) which never materialises the teacher's full
    distribution. Both distributions are restricted to the teacher's top-k ids per
    position and renormalised over that (masked) support, then blended:

        L = forward_weight · KL(P̃ ‖ Q̃) + (1 − forward_weight) · KL(Q̃ ‖ P̃)

    forward_weight=1 gives the renormalised top-k forward KL and 0 recovers
    run_reverse_topk_loss_from_teacher. log P̃ comes from the teacher's reported
    top-k log-probs; log Q̃ is gathered from the student's full-vocab log-softmax
    at the same ids, so gradients flow only through those k columns.

    Args:
        teacher_topk_ids:      Long tensor (N, k) — teacher top-k token ids, no grad.
        teacher_topk_logprobs: Float tensor (N, k) — teacher log P at those ids, no grad.
        student_logits:        Float tensor (N, V) — student logits, must carry grad.
        forward_weight:        Weight on forward KL in [0, 1]; reverse gets 1 − it.
        mask:                  Optional float tensor (N, k) — 1 for valid entries,
                               0 for padding (positions with < k alternatives).
                               Padded entries are dropped from the renormalised
                               support so they contribute nothing.

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError(
            f"teacher_topk_ids {teacher_topk_ids.shape} and teacher_topk_logprobs "
            f"{teacher_topk_logprobs.shape} must match."
        )
    if teacher_topk_ids.shape[0] != student_logits.shape[0]:
        raise ValueError(
            f"Position count mismatch: teacher has {teacher_topk_ids.shape[0]} rows, "
            f"student_logits has {student_logits.shape[0]}."
        )

    device = student_logits.device
    ids = teacher_topk_ids.to(device=device, dtype=torch.long)                    # (N, k)
    # Teacher log P at the top-k ids — fixed target, no grad.
    log_p_k = teacher_topk_logprobs.to(device=device, dtype=torch.float).detach()  # (N, k)

    # Student log q at the teacher's top-k ids (gathered from the full-vocab
    # log_softmax without materialising the (N, V) fp32 intermediate). Grad flows
    # through here (both the Q̃ weighting and log Q̃ below).
    logits_f = student_logits.float()                                            # (N, V)
    log_q_k = logits_f.gather(-1, ids) - torch.logsumexp(logits_f, dim=-1, keepdim=True)  # (N, k)

    # Drop padded entries from the support before renormalising (see
    # run_reverse_topk_loss_from_teacher and _MASK_NEG).
    mask_f = None
    if mask is not None:
        mask_f = mask.to(device=device, dtype=torch.float)                        # (N, k)
        invalid = mask_f == 0
        log_q_k = log_q_k.masked_fill(invalid, _MASK_NEG)
        log_p_k = log_p_k.masked_fill(invalid, _MASK_NEG)

    # Renormalise both over the (masked) top-k support so each direction is a true
    # KL divergence (not a partial sum).
    log_q_k = log_q_k - torch.logsumexp(log_q_k, dim=-1, keepdim=True)            # (N, k)
    log_p_k = log_p_k - torch.logsumexp(log_p_k, dim=-1, keepdim=True)            # (N, k)
    q_k = log_q_k.exp()                                                           # (N, k)
    p_k = log_p_k.exp()                                                           # (N, k)

    term = forward_weight * (p_k * (log_p_k - log_q_k)) \
        + (1.0 - forward_weight) * (q_k * (log_q_k - log_p_k))                    # (N, k)
    if mask_f is not None:
        # Multiply, don't masked_fill — see run_reverse_topk_loss_from_teacher.
        term = term * mask_f
    # Sum over the teacher's top-k support, mean over the POPULATED positions.
    loss: torch.Tensor = _mean_over_valid_positions(term, mask)
    return loss


def run_exopd_loss(
    logits_with_hint: torch.Tensor,
    logits_without_hint: torch.Tensor,
    logits_ref: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Generalized Off-Policy Distillation (EXOPD) loss over completion tokens.

    The student-rollout-only objective (config.EXOPD). The completion tokens are
    sampled from the student's own (without-hint) distribution, and this maximises

        J = E_{y~π_θ}[ λ·log(π*(y)/π_ref(y)) − KL(π_θ ‖ π_ref) ]

    where π* is the with-hint teacher, π_ref is the base model on the without-hint
    prefix (adapter disabled), and π_θ is the live student. Realised as the
    analytic per-position expectation under π_θ (like run_reverse_loss):

        J_pos = Σ_v q_v · [ λ·(log*_v − log_ref_v) − (log_θ_v − log_ref_v) ]
              = λ·E_q[log(π*/π_ref)] − KL(π_θ ‖ π_ref)
        loss  = − mean_over_positions(J_pos)

    q = softmax(logits_without_hint) is the student (gradient flows through both
    the q weighting and log q); the teacher (logits_with_hint) and reference
    (logits_ref) distributions are detached targets.

    Args:
        logits_with_hint:    Float tensor (N, V) — teacher (π*) logits, no grad needed.
        logits_without_hint: Float tensor (N, V) — student (π_θ) logits, must carry grad.
        logits_ref:          Float tensor (N, V) — reference (π_ref) logits, no grad needed.
        lam:                 λ, the weight on the teacher/reference log-ratio reward.

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if not (logits_with_hint.shape == logits_without_hint.shape == logits_ref.shape):
        raise ValueError(
            f"Logit shapes must match. Got teacher {logits_with_hint.shape}, "
            f"student {logits_without_hint.shape}, ref {logits_ref.shape}."
        )

    # Student log-probs — gradients flow through here (both q and log q below).
    log_theta = F.log_softmax(logits_without_hint.float(), dim=-1)            # (N, V)
    # Teacher / reference — fixed targets, no grad.
    log_star = F.log_softmax(logits_with_hint.detach().float(), dim=-1)       # (N, V)
    log_ref = F.log_softmax(logits_ref.detach().float(), dim=-1)             # (N, V)

    q = log_theta.exp()                                                       # (N, V)
    # J_pos = Σ_v q·[ λ(log* − log_ref) − (log_θ − log_ref) ]; loss = −mean(J_pos).
    reward = lam * (log_star - log_ref)                                      # (N, V)
    kl_term = log_theta - log_ref                                            # (N, V)
    j_pos = (q * (reward - kl_term)).sum(dim=-1)                             # (N,)
    loss: torch.Tensor = -j_pos.mean()
    return loss


def run_length_penalty(
    student_logits: torch.Tensor,
    end_token_ids: torch.Tensor,
    position_weights: torch.Tensor,
) -> torch.Tensor:
    """Token-budget penalty: weighted NLL of ENDING the turn, over a rollout's tail.

    The objective for rollouts that ran into (or close to) config.MAX_NEW_TOKENS
    (config.LENGTH_PENALTY). Those rollouts carry no teacher target — nothing was
    distilled onto them, because the failure is "never finished", not "reasoned
    wrongly" — so instead of a KL this scores the one behaviour that was missing:

        p_end(i) = Σ_{t ∈ end_token_ids} π_θ(t | y_<i)
        L        = (1/M) Σ_i  w_i · ( −log p_end(i) )

    i.e. at each scored position, how surprised the student would be to wrap the
    turn up there, weighted by how far past the soft budget that position is.
    Minimising it makes ending the turn progressively more available the deeper
    into the budget the rollout gets.

    Three properties this relies on (see config.LENGTH_PENALTY for the full
    rationale):
      • w_i is 0 below the soft threshold, so positions that are not over-long
        contribute neither loss nor gradient — the term cannot reward stopping
        early.
      • The mean is over the M scored positions, so the gradient magnitude does
        not scale with how far the rollout overran; severity lives in w_i alone.
      • −log p_end is a cross-entropy, so its gradient w.r.t. the logits is
        bounded (‖·‖₁ ≤ 2 per position) no matter how small p_end is. That is why
        it is left uncapped: clamping the value would kill the gradient exactly at
        the positions where the model is most committed to continuing, and the
        bound is what keeps one pathological item from dominating the shared
        clip_grad_norm_ window.

    Args:
        student_logits:   Float tensor (M, V) — student logits at the scored
                          positions, must carry grad. Row i is the distribution
                          the model used to pick rollout token i.
        end_token_ids:    Long tensor (S,) — ids that end the turn
                          (prompts.turn_end_token_ids). Must be non-empty.
        position_weights: Float tensor (M,) — w_i in [0, 1], one per scored
                          position (config.length_penalty_weight).

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if student_logits.ndim != 2:
        raise ValueError(f"student_logits must be (M, V); got {tuple(student_logits.shape)}.")
    if end_token_ids.numel() == 0:
        raise ValueError(
            "end_token_ids is empty: there is no turn-ending token to train toward, "
            "so the length penalty has no target (see prompts.turn_end_token_ids)."
        )
    if position_weights.shape[0] != student_logits.shape[0]:
        raise ValueError(
            f"position_weights has {position_weights.shape[0]} entries but "
            f"student_logits has {student_logits.shape[0]} rows."
        )

    device = student_logits.device
    ids = end_token_ids.to(device=device, dtype=torch.long).reshape(-1)      # (S,)
    w = position_weights.to(device=device, dtype=torch.float)                # (M,)

    logits_f = student_logits.float()                                        # (M, V)
    # log p_end = logsumexp over the end-token columns − logsumexp over the vocab.
    # Computed in log space (never as a sum of exponentiated probabilities) so a
    # confidently-continuing position underflows gracefully to a large finite NLL
    # instead of log(0).
    log_norm = torch.logsumexp(logits_f, dim=-1)                             # (M,)
    log_p_end = torch.logsumexp(logits_f.index_select(-1, ids), dim=-1) - log_norm

    # Mean over the scored positions, NOT over the non-zero-weight ones: a window
    # that is only partly over the threshold should contribute proportionally less.
    loss: torch.Tensor = (w * (-log_p_end)).mean()
    return loss


def run_topk_loss(
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Partial-sum top-k KL(P_hint ‖ P_nohint) averaged over completion tokens.

    Unlike run_loss (which sums over the full vocab), this evaluates the KL
    integrand only at the teacher's top-k token ids per position and sums those
    contributions *without renormalising* the support — a partial sum of the
    full-vocab KL restricted to where the teacher places most of its mass:

        L = (1/N) Σ_i Σ_{v ∈ topk_i}  P(v) · [log P(v) − log Q(v)]

    P(v) and log P(v) come straight from the teacher's (full-vocab-normalised)
    logprobs; log Q(v) is gathered from the student's full-vocab log-softmax at
    the same ids, so gradients flow only through those k columns per position.

    Args:
        teacher_topk_ids:      Long tensor (N, k) — teacher top-k token ids, no grad.
        teacher_topk_logprobs: Float tensor (N, k) — teacher log P at those ids, no grad.
        student_logits:        Float tensor (N, V) — student logits, must carry grad.
        mask:                  Optional float tensor (N, k) — 1 for valid entries,
                               0 for padding (positions with < k alternatives).

    Returns:
        Scalar tensor with gradient attached (ready for .backward()).
    """
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError(
            f"teacher_topk_ids {teacher_topk_ids.shape} and teacher_topk_logprobs "
            f"{teacher_topk_logprobs.shape} must match."
        )
    if teacher_topk_ids.shape[0] != student_logits.shape[0]:
        raise ValueError(
            f"Position count mismatch: teacher has {teacher_topk_ids.shape[0]} rows, "
            f"student_logits has {student_logits.shape[0]}."
        )

    device = student_logits.device
    ids = teacher_topk_ids.to(device=device, dtype=torch.long)
    log_p = teacher_topk_logprobs.to(device=device, dtype=torch.float).detach()  # (N, k)
    p = log_p.exp()                                                              # (N, k)

    # Student log-probs at the teacher's top-k ids. Equivalent to gathering from
    # the full-vocab log_softmax, but without materialising the (N, V) fp32
    # intermediate (and its backward buffer): log q = logit - logsumexp(logits).
    logits_f = student_logits.float()                                            # (N, V)
    log_q = logits_f.gather(-1, ids) - torch.logsumexp(logits_f, dim=-1, keepdim=True)  # (N, k)

    term = p * (log_p - log_q)                                                   # (N, k)
    if mask is not None:
        term = term * mask.to(device=device, dtype=torch.float)
    # Sum over the top-k support, mean over the POPULATED completion positions.
    loss: torch.Tensor = _mean_over_valid_positions(term, mask)
    return loss
