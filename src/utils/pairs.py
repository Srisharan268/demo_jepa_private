"""One-frame training pairs from an encoded window (stages 0 and 2, "pair mode").

Deploy plans from ONE frame of context, one step ahead: the predictor is called
with T=1 (mpc.rollout 1). Upstream trains on frame-causal windows instead, where
position k sees frames 0..k -- so 6 of 7 positions can read the arm's velocity
off earlier frames, and in planner demos that predicts the next motion: a way to
lower the loss without the action, which is how the predictor ended up ignoring
actions.

Pair mode keeps upstream's sampling and encoding (one window of T frames,
encoded once) but feeds the predictor T-1 INDEPENDENT samples, each holding one
context frame, the action taken from it, the pose there, and the next frame as
target. Each is exactly deploy's call, and no sample can see another's frames.

Index i = b * (T - 1) + k  <->  window b, transition k -> k+1.
"""
import torch


def to_pairs(h, actions, states, extrinsics, tokens_per_frame):
    """
    h          [B, T*tpf, D]  encoded window (layer-normed if training normalises)
    actions    [B, T-1, A]    action k takes frame k to frame k+1
    states     [B, T, S]      pose at each frame
    extrinsics [B, T, E] or None

    Returns ctx [N, tpf, D], actions [N, 1, A], states [N, 1, S],
    extrinsics [N, 1, E] or None, tgt [N, tpf, D], with N = B * (T - 1).
    """
    B, n_tok, D = h.shape
    T = n_tok // tokens_per_frame
    if T * tokens_per_frame != n_tok or T < 2:
        raise ValueError(f"h has {n_tok} tokens: not a whole number (>= 2) of "
                         f"{tokens_per_frame}-token frames")
    if tuple(actions.shape[:2]) != (B, T - 1):
        raise ValueError(f"actions {tuple(actions.shape)} != (B={B}, T-1={T - 1}, ...)")
    if tuple(states.shape[:2]) != (B, T):
        raise ValueError(f"states {tuple(states.shape)} != (B={B}, T={T}, ...)")

    N = B * (T - 1)
    hz = h.reshape(B, T, tokens_per_frame, D)
    ctx = hz[:, :-1].reshape(N, tokens_per_frame, D)
    tgt = hz[:, 1:].reshape(N, tokens_per_frame, D)
    a = actions.reshape(N, 1, actions.size(-1))
    s = states[:, :-1].reshape(N, 1, states.size(-1))
    e = None
    if extrinsics is not None:
        e = extrinsics[:, :-1].reshape(N, 1, extrinsics.size(-1))
    return ctx, a, s, e, tgt


def goals_to_pairs(goals, batch, n_pairs, tokens_per_frame):
    """Stage 2: dreamer goals [B, >=(T-1)*tpf, D], aligned so goal k is the target
    for transition k -> k+1 (as upstream's loss_fn_dreamer slices them), to
    [N, tpf, D] in the same order as to_pairs."""
    need = n_pairs * tokens_per_frame
    if goals.size(0) != batch or goals.size(1) < need:
        raise ValueError(f"dreamer goals {tuple(goals.shape)}: need batch {batch} and "
                         f">= {need} tokens")
    return goals[:, :need].reshape(batch * n_pairs, tokens_per_frame, goals.size(-1))
