"""CPU checks for pair mode (src/utils/pairs.py) and oracle goals (deploy). Run from the repo root:  python server/check_pairs.py"""
import sys, types
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.utils.pairs import to_pairs, goals_to_pairs
from src.models.ac_predictor import vit_ac_predictor

torch.manual_seed(0)
B, T, tpf, D = 3, 8, 4, 16          # 2x2 patch grid -> 4 tokens per frame
fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

# ---- 1. ordering: pair i = b*(T-1)+k is (frame k -> frame k+1) of window b
h = torch.randn(B, T * tpf, D)
actions = torch.randn(B, T - 1, 7)
states = torch.randn(B, T, 7)
ctx, a, s, e, tgt = to_pairs(h, actions, states, None, tpf)
hz = h.view(B, T, tpf, D)
ok = ctx.shape == (B * (T - 1), tpf, D) and a.shape == (B * (T - 1), 1, 7) and e is None
for b in range(B):
    for k in range(T - 1):
        i = b * (T - 1) + k
        ok &= torch.equal(ctx[i], hz[b, k]) and torch.equal(tgt[i], hz[b, k + 1])
        ok &= torch.equal(a[i, 0], actions[b, k]) and torch.equal(s[i, 0], states[b, k])
check("to_pairs: shapes and (window, step) ordering", bool(ok))

goals = torch.randn(B, (T - 1) * tpf, D)
gp = goals_to_pairs(goals, B, T - 1, tpf)
gz = goals.view(B, T - 1, tpf, D)
check("goals_to_pairs: goal k of window b -> pair b*(T-1)+k",
      all(torch.equal(gp[b * (T - 1) + k], gz[b, k]) for b in range(B) for k in range(T - 1)))

for bad, args in (("actions length", (h, actions[:, :-1], states, None, tpf)),
                  ("states length", (h, actions, states[:, :-1], None, tpf)),
                  ("token count", (h[:, :-1], actions, states, None, tpf))):
    try:
        to_pairs(*args); check(f"to_pairs rejects bad {bad}", False)
    except ValueError:
        check(f"to_pairs rejects bad {bad}", True)

# ---- 2. the real predictor architecture (small): no history leaks into a pair
pred = vit_ac_predictor(img_size=(32, 32), patch_size=16, num_frames=T, tubelet_size=1,
                        embed_dim=D, predictor_embed_dim=128, depth=2, num_heads=2,
                        is_frame_causal=True, use_rope=True, action_embed_dim=7).eval()
with torch.no_grad():
    out = pred(ctx, a, s, None)                                   # all pairs, batched
    ok = out.shape == (B * (T - 1), tpf, D)
    for b in range(B):
        for k in range(T - 1):
            single = pred(hz[b:b + 1, k], actions[b:b + 1, k:k + 1], states[b:b + 1, k:k + 1], None)
            ok &= torch.allclose(out[b * (T - 1) + k], single[0], atol=1e-5)
    check("pair output == deploy-style single-frame call (T=1)", bool(ok))

    h2 = h.clone().view(B, T, tpf, D)
    h2[:, 0] += 5.0                                                # change frame 0 only
    ctx2, a2, s2, _, _ = to_pairs(h2.view(B, T * tpf, D), actions, states, None, tpf)
    out2 = pred(ctx2, a2, s2, None)
    others = [b * (T - 1) + k for b in range(B) for k in range(1, T - 1)]
    check("changing frame 0 leaves pairs k>=1 unchanged (no history)",
          torch.allclose(out[others], out2[others], atol=1e-6))

    causal = pred(h[:, :-tpf], actions, states[:, :-1], None)     # upstream frame-causal
    check("pair k=0 == causal position 0 (same regime as before)",
          torch.allclose(out[0::T - 1], causal[:, :tpf], atol=1e-5))
    check("pair k>=1 != causal position k (history really removed)",
          not torch.allclose(out[1], causal[0, tpf:2 * tpf], atol=1e-4))

    # gradient reaches the action pathway through pair mode
pred.train()
z = pred(ctx, a, s, None)
F.l1_loss(z, tgt).backward()
check("loss backprops into action_encoder",
      pred.action_encoder.weight.grad is not None and pred.action_encoder.weight.grad.abs().sum() > 0)

# ---- 3. oracle goal mode in WorldModel (encoder / CEM stubbed)
# cem_utils imports h5py/wandb/tqdm at module level for helpers this check never calls
for _m in ('h5py', 'wandb', 'tqdm'):
    if _m not in sys.modules:
        try:
            __import__(_m)
        except ImportError:
            sys.modules[_m] = types.ModuleType(_m)
sys.modules['tqdm'].tqdm = getattr(sys.modules['tqdm'], 'tqdm', lambda x, **k: x)
import app.vjepa_2_1_dreamer_ac.cem_utils as cu
captured = {}
def fake_cem(context_frame, context_pose, goal_frame, world_model, **kw):
    captured["goal"] = goal_frame
    return torch.zeros(1, 7), 0.0
cu.cem = fake_cem
cu.quaternion_to_euler = lambda p: np.zeros((1, 7), np.float32)
dummy = torch.nn.Linear(1, 1)
wm = cu.WorldModel(encoder=dummy, predictor=dummy, dreamer_predictor=None, tokens_per_frame=tpf,
                   transform=None, device="cpu", dtype=torch.float32, goal_mode="oracle")
codes = {"cur": torch.full((1, tpf, D), 1.0), "ref_t": torch.full((1, tpf, D), 2.0),
         "ref_t1": torch.full((1, tpf, D), 3.0)}
wm.encode = lambda img: codes[img]
wm("cur", np.zeros((1, 8), np.float32), "ref_t", "ref_t1")
check("oracle: goal is encode(target_ref), dreamer never called",
      torch.equal(captured["goal"], codes["ref_t1"]))
try:
    cu.WorldModel(encoder=dummy, predictor=dummy, dreamer_predictor=None, tokens_per_frame=tpf,
                  transform=None, device="cpu", goal_mode="dreamer")
    check("dreamer mode without a dreamer is rejected", False)
except ValueError:
    check("dreamer mode without a dreamer is rejected", True)

class FakeDreamer(torch.nn.Module):
    def forward(self, xt, yt, yt_plus_1):
        return xt + yt + yt_plus_1
wm2 = cu.WorldModel(encoder=dummy, predictor=dummy, dreamer_predictor=FakeDreamer(),
                    tokens_per_frame=tpf, transform=None, device="cpu", dtype=torch.float32)
wm2.encode = lambda img: codes[img]
wm2("cur", np.zeros((1, 8), np.float32), "ref_t", "ref_t1")
check("dreamer mode unchanged: goal = dreamer(cur, ref_t, ref_t1)",
      torch.equal(captured["goal"], codes["cur"] + codes["ref_t"] + codes["ref_t1"]))

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
