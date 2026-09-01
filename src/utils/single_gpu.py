"""Skip DistributedDataParallel when there is only one process.

At world_size == 1 every DDP operation is a no-op with itself:

  all-reduce gradients across ranks   average of one value is that value
  divide by world_size                divide by 1
  broadcast params at construction    copy to self
  broadcast buffers each forward      copy to self (and there is no BatchNorm
                                      here -- LayerNorm/GroupNorm are per-sample
                                      and keep no running state)
  find_unused_parameters              only matters for the all-reduce
  static_graph                        replay optimisation, not numerical

But DDP still allocates a contiguous gradient bucket per module, sized
~4 bytes x (parameters with requires_grad=True), AT WRAP TIME. In stage 2 that
is ~11 GB, and three of the four wrapped modules are never trained at all:

  encoder          4.05 GB   not frozen, but absent from init_opt's param groups
  target_encoder   4.05 GB   frozen at train.py:362 -- AFTER the wrap
  dreamer_predictor 1.70 GB  frozen at train.py:375 -- AFTER the wrap
  AC predictor     1.22 GB   the only module actually optimised

Freezing before the wrap does not help: PyTorch refuses to wrap a module with no
trainable parameters at all ("DistributedDataParallel is not needed when a module
doesn't have any parameter that requires a gradient"). bucket_cap_mb changes
granularity, not total. There is no "wrap but allocate nothing" mode -- DDP's
answer to that request is "then do not wrap me."

The ONLY DDP behaviour this project depends on at one GPU is the naming: DDP
stores the model as an attribute called `module`, so state_dict() keys come out
`module.`-prefixed. repack_stage0.py deliberately adds that prefix, and
load_checkpoint / make_deploy_ckpt / deploy all expect it -- without it a
state_dict "loads successfully" while binding nothing (HANDOFF §6). NoDDP
reproduces exactly that shape, so checkpoints stay compatible in both directions.

Gradient accumulation is unaffected: it averages over MICRO-BATCHES via the
`/ n_micro` loss scaling plus autograd's default accumulation into .grad. DDP
averages over RANKS. Orthogonal axes.
"""

import os

import torch
from torch.nn.parallel import DistributedDataParallel


class NoDDP(torch.nn.Module):
    """Passthrough with DDP's attribute shape. No reducer, no buckets, no hooks."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def wrap_ddp(module, world_size, **ddp_kwargs):
    """Real DDP when it does something; NoDDP when it cannot.

    world_size > 1 MUST still return DistributedDataParallel -- otherwise the
    ranks silently stop synchronising and every GPU trains a different model.
    """
    # DJEPA_FORCE_DDP=1 restores real DDP at one rank. Only used by
    # server/verify_noddp.py, to prove the two paths give identical losses.
    if world_size > 1 or os.environ.get("DJEPA_FORCE_DDP") == "1":
        return DistributedDataParallel(module, **ddp_kwargs)
    return NoDDP(module)
