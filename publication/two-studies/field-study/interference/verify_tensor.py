"""Exhaust the labeled markets using PyTorch integers. Optional argument: cpu or cuda."""
import hashlib
import itertools
import json
import sys

import torch

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
if device not in ("cpu", "cuda"):
    raise ValueError("choose cpu or cuda")
torch.set_num_threads(1)
n = 27 * 27 * 216
ids = torch.arange(n, device=device)
applicants = torch.arange(3, device=device)
assignments = torch.arange(8, device=device)
treated = ((assignments[:, None] >> applicants) % 2).bool()
ranks = torch.tensor(list(itertools.permutations(range(3))), device=device).argsort(1)


def digits(code, base):
    return torch.stack([code // (base ** j) % base for j in (2, 1, 0)], dim=1)


baseline = digits(ids // (27 * 216), 3)
alternative = digits(ids // 216 % 27, 3)
school_orders = digits(ids % 216, 6)
choices = torch.where(treated[None, :, :], alternative[:, None, :], baseline[:, None, :])
priority = torch.stack([ranks[school_orders.gather(1, choices[:, :, i]), i]
                        for i in range(3)], dim=2)
admitted = torch.ones_like(choices, dtype=torch.bool)
for i in range(3):
    for j in range(3):
        admitted[:, :, i] &= ((choices[:, :, i] != choices[:, :, j]) |
                              (priority[:, :, i] <= priority[:, :, j]))
counts = admitted.sum(2)
own = torch.zeros((n, 3), dtype=torch.int32, device=device)
spill = torch.zeros_like(own)
for i in range(3):
    for z in range(8):
        if z & (1 << i):
            continue
        other_treated = bin(z).count("1")
        change = admitted[:, z + (1 << i), :].int() - admitted[:, z, :].int()
        own[:, other_treated] += change[:, i]
        spill[:, other_treated] += sum(change[:, j] for j in range(3) if j != i)
sums = torch.stack([counts[:, treated.sum(1) == k].sum(1) for k in range(4)], dim=1)
g0, g1, g2, g3 = sums.T
q0, q1, q2 = (own + spill).T
assert torch.equal(-3*g0 + g1, q0)
assert torch.equal(6*g0 - 4*g1 + 2*g2, q1 - 2*q0)
assert torch.equal(-3*g0 + 3*g1 - 3*g2 + 3*g3, q0 - q1 + q2)
rollout = g3 - g0
assert torch.equal(6*rollout, 2*q0 + q1 + 2*q2)
d = 9*own[:, 0] + 3*own[:, 1] + own[:, 2]
s = 9*spill[:, 0] + 3*spill[:, 1] + spill[:, 2]
masks = (admitted.int() * (1 << applicants)).sum(2)
rows = torch.cat([masks, rollout[:, None], d[:, None], s[:, None]], dim=1)
assert int(rows.min()) >= -128 and int(rows.max()) <= 127
checksum = hashlib.sha256(rows.to(torch.int8).cpu().contiguous().numpy().tobytes()).hexdigest()
print(json.dumps({
    "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
    "torch": str(torch.__version__), "markets": n, "assignments": n * 8,
    "positive_direct_and_rollout_loss": int(((d > 0) & (rollout < 0)).sum()),
    "market_record_sha256": checksum,
}, indent=2))
