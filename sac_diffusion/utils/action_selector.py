```python
import torch
import torch.nn.functional as F
from typing import Optional


def select_action(
    critic_network,
    action_seq,          # [K, T, D] candidate action sequences
    n_cond,              # [1, To, Do] or [K, To, Do]
    top_k: int = 3,
    temperature: float = 0.5,
    aggregate: str = "last4_mean",
    visual_obs_seq: Optional[torch.Tensor] = None,
):
    """
    Select one action sequence from K diffusion-policy candidates
    using Double-Q critic scores.

    Returns:
        selected_action: [T, D]
        selected_idx: int
        scores: [K]
        probs: [top_k]
    """
    K = action_seq.shape[0]

    # Repeat the current observation for all K action candidates.
    if n_cond.shape[0] == 1:
        n_cond = n_cond.repeat(K, 1, 1)
    elif n_cond.shape[0] != K:
        raise ValueError(
            f"obs_seq batch must be 1 or K={K}, got {n_cond.shape[0]}"
        )

    with torch.no_grad():
        # Evaluate all candidate action sequences with the Double-Q critic.
        q1, q2 = critic_network(
            state=n_cond,
            visual_obs=visual_obs_seq,
            action=action_seq,
        )

        # Use clipped Double-Q to reduce value overestimation.
        q = torch.min(q1, q2)

        # Aggregate temporal Q-values into one score per candidate.
        if q.dim() == 3:
            if aggregate == "mean":
                scores = q.mean(dim=1).squeeze(-1)

            elif aggregate == "last4_mean":
                last_n = min(4, q.shape[1])
                scores = q[:, -last_n:, :].mean(dim=1).squeeze(-1)

            elif aggregate == "last":
                scores = q[:, -1, :].squeeze(-1)

            else:
                raise ValueError(
                    f"Unknown aggregate mode: {aggregate}"
                )

        elif q.dim() == 2:
            scores = q.squeeze(-1)

        else:
            raise ValueError(
                f"Unexpected q shape: {q.shape}"
            )

        # Keep the top-k candidates according to critic scores.
        k_sel = min(top_k, K)
        top_vals, top_idx = torch.topk(
            scores,
            k=k_sel,
            dim=0,
        )

        # Sample from top-k candidates using temperature-scaled softmax.
        temperature = max(float(temperature), 1e-6)
        probs = F.softmax(
            top_vals / temperature,
            dim=0,
        )

        local_idx = torch.multinomial(
            probs,
            num_samples=1,
        ).item()

        # Map the sampled top-k index back to the original candidate index.
        selected_idx = top_idx[local_idx].item()
        selected_action = action_seq[selected_idx]

    return selected_action, selected_idx, scores, probs
```
