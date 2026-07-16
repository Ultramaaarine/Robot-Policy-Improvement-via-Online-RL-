import torch
import torch.nn.functional as F
from typing import Optional

def select_action(
    critic_network,
    action_seq,          # [K, T, D] from diffusion model
    n_cond,             # [1, To, Do] or [K, To, Do] assembled obs dim dependes on cfg    

    top_k: int = 3,
    temperature: float = 0.5,
    aggregate: str = "last4_mean",
    visual_obs_seq:Optional[torch.Tensor]= None,
):
    """
    Use critic to select one action sequence from K candidates.

    Returns:
        selected_action: [T, D]
        selected_idx: int
        scores: [K]
        probs: [top_k]
    """
    K = action_seq.shape[0]

    # Expand obs to match K candidates if needed 
    if n_cond.shape[0] == 1:
        n_cond = n_cond.repeat(K, 1, 1)
    elif n_cond.shape[0] != K:
        raise ValueError(
            f"obs_seq batch must be 1 or K={K}, got {n_cond.shape[0]}"
        )

    with torch.no_grad():
        # Adjust argument order here if your critic expects (action, obs)
        q1, q2 = critic_network(state = n_cond, visual_obs = visual_obs_seq, action = action_seq)

        # q could be [K,1] or [K,T,1]
        q = torch.min(q1, q2)

        if q.dim() == 3:
            if aggregate == "mean":
                scores = q.mean(dim=1).squeeze(-1)          # [K]
            elif aggregate == "last4_mean":
                last_n = min(4, q.shape[1])
                scores = q[:, -last_n:, :].mean(dim=1).squeeze(-1)  # [K]
            elif aggregate == "last":
                scores = q[:, -1, :].squeeze(-1)            # [K]
            else:
                raise ValueError(f"Unknown aggregate mode: {aggregate}")
        elif q.dim() == 2:
            scores = q.squeeze(-1)                          # [K]
        else:
            raise ValueError(f"Unexpected q shape: {q.shape}")

        # top-k soft selection
        k_sel = min(top_k, K)
        top_vals, top_idx = torch.topk(scores, k=k_sel, dim=0) # select among top k
        temperature = max(float(temperature), 1e-6)
        probs = F.softmax(top_vals / temperature, dim=0)
        local_idx = torch.multinomial(probs, num_samples=1).item() # sample a index according to probs
        selected_idx = top_idx[local_idx].item()

        selected_action = action_seq[selected_idx]          # [T, D]

    return selected_action, selected_idx, scores, probs