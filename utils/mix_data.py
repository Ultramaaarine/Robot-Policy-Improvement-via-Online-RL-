import torch


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _fix_rgb_gripper(v: torch.Tensor, key_path: str):
    """
    Unify rgb_gripper to [B, T, 3, 84, 84].
    """
    if "rgb_gripper" not in key_path:
        return v

    if not isinstance(v, torch.Tensor):
        return v

    if v.dim() != 5:
        raise ValueError(f"Unexpected rgb_gripper shape at {key_path}: {v.shape}")

    # [B,T,H,W,C] -> [B,T,C,H,W]
    if v.shape[-1] == 3:
        v = v.permute(0, 1, 4, 2, 3).contiguous()

    # already [B,T,C,H,W]
    elif v.shape[2] == 3:
        v = v.contiguous()

    else:
        raise ValueError(f"Cannot infer channel dim at {key_path}: {v.shape}")

    return v


def _mix_value(v_online, v_offline, b_on, b_off, key_path=""):
    if v_online is None or v_offline is None:
        return None

    if isinstance(v_online, torch.Tensor) and isinstance(v_offline, torch.Tensor):
        v_online = _fix_rgb_gripper(v_online, key_path + ".online")
        v_offline = _fix_rgb_gripper(v_offline, key_path + ".offline")

        assert v_online.shape[1:] == v_offline.shape[1:], (
            f"Shape mismatch at {key_path}: {v_online.shape} vs {v_offline.shape}"
        )

        return torch.cat([v_online[:b_on], v_offline[:b_off]], dim=0)

    if isinstance(v_online, dict) and isinstance(v_offline, dict):
        shared = set(v_online.keys()) & set(v_offline.keys())
        out = {}

        for k in shared:
            mixed = _mix_value(
                v_online[k],
                v_offline[k],
                b_on,
                b_off,
                key_path=f"{key_path}.{k}" if key_path else k,
            )
            if mixed is not None:
                out[k] = mixed

        return out if len(out) > 0 else None

    print(
        f"[mix_data] skip {key_path}: "
        f"online={type(v_online)}, offline={type(v_offline)}"
    )
    return None


def mix_data(online_batch, offline_batch, ratio=0.5):
    assert isinstance(online_batch, dict)
    assert isinstance(offline_batch, dict)

    shared_keys = set(online_batch.keys()) & set(offline_batch.keys())

    first_online = _first_tensor(online_batch)
    first_offline = _first_tensor(offline_batch)

    if first_online is None or first_offline is None:
        raise ValueError("No tensor found in online/offline batch.")

    B_online = first_online.shape[0]
    B_offline = first_offline.shape[0]

    B = min(B_online, B_offline)
    B_online_take = int(B * ratio)
    B_offline_take = B - B_online_take

    mixed_batch = {}

    for k in shared_keys:
        mixed = _mix_value(
            online_batch[k],
            offline_batch[k],
            B_online_take,
            B_offline_take,
            key_path=k,
        )
        if mixed is not None:
            mixed_batch[k] = mixed

    print(f"[mix_data] final keys: {mixed_batch.keys()}")
    return mixed_batch