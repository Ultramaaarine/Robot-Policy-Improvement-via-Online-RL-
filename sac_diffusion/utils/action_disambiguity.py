import torch


class DiffusionActionSelector:
    def __init__(
        self,
        policy,
        num_samples: int = 8,
        use_goal_score: bool = True,
    ):
        """
        policy: HybridPolicy (用于调用 sampling_action_from_cond)
        """
        self.policy = policy
        self.num_samples = num_samples
        self.use_goal_score = use_goal_score

    @torch.no_grad()
    def sample_multiple(
        self,
        cond,
        action_mode,
        cond_norm_params,
        rollout_step,
        batch_size=None,
    ):
        """
        return: [N, B, T, Da]
        """
        all_actions = []

        for _ in range(self.num_samples):
            action_seq = self.policy.sampling_action_from_cond(
                cond,
                action_mode,
                cond_norm_params,
                rollout_step=rollout_step,
                batch_size=batch_size,
            )  # [B, T, Da]

            all_actions.append(action_seq)

        return torch.stack(all_actions, dim=0)

    def score(self, action_seqs, cond):
        """
        action_seqs: [N, B, T, Da]
        return: scores [N, B]
        """
        # 只用第一步
        a0 = action_seqs[:, :, 0, :]  # [N, B, Da]

        cond_dict = cond["cond"] if "cond" in cond else cond

        if "pos" not in cond_dict:
            raise ValueError("cond must contain pos for scoring")

        # 当前末端位置
        pos = cond_dict["pos"][:, -1, :]  # [B, 3]
        pos = pos.unsqueeze(0).expand(a0.shape[0], -1, -1)  # [N, B, 3]

        # 预测下一步位置（delta）
        pred_pos = pos + a0

        # --- scoring ---
        if self.use_goal_score and "goal_error" in cond_dict:
            goal_error = cond_dict["goal_error"][:, 0, :]  # [B, 3]
            goal = pos + goal_error.unsqueeze(0)

            score = torch.norm(pred_pos - goal, dim=-1)  # [N, B]

        else:
            # fallback
            score = torch.norm(a0, dim=-1)

        return score

    def select(self, action_seqs, cond):
        """
        return: best_action [B, Da]
        """
        scores = self.score(action_seqs, cond)  # [N, B]

        best_idx = torch.argmin(scores, dim=0)  # [B]
        B = scores.shape[1]

        best_actions = []
        for b in range(B):
            best_actions.append(action_seqs[best_idx[b], b, 0])

        return torch.stack(best_actions, dim=0)

    @torch.no_grad()
    def predict(
        self,
        cond,
        action_mode,
        cond_norm_params,
        rollout_step,
        batch_size=None,
    ):
        """
        一步完成：sample + select
        """
        action_seqs = self.sample_multiple(
            cond,
            action_mode,
            cond_norm_params,
            rollout_step,
            batch_size=batch_size,
        )

        best_action = self.select(action_seqs, cond)

        return best_action