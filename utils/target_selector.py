# return a target time in dict form {"target_num": t} for better distinguish
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Literal, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

TakeMode = Literal["all", "first", "last"]
AggMode = Literal["median", "mean"]


@dataclass
class ProgressSummary:
    label:str
    count: int
    mean: float
    std: float
    min: float
    max: float
    p25: float
    median: float
    p75: float


class TargetSelector:
    """
    从 annotations_time/ep_*.json 里读取 annotations:
      [{"t": int, "progress": float, "label": int}, ...]

    - scan() 后可统计 progress 分布
    - predict_t(): 用统计得到的典型 progress 反推 t（支持噪声 + 起始点限制）
    """

    def __init__(
        self,
        skill: str,
        sort_by: str,
        label: Optional[List[int]] = None,
        take: TakeMode = "all",
        use_saved_progress: bool = True,
        round_decimals: int = 6,
    ):
        self.skill = skill  # 未来可以用 skill 来决定子目录
        self.ann_dir = Path(__file__).parent.parent.parent / "annotations_time"
        self.label = label
        self.take = take
        self.use_saved_progress = use_saved_progress
        self.round_decimals = round_decimals
        self.sort_by = sort_by
        self._per_episode: Dict[str, List[Tuple[int,int,float]]] = {}
        self._all_lp: Optional[np.ndarray] = None
        self._labels_seen:set[int] = set()
    def scan(self, pattern: str = "ep_*.json") -> "TargetSelector":
        self._labels_seen.clear()
        files = sorted(self.ann_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No {pattern} found under: {self.ann_dir.resolve()}")

        per_ep: Dict[str, List[Tuple[int,int,float]]] = {}
        all_labeled_progress: List[Tuple[int,float]] = []

        for fp in files:
            items = self._load_goal(fp)  # list[tuple[t,label,progress] list of progress in one file
            #print(f"scan has goal: {items}") #正确
            if not items:
                continue

            if self.take == "first": # 第一个target的时刻
                items = [items[0]]
            elif self.take == "last":
                items = [items[-1]] #最后一个 target的时刻
            elif self.take != "all":
                raise ValueError(f"Unknown take mode: {self.take}")

            per_ep[fp.stem] = items
            
            all_labeled_progress.extend([(label,p) for (_,label,p) in items]) # take the label-progress tuple  
 
        if not all_labeled_progress:
            raise ValueError(
                "No progress found. Check if your label filter removed all annotations, "
                "or json files have empty annotations."
            )

        self._per_episode = per_ep
        self._all_lp = np.asarray(all_labeled_progress, dtype=np.float64)
        #print(f"self._all_labled progress: {self._all_lp}")
        #print(f"self._per_episode is {self._per_episode}") 正确
        return self # update target selector add "self._all_lp and self._per_episode"

    def _load_goal(self, path: Path) -> List[Tuple[int, int, float]]: # list with one tuple
        """
       读取单个 json，提取 (t, label, progress)
       - 返回按 t 升序排列的三元组列表
       """
        data = json.loads(path.read_text(encoding="utf-8"))
        ann = data.get("annotations", [])
        if not isinstance(ann, list) or len(ann) == 0:
            return []

        out: List[Tuple[int, int, float]] = []

        for a in ann:# element in ann dict
            if not isinstance(a, dict):
                continue

        # label
            if "label" not in a or "t" not in a or "progress" not in a:
                continue
            try:
                label = int(a["label"])
                t = int(a["t"])
                p = float(a["progress"])
                
            except Exception:
                continue
            self._labels_seen.add(label)

            if self.label is not None and label not in self.label:
                continue

            out.append((t, label, p))
        if self.sort_by == "t":
            out.sort(key=lambda x: x[0])  # sort by t
        elif self.sort_by == "label": 
            out.sort(key=lambda x: x[1])
        else:
            raise ValueError(f"Unknown sort_by:{self.sort_by}")
        return out
    @property # convert all_labeled_progress(list[tuple]) to ndarray[[label,progress],[label,progress]...]
    def all_labeled_progress(self)->np.ndarray:
        if self._all_lp is None:
            raise RuntimeError("Call scan() first.")
        return self._all_lp

    @property
    def per_episode(self) -> Dict[str, List[Tuple[int,int,float]]]:
        if self._all_lp is None:
            raise RuntimeError("Call scan() first.")
        return self._per_episode


    def summary(self, labels: Optional[list[int]] = None) -> Dict[int, ProgressSummary]:
        lp = self.all_labeled_progress  # np.ndarray shape [N,2] = [label, progress]
        #print(f"self.all_labeled_progress:{self.all_labeled_progress},shape: {self.all_labeled_progress.shape}") # shape: 66,2
        if lp.ndim != 2 or lp.shape[1] != 2:
            raise RuntimeError("all_label_progress must be shape [N,2]")

    # 如果不传 labels，就用 scan 时收集到的 labels_seen
        if labels is None:
            labels = sorted(self._labels_seen)  # set -> sorted list

        summaries: Dict[int, ProgressSummary] = {}

        for lab in labels:
            mask = (lp[:, 0].astype(int) == int(lab))
            arr = lp[mask, 1].astype(np.float64)

            if arr.size == 0:
            # 你也可以选择 continue 而不是报错
                raise ValueError(f"No progress found for label = {lab}")

            summaries[int(lab)] = ProgressSummary(
                label=str(lab),
                count=int(arr.size),
                mean=float(arr.mean()),
                std=float(arr.std()),
                min=float(arr.min()),
                p25=float(np.quantile(arr, 0.25)),
                median=float(np.quantile(arr, 0.50)),
                p75=float(np.quantile(arr, 0.75)),
                max=float(arr.max()),
            )

        return summaries

    def topk(self, label,k: int = 10) -> List[Tuple[float, int]]:
        lp= self.all_labeled_progress
        if lp.ndim !=2 or lp.shape[1] !=2:
            raise RuntimeError("all_label_progress must be shape [N,2]")
        if label is None:
            arr = lp[:,1].astype(np.float64) # take all progresses
        else: 
            mask = (lp[:,0].astype(int)==int(label)) # mask: [true,false,flase...] bool list compare lp[:,0] and label
            arr = lp[mask,1] # pick up all true 
        rounded = np.round(arr, self.round_decimals)
        cnt = Counter(rounded.tolist())
        return cnt.most_common(k)

    def typical_progress(
        self,
        label: Optional[int] = None,
        method: AggMode = "mean"
    ):
           # Dict[label -> ProgressSummary]

    # ===== case 1: 所有 label =====
         if label is None:
            s = self.summary()
            labels = sorted(self._labels_seen)

            result = []
            for lab in labels:
               if method == "median":
                   m = float(s[lab].median)
               elif method == "mean":
                   m = float(s[lab].mean)

               std = float(s[lab].std)
               result.append((m, std))

            return result   # [(mean,std), (mean,std), ...]

    # ===== case 2: 单个 label =====
         lab = int(label)
         s = self.summary(labels=[lab])
         if method == "median":
            m = float(s[lab].median)
         elif method == "mean":
            m = float(s[lab].mean)

         std = float(s[lab].std)

         return (m, std)

    def predict_t(
       self,
       T: int,
       label: Optional[int] = None,
       method: AggMode = "median",
       sigma: float = 3.0,
       min_t: int = 0,
       max_t: Optional[int] = None,
    ) -> Dict[int, int]:
      if T <= 0:
          raise ValueError("T must be positive.")
      denom = max(T - 1, 1)
      max_t_eff = (T - 1) if max_t is None else int(max_t)

    # p: Dict[int, float]  (label -> typical progress)
      p = self.typical_progress(label=label, method=method)
      if isinstance(p,tuple):
          mean,std = p
          tt = int(np.round(mean*denom))

          if sigma and sigma > 0:
              tt += int(np.round(np.random.normal(loc = 0.0, scale = sigma)))
          tt = int(np.clip(tt,min_t,max_t_eff))
          return tt
      elif isinstance(p,list):
          labels = sorted(self._labels_seen)
          out: Dict[int, int] = {}

          for lab, (mean, std) in zip(labels, p):
              tt = int(np.round(mean * denom))

              if sigma and sigma > 0:
                  tt += int(np.round(np.random.normal(loc=0.0, scale=sigma)))

              tt = int(np.clip(tt, min_t, max_t_eff))
              out[int(lab)] = tt

          return out

      else:
          raise TypeError(f"Unexpected typical_progress return type: {type(p)}")
    # def target_dict(
    #     self,
    #     T: int,
    #     label:int,
       
    #     method: AggMode = "median",
    #     sigma: float = 3.0,
    #     min_t: int = 0,
    #     max_t: Optional[int] = None,
    # ) -> Dict[str, int]:
    #     """
    #     按你说的形式返回：{"target_num": t}
    #     这里 target_num 用于区分第几个子目标（1/2/...）
    #     """
    #     (label,t) = self.predict_t(T=T, label=label,method=method, sigma=sigma, min_t=min_t, max_t=max_t)
    #     return {str(label): int(t)}

    def report_text(self, label:Optional[int] = None, topk: int = 10) -> str:
        lines: List[str] = []
        lines.append("[Progress Stats]")
        lines.append(f"ann_dir: {self.ann_dir.resolve()}")
        lines.append(f"episodes with marks: {len(self._per_episode)}")
        lines.append(f"take={self.take} label={self.label} use_saved_progress={self.use_saved_progress}")
        lines.append("")
        if label is None:
            sdict = self.summary()
        else: 
            sdict = self.summary(label)
        for lab, s in sdict.items(): # get stats separatly
          lines.append(f"label: {s.label}")
          lines.append(f"count : {s.count}")
          lines.append(f"mean  : {s.mean:.6f}")
          lines.append(f"std   : {s.std:.6f}")
          lines.append(f"min   : {s.min:.6f}")
          lines.append(f"p25   : {s.p25:.6f}")
          lines.append(f"median: {s.median:.6f}")
          lines.append(f"p75   : {s.p75:.6f}")
          lines.append(f"max   : {s.max:.6f}")
          lines.append("")

        lines.append(f"[Top {topk}] (rounded={self.round_decimals} decimals)")
        for v, c in self.topk(label,topk):
            lines.append(f"{v:.{self.round_decimals}f}  x{c}")
        return "\n".join(lines)
    def report_dict(self,label:Optional[int] = None, topk: int = 10):

        if label is None:
            sdict = self.summary()
        else: 
            sdict = self.summary(label)
        return sdict


class GMMTargetSelector(nn.Module):
    def __init__(self,goal_num:int,init_sigma:float,init_mu:float):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))
        self.log_mu = nn.Parameter(torch.log(torch.tensor(init_mu)))
        self.logits_pi = nn.Parameter(torch.zeros(goal_num)) # [G] 每个目标一个 pi logit (pi weight in gmm)
        self.goal_num = goal_num

    def init_logp(self,x):
        x = x.float().flatten()
        log_pi = F.log_softmax(self.logits_pi,dim=-1) # [N]
        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma) + 1e-8
        log_gaussian = -0.5 * ((x[:, None] - self.mu[None, :]) / sigma[None, :]).pow(2) - log_sigma[None, :] - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        log_gmm = torch.logsumexp(log_gaussian + log_pi[None, :], dim=-1)
        return log_gmm
    
    def nll(self,x):
        return -self.init_logp(x).mean()