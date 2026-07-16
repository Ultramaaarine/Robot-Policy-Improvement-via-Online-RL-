import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Literal, Dict, List, Tuple

import numpy as np
from pathlib import Path


class TargetSelector():

    def __init__(self):
        self.ann_dir = Path(__file__).parent.parent.parent.joinpath("annoatations_time")

    def load_goal(self,path:Path):

        data = json.loads(path.read_text(encoding="utf-8"))
        ann = data.get("annoations",[])

