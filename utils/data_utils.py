import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

import numpy as np
from sympy.logic.boolalg import Boolean
from tqdm import tqdm

from jax import nn, random, vmap, clear_caches
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam
from numpyro.ops.indexing import Vindex

import json

from sklearn.metrics import log_loss,f1_score

def read_jsonl(file:str) -> List[Dict]:
    with open(file, "r") as f:
        x = f.read()
        x = x.split("\n")
        res = []
        for x_val in x:
            try:
                res.append(json.loads(x_val))
            except:
                print(x_val)
    return res