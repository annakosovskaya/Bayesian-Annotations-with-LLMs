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

def multinomial(annotations,logits=None,test:bool=False):
    """
    This model corresponds to the plate diagram in Figure 1 of reference [1].
    """
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        zeta = numpyro.sample("zeta", dist.Dirichlet(jnp.ones(num_classes)))

    if logits is None:
        pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))
    # pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)),obs=nn.softmax(logits).mean(0))
    # pi = nn.softmax(logits).mean(0)
    # c = numpyro.sample("c", dist.Categorical(logits = logits[:,np.newaxis,:]), infer={"enumerate": "parallel"})

    with numpyro.plate("item", num_items, dim=-2):
        if logits is None:
            c = numpyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})
        else:
            c = numpyro.sample("c", dist.Categorical(logits = logits[:,np.newaxis,:]), infer={"enumerate": "parallel"})
        with numpyro.plate("position", num_positions):
            if test:
                numpyro.sample("y", dist.Categorical(zeta[c]))
            else:
                numpyro.sample("y", dist.Categorical(zeta[c]), obs=annotations)

def item_difficulty(annotations,logits):
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        eta = numpyro.sample(
            "eta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        chi = numpyro.sample(
            "Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    # pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))
    # pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)),obs = nn.softmax(logits).mean(0))
    # pi = nn.softmax(logits).mean(0)
    c = numpyro.sample("c", dist.Categorical(logits = logits[:,np.newaxis,:]), infer={"enumerate": "parallel"})

    with numpyro.plate("item", num_items, dim=-2):
        # c = numpyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})

        with handlers.reparam(config={"theta": LocScaleReparam(0)}):
            theta = numpyro.sample("theta", dist.Normal(eta[c], chi[c]).to_event(1))
            theta = jnp.pad(theta, [(0, 0)] * (jnp.ndim(theta) - 1) + [(0, 1)])

        with numpyro.plate("position", annotations.shape[-1]):
            numpyro.sample("y", dist.Categorical(logits=theta), obs=annotations)


if __name__ == "__main__":
    res = read_jsonl("ghc_train.jsonl")
    annotators = np.array([np.array(it["annotators"]) for it in res if len(it["annotators"]) == 3])
    annotations = np.array([np.array(it["labels"]) for it in res if len(it["annotators"]) == 3])
    logits = np.load("llm_data/Qwen2.5-32B/train/logits.npy")
    logits = np.array([x for i, x in enumerate(logits[:, :2]) if len(res[i]["annotators"]) == 3])