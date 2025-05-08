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

    model = multinomial

    mcmc = MCMC(
        NUTS(model),
        num_warmup=100,
        num_samples=500,
        num_chains=1,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )

    train_size = round(annotations.shape[0] * 0.9)

    train_data = (
        (annotations[:train_size], logits[:train_size])
        if model in [multinomial, item_difficulty]
        else (annotators[:train_size], annotations[:train_size], logits[:train_size])
    )

    test_data = (
        (annotations[train_size:], logits[train_size:], True)
        if model in [multinomial, item_difficulty]
        else (annotators[train_size:], annotations[train_size:], logits[train_size:], True)
    )

    mcmc.run(random.PRNGKey(0), *train_data)
    mcmc.print_summary()

    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples, infer_discrete=True)
    discrete_samples = predictive(random.PRNGKey(1), *test_data)

    annotator_probs = vmap(lambda x: x.mean(0), in_axes=1)(
        discrete_samples["y"]
    )

    print(f'log loss over positions = {log_loss(annotations[train_size:].flatten(), annotator_probs.flatten())}')
    print(f'log loss over items = {log_loss(np.rint(annotations[train_size:].mean(1)), annotator_probs.mean(1))}')
    print('F1 score with position predictions')
    print(
        f'Binary F1 score = {f1_score(annotations[train_size:].flatten(), np.rint(annotator_probs).flatten(), pos_label=1)}')
    print(
        f'Micro F1 score = {f1_score(annotations[train_size:].flatten(), np.rint(annotator_probs).flatten(), pos_label=1, average="micro")}')
    print(
        f'Macro F1 score = {f1_score(annotations[train_size:].flatten(), np.rint(annotator_probs).flatten(), pos_label=1, average="macro")}')
    print(
        f'Weighted F1 score = {f1_score(annotations[train_size:].flatten(), np.rint(annotator_probs).flatten(), pos_label=1, average="weighted")}')
    print('F1 score with predictions of number of 1s out of all positions')
    print(
        f'Micro F1 score = {f1_score(annotations[train_size:].sum(1), np.rint(annotator_probs).sum(1), average="micro")}')
    print(
        f'Macro F1 score = {f1_score(annotations[train_size:].sum(1), np.rint(annotator_probs).sum(1), average="macro")}')
    print(
        f'Weighted F1 score = {f1_score(annotations[train_size:].sum(1), np.rint(annotator_probs).sum(1), average="weighted")}')
    print('F1 score with majority vote')
    print(
        f'Binary F1 score = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1)}')
    print(
        f'Micro F1 score = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1, average="micro")}')
    print(
        f'Macro F1 score = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1, average="macro")}')
    print(
        f'Weighted F1 score = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1, average="weighted")}')