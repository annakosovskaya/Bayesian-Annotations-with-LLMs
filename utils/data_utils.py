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


def create_annotator_mapping(data):
    from collections import defaultdict
    annotator_positions = defaultdict(set)
    for item in data:
        for pos, ann in enumerate(item['annotators']):
            annotator_positions[ann].add(pos)
    annotator_to_positions = {}
    current_position = 0
    for annotator in sorted(annotator_positions.keys()):
        positions = sorted(annotator_positions[annotator])
        for pos in positions:
            annotator_to_positions[(annotator, pos)] = current_position
            current_position += 1
    return annotator_to_positions


def process_annotations(data, annotator_mapping=None):
    if annotator_mapping is None:
        annotator_mapping = create_annotator_mapping(data)

    total_positions = max(annotator_mapping.values()) + 1
    positions = np.zeros(total_positions, dtype=int)
    annotations = np.zeros((len(data), total_positions), dtype=int)
    masks = np.zeros((len(data), total_positions), dtype=bool)

    for item_idx, item in enumerate(data):
        for pos, (annotator, label) in enumerate(zip(item['annotators'], item['labels'])):
            if (annotator, pos) in annotator_mapping:
                matrix_pos = annotator_mapping[(annotator, pos)]
                annotations[item_idx, matrix_pos] = label
                masks[item_idx, matrix_pos] = True
                positions[matrix_pos] = annotator
    return positions, annotations, masks