
import numpy as np
import os

import numpy as np

from jax import random, vmap

from numpyro.infer import MCMC, NUTS, Predictive


from sklearn.metrics import log_loss,f1_score


from utils.data_utils import read_jsonl
from models.models_numpyro import multinomial, item_difficulty, dawid_skene, hierarchical_dawid_skene, mace, logistic_random_effects

if __name__ == "__main__":
    res = read_jsonl("data/ghc_train.jsonl")
    annotators = np.array([np.array(it["annotators"]) for it in res if len(it["annotators"]) == 3])
    annotations = np.array([np.array(it["labels"]) for it in res if len(it["annotators"]) == 3])
    logits = np.load("llm_data/Qwen2.5-32B/train/logits.npy")
    logits = np.array([x for i, x in enumerate(logits[:, :2]) if len(res[i]["annotators"]) == 3])

    model = logistic_random_effects

    mcmc = MCMC(
        NUTS(model),
        num_warmup=1000,
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
        (annotations[train_size:], logits[train_size:], [True] * annotations[train_size:].shape[0])
        if model in [multinomial, item_difficulty]
        else (annotators[train_size:], annotations[train_size:], logits[train_size:], [True] * annotations[train_size:].shape[0])
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