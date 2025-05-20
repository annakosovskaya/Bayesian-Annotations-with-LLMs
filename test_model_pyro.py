import torch

from pyro.infer import MCMC, NUTS, Predictive

import numpy as np

from sklearn.metrics import log_loss, f1_score

from utils.data_utils import read_jsonl
from models.models_pyro import multinomial, item_difficulty, dawid_skene, hierarchical_dawid_skene, mace, logistic_random_effects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(device)
    res = read_jsonl("data/ghc_train.jsonl")
    annotators = torch.from_numpy(np.array([np.array(it["annotators"]) for it in res if len(it["annotators"]) == 3])).to(device)
    annotations = torch.from_numpy(np.array([np.array(it["labels"]) for it in res if len(it["annotators"]) == 3])).to(device)
    logits = torch.from_numpy(np.load("data/llm_data/Qwen2.5-32B/train/logits.npy"))
    logits = torch.from_numpy(np.array([x for i, x in enumerate(logits[:, :2]) if len(res[i]["annotators"]) == 3])).to(device)

    model = dawid_skene

    mcmc = MCMC(
        NUTS(model),
        warmup_steps=1000,
        num_samples=500,
        num_chains=1
    )

    train_size = max(1, round(annotations.shape[0] * 0.5))

    train_data = (
        (annotations[:train_size], logits[:train_size])
        if model in [multinomial, item_difficulty]
        else (annotators[:train_size], annotations[:train_size], logits[:train_size])
    )

    test_data = (
        (annotations[train_size:], logits[train_size:])
        if model in [multinomial, item_difficulty]
        else (annotators[train_size:], annotations[train_size:], logits[train_size:])
    )

    mcmc.run(*train_data)
    mcmc.summary()

    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples, infer_discrete=True).to(device)
    discrete_samples = predictive(*test_data)

    annotator_probs = torch.vmap(lambda x: x.mean(0), in_axes=1)(
        discrete_samples["y"]
    )

    # Move tensors back to CPU for sklearn metrics
    ann_test = annotations[train_size:].cpu()
    ann_probs = annotator_probs.cpu()

    print(f'log loss over positions = {log_loss(ann_test.flatten(), ann_probs.flatten())}')
    print(f'log loss over items = {log_loss(torch.round(ann_test.mean(1)), ann_probs.mean(1))}')
    print('F1 score with position predictions')
    print(
        f'Binary F1 score = {f1_score(ann_test.flatten(), torch.round(ann_probs).flatten(), pos_label=1)}')
    print(
        f'Micro F1 score = {f1_score(ann_test.flatten(), torch.round(ann_probs).flatten(), pos_label=1, average="micro")}')
    print(
        f'Macro F1 score = {f1_score(ann_test.flatten(), torch.round(ann_probs).flatten(), pos_label=1, average="macro")}')
    print(
        f'Weighted F1 score = {f1_score(ann_test.flatten(), torch.round(ann_probs).flatten(), pos_label=1, average="weighted")}')
    print('F1 score with predictions of number of 1s out of all positions')
    print(
        f'Micro F1 score = {f1_score(ann_test.sum(1), torch.round(ann_probs).sum(1), average="micro")}')
    print(
        f'Macro F1 score = {f1_score(ann_test.sum(1), torch.round(ann_probs).sum(1), average="macro")}')
    print(
        f'Weighted F1 score = {f1_score(ann_test.sum(1), torch.round(ann_probs).sum(1), average="weighted")}')
    print('F1 score with majority vote')
    print(
        f'Binary F1 score = {f1_score(torch.round(ann_test.mean(1)), torch.round(torch.round(ann_probs).mean(1)), pos_label=1)}')
    print(
        f'Micro F1 score = {f1_score(torch.round(ann_test.mean(1)), torch.round(torch.round(ann_probs).mean(1)), pos_label=1, average="micro")}')
    print(
        f'Macro F1 score = {f1_score(torch.round(ann_test.mean(1)), torch.round(torch.round(ann_probs).mean(1)), pos_label=1, average="macro")}')
    print(
        f'Weighted F1 score = {f1_score(torch.round(ann_test.mean(1)), torch.round(torch.round(ann_probs).mean(1)), pos_label=1, average="weighted")}')