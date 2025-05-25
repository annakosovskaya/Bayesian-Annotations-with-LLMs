import numpy as np
import os
import config

import numpy as np

from jax import random, vmap

from numpyro.infer import MCMC, NUTS, Predictive


from sklearn.metrics import log_loss,f1_score
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


from utils.data_utils import read_jsonl, process_annotations, create_annotator_mapping
from models.models_numpyro import multinomial, item_difficulty, dawid_skene, hierarchical_dawid_skene, mace, logistic_random_effects

if __name__ == "__main__":
    res = read_jsonl(config.TRAIN_FILE_PATH)
    annotators = np.array([np.array(it["annotators"]) for it in res if len(it["annotators"]) == 3])
    annotations = np.array([np.array(it["labels"]) for it in res if len(it["annotators"]) == 3])
    positions_, annotations_, masks_ = process_annotations(res)
    global_num_classes = int(np.max(annotations_)) + 1
    logits = np.load(config.LOGITS_PATH)
    logits = np.array([x for i, x in enumerate(logits[:, :2]) if len(res[i]["annotators"]) == 3])

    model = dawid_skene  # choose your model

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
        else (positions_, annotations_[:train_size], masks_[:train_size], global_num_classes, True, logits[:train_size])
        if model == dawid_skene
        else (annotators[:train_size], annotations[:train_size], logits[:train_size])
    )

    test_data = (
        (annotations[train_size:], logits[train_size:], [True] * annotations[train_size:].shape[0])
        if model in [multinomial, item_difficulty]
        else (positions_, annotations_[train_size:], masks_[train_size:], global_num_classes, True, logits[train_size:], [True] * annotations[train_size:].shape[0])
        if model == dawid_skene
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

    pred_probs = np.vstack((annotator_probs.mean(1), 1 - annotator_probs.mean(1)))
    emp_probs = np.vstack((annotations[train_size:].mean(1), 1 - annotations[train_size:].mean(1)))

    print(f'Average Jensen-Shannon divergence across items = {np.power(jensenshannon(emp_probs, pred_probs), 2).mean()}')
    print(f'Average KL divergence across items = {entropy(emp_probs, pred_probs).mean()}')
    print(
        f'Binary F1 score with majority vote = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1)}')

    # print(f'log loss over positions = {log_loss(annotations[train_size:].flatten(), annotator_probs.flatten())}')
    # print(f'log loss over items = {log_loss(np.rint(annotations[train_size:].mean(1)), annotator_probs.mean(1))}')
    # print('F1 score with position predictions')
    # print(
    #     f'Binary F1 score = {f1_score(annotations[train_size:].flatten(), np.rint(annotator_probs).flatten(), pos_label=1)}')
    # print(
    #     f'Micro F1 score = {f1_score(annotations[train_size:].flatten(), np.rint(annotator_probs).flatten(), pos_label=1, average="micro")}')
    # print(
    #     f'Macro F1 score = {f1_score(annotations[train_size:].flatten(), np.rint(annotator_probs).flatten(), pos_label=1, average="macro")}')
    # print(
    #     f'Weighted F1 score = {f1_score(annotations[train_size:].flatten(), np.rint(annotator_probs).flatten(), pos_label=1, average="weighted")}')
    # print('F1 score with predictions of number of 1s out of all positions')
    # print(
    #     f'Micro F1 score = {f1_score(annotations[train_size:].sum(1), np.rint(annotator_probs).sum(1), average="micro")}')
    # print(
    #     f'Macro F1 score = {f1_score(annotations[train_size:].sum(1), np.rint(annotator_probs).sum(1), average="macro")}')
    # print(
    #     f'Weighted F1 score = {f1_score(annotations[train_size:].sum(1), np.rint(annotator_probs).sum(1), average="weighted")}')
    # print('F1 score with majority vote')
    # print(
    #     f'Binary F1 score = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1)}')
    # print(
    #     f'Micro F1 score = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1, average="micro")}')
    # print(
    #     f'Macro F1 score = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1, average="macro")}')
    # print(
    #     f'Weighted F1 score = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1, average="weighted")}')
