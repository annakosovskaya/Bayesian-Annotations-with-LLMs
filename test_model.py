import numpy as np
import os
import config

import numpy as np

from jax import random, vmap
import jax.numpy as jnp

from numpyro.infer import MCMC, NUTS, Predictive


from sklearn.metrics import log_loss,f1_score
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import StratifiedShuffleSplit



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

    #Prepare stratified splits
    annotations_balance = np.sum(annotations, axis=1)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)

    splits = list(sss.split(logits, annotations_balance))

    # Choose model here
    model = item_difficulty

    # Collect results across splits
    js_divs = []
    kl_divs = []
    f1s    = []


    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\n=== Fold {fold_idx} ===")

        # Slice data for this split
        logits_train       = logits[train_idx]
        logits_test        = logits[test_idx]

        ann_train          = annotations[train_idx]
        ann_test           = annotations[test_idx]

        ators_train        = annotators[train_idx]
        ators_test         = annotators[test_idx]

        # Build NumPyro inputs depending on model signature
        if model == multinomial:
            train_data = (ann_train, logits_train)
            test_data  = (ann_test,  logits_test, [True] * len(test_idx))
        elif model == item_difficulty:
            mask = jnp.array([True] * logits_train.shape[0] + [False] * logits_test.shape[0]).reshape(-1, 1)
            mask = jnp.tile(mask, (1, 3))
            train_data = (annotations,logits, mask)
        elif model == dawid_skene:
            train_data = (positions_, ann_train, masks_[train_idx], global_num_classes, True, logits_train)
            test_data  = (positions_, ann_test,  masks_[test_idx],  global_num_classes, True, logits_test, [True] * len(test_idx))
        else:  # hierarchical_dawid_skene, mace, logistic_random_effects
            train_data = (ators_train, ann_train, logits_train)
            test_data  = (ators_test,  ann_test,  logits_test,  [True] * len(test_idx))

        mcmc = MCMC(
            NUTS(model),
            num_warmup=1000,
            num_samples=500,
            num_chains=1,
            progress_bar=not ("NUMPYRO_SPHINXBUILD" in os.environ),
        )

        seed = 0 + fold_idx
        mcmc.run(random.PRNGKey(seed), *train_data)
        mcmc.print_summary()

        posterior = mcmc.get_samples()
        predictive = Predictive(model, posterior, infer_discrete=True)
        if model == item_difficulty:
            discrete_samples = predictive(random.PRNGKey(seed + 100), *train_data)
        else:
            discrete_samples = predictive(random.PRNGKey(seed + 100), *test_data)

        if model == item_difficulty:
            annotator_probs = vmap(lambda x: x.mean(0), in_axes=1)(
                discrete_samples["y_unobserved"][:, logits_train.shape[0]:],
            )
        else:
            annotator_probs = vmap(lambda x: x.mean(0), in_axes=1)(discrete_samples["y"])
        pred_probs = np.vstack((annotator_probs.mean(1), 1 - annotator_probs.mean(1)))
        emp_probs  = np.vstack((ann_test.mean(1),           1 - ann_test.mean(1)))

        js = np.power(jensenshannon(emp_probs, pred_probs), 2).mean()
        kl = entropy(emp_probs, pred_probs).mean()
        fv = f1_score(np.rint(ann_test.mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1)

        print(f"Average JS divergence = {js:.4f}")
        print(f"Average KL divergence = {kl:.4f}")
        print(f"Binary F1 (majority vote) = {fv:.4f}")

        js_divs.append(js)
        kl_divs.append(kl)
        f1s.append(fv)

    # --- Summary across folds ---
    print("\n=== Cross-Validation Summary ===")
    print(f"Mean JS divergence: {np.mean(js_divs):.4f} ± {np.std(js_divs):.4f}")
    print(f"Mean KL divergence: {np.mean(kl_divs):.4f} ± {np.std(kl_divs):.4f}")
    print(f"Mean F1 (maj. vote): {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")


    # model = dawid_skene  # choose your model
    #
    # mcmc = MCMC(
    #     NUTS(model),
    #     num_warmup=1000,
    #     num_samples=500,
    #     num_chains=1,
    #     progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    # )
    #
    # train_size = round(annotations.shape[0] * 0.9)
    #
    # train_data = (
    #     (annotations[:train_size], logits[:train_size])
    #     if model in [multinomial, item_difficulty]
    #     else (positions_, annotations_[:train_size], masks_[:train_size], global_num_classes, True, logits[:train_size])
    #     if model == dawid_skene
    #     else (annotators[:train_size], annotations[:train_size], logits[:train_size])
    # )
    #
    # test_data = (
    #     (annotations[train_size:], logits[train_size:], [True] * annotations[train_size:].shape[0])
    #     if model in [multinomial, item_difficulty]
    #     else (positions_, annotations_[train_size:], masks_[train_size:], global_num_classes, True, logits[train_size:], [True] * annotations[train_size:].shape[0])
    #     if model == dawid_skene
    #     else (annotators[train_size:], annotations[train_size:], logits[train_size:], [True] * annotations[train_size:].shape[0])
    # )
    #
    # mcmc.run(random.PRNGKey(0), *train_data)
    # mcmc.print_summary()
    #
    # posterior_samples = mcmc.get_samples()
    # predictive = Predictive(model, posterior_samples, infer_discrete=True)
    # discrete_samples = predictive(random.PRNGKey(1), *test_data)
    #
    # annotator_probs = vmap(lambda x: x.mean(0), in_axes=1)(discrete_samples["y"]    )
    #
    # pred_probs = np.vstack((annotator_probs.mean(1), 1 - annotator_probs.mean(1)))
    # emp_probs = np.vstack((annotations[train_size:].mean(1), 1 - annotations[train_size:].mean(1)))
    #
    # print(f'Average Jensen-Shannon divergence across items = {np.power(jensenshannon(emp_probs, pred_probs), 2).mean()}')
    # print(f'Average KL divergence across items = {entropy(emp_probs, pred_probs).mean()}')
    # print(
    #     f'Binary F1 score with majority vote = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1)}')

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
