import config
import jax
from jax import nn
import numpy as np
from jax import grad, jit, vmap

import jax.numpy as jnp

from sklearn.metrics import log_loss,f1_score
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


from utils.data_utils import read_jsonl

from models.logistic_regression_jax import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit

from utils.data_utils import read_jsonl, process_annotations, create_annotator_mapping



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

    # Collect results across splits
    js_divs = []
    kl_divs = []
    f1s    = []

    js_divs_argmax = []
    kl_divs_argmax = []
    f1s_argmax    = []


    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\n=== Fold {fold_idx} ===")

        # Slice data for this split
        logits_train       = logits[train_idx]
        logits_test        = logits[test_idx]

        ann_train          = annotations[train_idx]
        ann_test           = annotations[test_idx]

        ators_train        = annotators[train_idx]
        ators_test         = annotators[test_idx]

        model = LogisticRegression(input_dim=logits_train.shape[1] + 1)
        params = model.params

        train_size = round(annotations.shape[0] * 0.9)

        # Interleave logits and annotations for train and test separately
        interleaved_logits = []
        interleaved_annotations = []

        # For train set
        for i in train_idx:
            for j in range(annotators.shape[1]):
                interleaved_logits.append(logits[i])
                interleaved_annotations.append(annotations[i][j])

        # For test set
        for i in test_idx:
            for j in range(annotators.shape[1]):
                interleaved_logits.append(logits[i])
                interleaved_annotations.append(annotations[i][j])

        interleaved_logits = np.array(interleaved_logits)
        interleaved_annotations = np.array(interleaved_annotations)

        train_data = (
            (np.concat((np.expand_dims(ators_train.flatten(), axis=1), interleaved_logits[:train_size * annotators.shape[1]]), axis=1), interleaved_annotations[:train_size * annotators.shape[1]])
        )

        test_data = (
            (np.concat((np.expand_dims(ators_test.flatten(), axis=1), interleaved_logits[train_size * annotators.shape[1]:]), axis=1), interleaved_annotations[train_size * annotators.shape[1]:])
        )

        # Training loop
        clf = SklearnLogisticRegression(max_iter=10000)
        clf.fit(train_data[0], train_data[1])

        annotator_probs = clf.predict_proba(test_data[0])[:, 1]
        # Reshape so each row corresponds to one sample, columns to annotators
        annotator_probs = annotator_probs.reshape(-1, annotators.shape[1])

        pred_probs = np.vstack((annotator_probs.mean(1), 1 - annotator_probs.mean(1)))
        emp_probs  = np.vstack((ann_test.mean(1),           1 - ann_test.mean(1)))

        js = np.power(jensenshannon(emp_probs, pred_probs), 2).mean()
        kl = entropy(emp_probs, pred_probs).mean()
        fv = f1_score(np.rint(ann_test.mean(1)), np.rint(np.rint(annotator_probs).mean(1)), pos_label=1)

        print('---- logistic regression baseline ----')
        print(f"Average JS divergence = {js:.4f}")
        print(f"Average KL divergence = {kl:.4f}")
        print(f"Binary F1 (majority vote) = {fv:.4f}")

        js_divs.append(js)
        kl_divs.append(kl)
        f1s.append(fv)

        # ---- part for argmax baseline  ----
        num_train_fold = len(train_idx) * annotators.shape[1]
        test_interleaved_logits = interleaved_logits[num_train_fold:]

        # pred_argmax will have shape (num_test_items, num_annotators) with 0/1 predictions for each annotation.
        pred_argmax = np.argmax(test_interleaved_logits, axis=-1).reshape(-1, annotators.shape[1])
        
        # predicted probabilities -> reshape to (num_test_items, num_annotators)
        p1_test_individual = jax.nn.softmax(test_interleaved_logits, axis=-1)[:, 1]
        p1_test_reshaped = p1_test_individual.reshape(-1, annotators.shape[1])

        # Average P(class=1) across annotators for each item
        p1_test_item_avg = p1_test_reshaped.mean(axis=1)
        # Construct the probability distribution [P(class=1), P(class=0)] for each test item
        probs_for_js_kl_argmax = np.vstack((p1_test_item_avg, 1 - p1_test_item_avg))

        print('---- argmax baseline ----')
        js_argmax = np.power(jensenshannon(emp_probs, probs_for_js_kl_argmax), 2).mean()
        kl_argmax = entropy(emp_probs, probs_for_js_kl_argmax).mean()
        
        # np.rint(ann_test.mean(1)) gives the majority vote for true labels.
        # np.rint(pred_argmax.mean(axis=1)) gives the majority vote for predicted labels from argmax.
        # pred_argmax already contains 0/1 predictions from np.argmax.
        fv_argmax = f1_score(np.rint(ann_test.mean(1)), np.rint(pred_argmax.mean(axis=1)), pos_label=1)
        print(f"Average JS divergence: {js_argmax:.4f}, Mean KL divergence: {kl_argmax:.4f}, Mean F1 (maj. vote): {fv_argmax:.4f}")

        js_divs_argmax.append(js_argmax)
        kl_divs_argmax.append(kl_argmax)
        f1s_argmax.append(fv_argmax)

    # --- Summary across folds ---
    print("\n=== Cross-Validation Summary for logistic regression baseline ===")
    print(f"Mean JS divergence: {np.mean(js_divs):.4f} ± {np.std(js_divs):.4f}")
    print(f"Mean KL divergence: {np.mean(kl_divs):.4f} ± {np.std(kl_divs):.4f}")
    print(f"Mean F1 (maj. vote): {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print("\n=== Cross-Validation Summary for argmax baseline ===")
    print(f"Mean JS divergence: {np.mean(js_divs_argmax):.4f} ± {np.std(js_divs_argmax):.4f}")
    print(f"Mean KL divergence: {np.mean(kl_divs_argmax):.4f} ± {np.std(kl_divs_argmax):.4f}")
    print(f"Mean F1 (maj. vote): {np.mean(f1s_argmax):.4f} ± {np.std(f1s_argmax):.4f}")
