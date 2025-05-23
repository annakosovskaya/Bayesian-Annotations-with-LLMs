
import jax
import numpy as np
from jax import grad, jit, vmap

import jax.numpy as jnp

from sklearn.metrics import log_loss,f1_score
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


from utils.data_utils import read_jsonl

from models.logistic_regression_jax import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

# Example usage:
if __name__ == "__main__":

    res = read_jsonl("data/ghc_train.jsonl")
    annotators = np.array([np.array(it["annotators"]) for it in res if len(it["annotators"]) == 3])
    annotations = np.array([np.array(it["labels"]) for it in res if len(it["annotators"]) == 3])
    logits = np.load("llm_data/Qwen2.5-32B/train/logits.npy")
    logits = np.array([x for i, x in enumerate(logits[:, :2]) if len(res[i]["annotators"]) == 3])

    model = LogisticRegression(input_dim=logits.shape[1] + 1)
    params = model.params

    train_size = round(annotations.shape[0] * 0.9)

    interleaved_logits = []
    interleaved_annotations = []

    for i in range(annotators.shape[0]):
        for j in range(annotators.shape[1]):
            interleaved_logits.append(logits[i])
            interleaved_annotations.append(annotations[i][j])

    interleaved_logits = np.array(interleaved_logits)
    interleaved_annotations = np.array(interleaved_annotations)

    train_data = (
        (np.concat((np.expand_dims(annotators[:train_size].flatten(), axis=1), interleaved_logits[:train_size * annotators.shape[1]]), axis=1), interleaved_annotations[:train_size * annotators.shape[1]])
    )

    test_data = (
        (np.concat((np.expand_dims(annotators[train_size:].flatten(), axis=1), interleaved_logits[train_size * annotators.shape[1]:]), axis=1), interleaved_annotations[train_size * annotators.shape[1]:])
    )

    # Training loop
    clf = SklearnLogisticRegression(max_iter=10000)
    clf.fit(train_data[0], train_data[1])

    pred = clf.predict(test_data[0])
    pred_probs = clf.predict_proba(test_data[0])[:, 1]

    print(f'Average Jensen-Shannon divergence across items= {np.power(jensenshannon(test_data[1], pred_probs), 2).mean()}')
    print(f'Average KL divergence across items= {entropy(test_data[1], pred_probs).mean()}')
    print(
        f'Binary F1 score with majority vote = {f1_score(np.rint(annotations[train_size:].mean(1)), np.rint(np.rint(pred.reshape((-1, 3))).mean(1)), pos_label=1)}')