import jax
from jax import grad, jit
from functools import partial

import jax.numpy as jnp

class LogisticRegression:
    def __init__(self, n_features):
        key = jax.random.PRNGKey(0)
        self.params = {
            "w": jax.random.normal(key, (n_features,)),
            "b": 0.0
        }

    def predict_proba(self, X, params=None):
        if params is None:
            params = self.params
        logits = jnp.dot(X, params["w"]) + params["b"]
        return jax.nn.sigmoid(logits)

    def predict(self, X, params=None):
        proba = self.predict_proba(X, params)
        return (proba >= 0.5).astype(jnp.int32)

    def loss(self, params, X, y):
        preds = self.predict_proba(X, params)
        # Binary cross-entropy loss
        return -jnp.mean(y * jnp.log(preds + 1e-8) + (1 - y) * jnp.log(1 - preds + 1e-8))

    def fit(self, X, y, lr=0.1, epochs=100):
        loss_grad = jit(grad(self.loss))
        params = self.params
        for epoch in range(epochs):
            grads = loss_grad(params, X, y)
            params = {
                "w": params["w"] - lr * grads["w"],
                "b": params["b"] - lr * grads["b"]
            }
        self.params = params

# Example usage:
# X = jnp.array([[...], ...])  # shape (n_samples, n_features)
# y = jnp.array([...])         # shape (n_samples,)
# model = LogisticRegression(n_features=X.shape[1])
# model.fit(X, y)
# preds = model.predict(X)