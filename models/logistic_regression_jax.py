import jax
from jax import grad, jit, vmap

import jax.numpy as jnp

class LogisticRegression:
    def __init__(self, input_dim):
        # Initialize weights and bias
        self.params = {
            'w': jnp.zeros(input_dim),
            'b': 0.0
        }

    def predict_proba(self, params, X):
        logits = jnp.dot(X, params['w']) + params['b']
        return jax.nn.sigmoid(logits)

    def predict(self, params, X, threshold=0.5):
        proba = self.predict_proba(params, X)
        return (proba >= threshold).astype(jnp.int32), proba.astype(jnp.float32)

    def loss(self, params, X, y):
        proba = self.predict_proba(params, X)
        # Binary cross-entropy loss
        return -jnp.mean(y * jnp.log(proba + 1e-8) + (1 - y) * jnp.log(1 - proba + 1e-8))

    def update(self, params, X, y, lr=0.01):
        grads = jax.grad(self.loss)(params, X, y)
        new_params = {
            'w': params['w'] - lr * grads['w'],
            'b': params['b'] - lr * grads['b']
        }
        return new_params
