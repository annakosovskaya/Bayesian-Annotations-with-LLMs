import numpy as np

from jax import nn
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam
from numpyro.ops.indexing import Vindex
from numpyro.handlers import mask


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
    else:
        w = numpyro.sample("w", dist.Normal(0, 1).expand([logits.shape[-1], num_classes]).to_event(2))
        bias = numpyro.sample("bias", dist.Normal(0, 1).expand([num_classes]).to_event(1))
        embedding = jnp.einsum('ik,kj->ij', logits, w) + bias  # (num_items, num_classes)

    with numpyro.plate("item", num_items, dim=-2):
        if logits is None:
            c = numpyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})
        else:
            c = numpyro.sample("c", dist.Categorical(logits = embedding[:, np.newaxis, :]), infer={"enumerate": "parallel"})
        with numpyro.plate("position", num_positions):
            if test:
                numpyro.sample("y", dist.Categorical(zeta[c]))
            else:
                numpyro.sample("y", dist.Categorical(zeta[c]), obs=annotations)

def dawid_skene(positions, annotations, logits, test:bool=False):
    """
    This model corresponds to the plate diagram in Figure 2 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            beta = numpyro.sample("beta", dist.Dirichlet(jnp.ones(num_classes)))

    if logits is None:
        pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))
    else:
        w = numpyro.sample("w", dist.Normal(0, 1).expand([logits.shape[-1], num_classes]).to_event(2))
        bias = numpyro.sample("bias", dist.Normal(0, 1).expand([num_classes]).to_event(1))
        embedding = jnp.einsum('ik,kj->ij', logits, w) + bias  # (num_items, num_classes)

    with numpyro.plate("item", num_items, dim=-2):
        if logits is None:
            c = numpyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})
        else:
            c = numpyro.sample("c", dist.Categorical(logits = embedding[:,np.newaxis,:]), infer={"enumerate": "parallel"})

        # here we use Vindex to allow broadcasting for the second index `c`
        # ref: http://num.pyro.ai/en/latest/utilities.html#numpyro.contrib.indexing.vindex
        with numpyro.plate("position", num_positions):
            if test:
                numpyro.sample("y", dist.Categorical(Vindex(beta)[positions, c, :]))
            else:
                numpyro.sample("y", dist.Categorical(Vindex(beta)[positions, c, :]), obs=annotations)

def mace(positions, annotations, logits=None, test:bool=False):
    """
    This model corresponds to the plate diagram in Figure 3 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators):
        epsilon = numpyro.sample("epsilon", dist.Dirichlet(jnp.full(num_classes, 10)))
        theta = numpyro.sample("theta", dist.Beta(0.5, 0.5))

    if logits is None:
        pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))
    else:
        w = numpyro.sample("w", dist.Normal(0, 1).expand([logits.shape[-1], num_classes]).to_event(2))
        bias = numpyro.sample("bias", dist.Normal(0, 1).expand([num_classes]).to_event(1))
        embedding = jnp.einsum('ik,kj->ij', logits, w) + bias  # (num_items, num_classes)

    with numpyro.plate("item", num_items, dim=-2):

        if logits is None:
            c = numpyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})
        else:
            c = numpyro.sample("c", dist.Categorical(logits = embedding[:,np.newaxis,:]), infer={"enumerate": "parallel"})

        with numpyro.plate("position", num_positions):
            s = numpyro.sample(
                "s",
                dist.Bernoulli(1 - theta[positions]),
                infer={"enumerate": "parallel"},
            )
            probs = jnp.where(
                s[..., None] == 0, nn.one_hot(c, num_classes), epsilon[positions]
            )
            if test:
                numpyro.sample("y", dist.Categorical(probs))
            else:
                numpyro.sample("y", dist.Categorical(probs), obs=annotations)



def hierarchical_dawid_skene(positions, annotations, logits=None, test:bool=False):
    """
    This model corresponds to the plate diagram in Figure 4 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        # NB: we define `beta` as the `logits` of `y` likelihood; but `logits` is
        # invariant up to a constant, so we'll follow [1]: fix the last term of `beta`
        # to 0 and only define hyperpriors for the first `num_classes - 1` terms.
        zeta = numpyro.sample(
            "zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        omega = numpyro.sample(
            "Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            # non-centered parameterization
            with handlers.reparam(config={"beta": LocScaleReparam(0)}):
                beta = numpyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
            # pad 0 to the last item
            beta = jnp.pad(beta, [(0, 0)] * (jnp.ndim(beta) - 1) + [(0, 1)])

    if logits is None:
        pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))
    else:
        w = numpyro.sample("w", dist.Normal(0, 1).expand([logits.shape[-1], num_classes]).to_event(2))
        bias = numpyro.sample("bias", dist.Normal(0, 1).expand([num_classes]).to_event(1))
        embedding = jnp.einsum('ik,kj->ij', logits, w) + bias  # (num_items, num_classes)
    

    with numpyro.plate("item", num_items, dim=-2):
        if logits is None:
            c = numpyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})
        else:
            c = numpyro.sample("c", dist.Categorical(logits = embedding[:,np.newaxis,:]), infer={"enumerate": "parallel"})

        with numpyro.plate("position", num_positions):
            if test:
                local_logits = Vindex(beta)[positions, c, :]
                numpyro.sample("y", dist.Categorical(logits=local_logits))
            else:
                local_logits = Vindex(beta)[positions, c, :]
                numpyro.sample("y", dist.Categorical(logits=local_logits), obs=annotations)



def item_difficulty(annotations,logits,mask=None):
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

    w = numpyro.sample("w", dist.Normal(0, 1).expand([logits.shape[-1], num_classes]).to_event(2))
    bias = numpyro.sample("bias", dist.Normal(0, 1).expand([num_classes]).to_event(1))
    embedding = jnp.einsum('ik,kj->ij', logits, w) + bias  # (num_items, num_classes)

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(logits = embedding[:,np.newaxis,:]), infer={"enumerate": "parallel"})

        with handlers.reparam(config={"theta": LocScaleReparam(0)}):
            theta = numpyro.sample("theta", dist.Normal(eta[c], chi[c]).to_event(1))
            theta = jnp.pad(theta, [(0, 0)] * (jnp.ndim(theta) - 1) + [(0, 1)])

        with numpyro.plate("position", num_positions):
            numpyro.sample("y", dist.Categorical(logits=theta), obs=annotations, obs_mask=mask)

def logistic_random_effects(positions, annotations,logits, test=False):
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        zeta = numpyro.sample(
            "zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        omega = numpyro.sample(
            "Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )
        # chi = numpyro.sample(
        #     "Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        # )

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            with handlers.reparam(config={"beta": LocScaleReparam(0)}):
                beta = numpyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
                beta = jnp.pad(beta, [(0, 0)] * (jnp.ndim(beta) - 1) + [(0, 1)])

    # theta: item-specific random effect, regressed on logits using numpyro
    num_items = logits.shape[0]
    num_features = logits.shape[-1]

    w = numpyro.sample("w", dist.Normal(0, 1).expand([num_classes - 1, num_features]).to_event(2))
    bias = numpyro.sample("bias", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1))

    embedding = jnp.einsum('ik,jk->ij', logits, w) + bias  # (num_items, num_classes-1)

    # theta_raw = numpyro.sample("theta_raw", dist.Normal(0, embedding).to_event(1))
    # theta = jnp.pad(theta_raw, [(0, 0)] * (jnp.ndim(theta_raw) - 1) +  [(0, 1)])

    with numpyro.plate("item", num_items, dim=-2):
        
        c = numpyro.sample("c", dist.Categorical(logits = embedding), infer={"enumerate": "parallel"})        

        with numpyro.plate("position", num_positions):
            y_logits = Vindex(beta)[positions, c, :]
            with numpyro.plate("position", num_positions):
                if test:
                    numpyro.sample("y", dist.Categorical(logits=y_logits))
                else:
                    numpyro.sample("y", dist.Categorical(logits=y_logits), obs=annotations)
