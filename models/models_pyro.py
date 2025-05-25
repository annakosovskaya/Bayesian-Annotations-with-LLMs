

import torch

from torch.nn import functional as F
from torch import nn


import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.infer.reparam import LocScaleReparam
from pyro.ops.indexing import Vindex


def multinomial(annotations,logits=None,test:bool=False):
    """
    This model corresponds to the plate diagram in Figure 1 of reference [1].
    """
    num_classes = int(torch.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with pyro.plate("class", num_classes):
        zeta = pyro.sample("zeta", dist.Dirichlet(torch.ones(num_classes)))

    if logits is None:
        pi = pyro.sample("pi", dist.Dirichlet(torch.ones(num_classes)))
    # pi = pyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)),obs=nn.softmax(logits).mean(0))
    # pi = nn.softmax(logits).mean(0)
    # c = pyro.sample("c", dist.Categorical(logits = logits[:,np.newaxis,:]), infer={"enumerate": "parallel"})

    with pyro.plate("item", num_items, dim=-2):
        if logits is None:
            c = pyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})
        else:
            c = pyro.sample("c", dist.Categorical(logits = logits[:, torch.newaxis,:]), infer={"enumerate": "parallel"})
        with pyro.plate("position", num_positions):
            if test:
                pyro.sample("y", dist.Categorical(zeta[c]))
            else:
                pyro.sample("y", dist.Categorical(zeta[c]), obs=annotations)

def dawid_skene(positions, annotations,logits):
    """
    This model corresponds to the plate diagram in Figure 2 of reference [1].
    """
    num_annotators = int(torch.max(positions)) + 1
    num_classes = int(torch.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with pyro.plate("annotator", num_annotators, dim=-2):
        with pyro.plate("class", num_classes):
            beta = pyro.sample("beta", dist.Dirichlet(torch.ones(num_classes, device=annotations.device)))

   
    with pyro.plate("item", num_items, dim=-2):
        # c = pyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})
        c = pyro.sample("c", dist.Categorical(logits = logits[:,torch.newaxis,:]), infer={"enumerate": "parallel"})

        # here we use Vindex to allow broadcasting for the second index `c`
        # ref: http://num.pyro.ai/en/latest/utilities.html#pyro.contrib.indexing.vindex
        with pyro.plate("position", num_positions):
            y=pyro.sample(
                "y", dist.Categorical(Vindex(beta)[positions, c, :]), obs=annotations
            )

def mace(positions, annotations, logits, test:bool=False): 
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

    with numpyro.plate("item", num_items, dim=-2):

        if logits is None:
            c = numpyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})
        else:
            c = numpyro.sample("c", dist.Categorical(logits = logits[:,np.newaxis,:]), infer={"enumerate": "parallel"})

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



def hierarchical_dawid_skene(positions, annotations,logits, test:bool=False):
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
    

    with numpyro.plate("item", num_items, dim=-2):
        if logits is None:
            c = numpyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})
        else:
            c = numpyro.sample("c", dist.Categorical(logits = logits[:,np.newaxis,:]), infer={"enumerate": "parallel"})

        with numpyro.plate("position", num_positions):
            if test:
                local_logits = Vindex(beta)[positions, c, :]
                numpyro.sample("y", dist.Categorical(logits=local_logits))
            else:
                local_logits = Vindex(beta)[positions, c, :]
                numpyro.sample("y", dist.Categorical(logits=local_logits), obs=annotations)



def item_difficulty(annotations,logits):
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_classes = int(torch.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with pyro.plate("class", num_classes):
        eta = pyro.sample(
            "eta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        chi = pyro.sample(
            "Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    # pi = pyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))
    # pi = pyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)),obs = nn.softmax(logits).mean(0))
    # pi = nn.softmax(logits).mean(0)
    c = pyro.sample("c", dist.Categorical(logits = logits[:,torch.newaxis,:]), infer={"enumerate": "parallel"})

    with pyro.plate("item", num_items, dim=-2):
        # c = pyro.sample("c", dist.Categorical(probs=pi), infer={"enumerate": "parallel"})

        with poutine.reparam(config={"theta": LocScaleReparam(0)}):
            theta = pyro.sample("theta", dist.Normal(eta[c], chi[c]).to_event(1))
            theta = F.pad(theta, [(0, 0)] * (theta.ndim - 1) + [(0, 1)])

        with pyro.plate("position", annotations.shape[-1]):
            pyro.sample("y", dist.Categorical(logits=theta), obs=annotations)


def logistic_random_effects(positions, annotations,logits):
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_annotators = int(torch.max(positions)) + 1
    num_classes = int(torch.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with pyro.plate("class", num_classes):
        zeta = pyro.sample(
            "zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        omega = pyro.sample(
            "Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )
        chi = pyro.sample(
            "Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    with pyro.plate("annotator", num_annotators, dim=-2):
        with pyro.plate("class", num_classes):
            with poutine.reparam(config={"beta": LocScaleReparam(0)}):
                beta = pyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
                beta = F.pad(beta, [(0, 0)] * (beta.ndim - 1) + [(0, 1)])

    # pi = pyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))
    # pi = pyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)),obs = nn.softmax(logits).mean(0))
    # pi = nn.softmax(logits).mean(0)
    c = pyro.sample("c", dist.Categorical(logits = logits[:, torch.newaxis,:]), infer={"enumerate": "parallel"})

    with pyro.plate("item", num_items, dim=-2):
        # c = pyro.sample("c", dist.Categorical(pi), infer={"enumerate": "parallel"})

        with poutine.reparam(config={"theta": LocScaleReparam(0)}):
            theta = pyro.sample("theta", dist.Normal(0, chi[c]).to_event(1))
            theta = F.pad(theta, [(0, 0)] * (theta.ndim - 1) + [(0, 1)])

        with pyro.plate("position", num_positions):
            logits = Vindex(beta)[positions, c, :] - theta
            pyro.sample("y", dist.Categorical(logits=logits), obs=annotations)
