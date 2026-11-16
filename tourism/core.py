"""Numerical model, ordered-logistic observations, SMC and finite-action EVSI."""
from dataclasses import dataclass
from itertools import combinations, product
import numpy as np
from scipy.special import expit, logsumexp

ANSWERS = ("A", "B", "tie", "skip")

@dataclass(frozen=True)
class Observation:
    event_id: str
    i: int
    j: int | None
    target: str
    answer: str
    context: tuple

class Posterior:
    def __init__(self, shared, media, context, particles=512, seed=11,
                 sigma=1., alpha=2., beta=2., tau=1., delta=.2,
                 mode="transfer", rejuvenation=2, ess_fraction=.5):
        if min(sigma, alpha, beta, tau, delta) <= 0:
            raise ValueError("Proper scales and positive tie margin required")
        if particles < 2 or mode not in {"transfer", "independent", "shared", "single"}:
            raise ValueError("Invalid particle count or mode")
        self.d, self.m, self.c = shared, media, context
        self.sigma, self.alpha, self.beta = sigma, alpha, beta
        self.tau, self.delta, self.mode = tau, delta, mode
        self.rejuvenation, self.ess_fraction = rejuvenation, ess_fraction
        self.rng = np.random.default_rng(seed)
        self.real_dim = 2*shared + media + context
        real = self.rng.normal(0, sigma, (particles, self.real_dim))
        g = self.rng.beta(alpha, beta, (particles, shared))
        self.state = np.column_stack((real, np.log(g)-np.log1p(-g)))
        self.weights = np.full(particles, 1/particles)
        self.history, self.accepted = [], set()
        self.resamples, self.mh_acceptances, self.mh_proposals = 0, 0, 0

    def utilities(self, phi, psi, chi, state=None):
        s = self.state if state is None else state
        a, b = s[:, :self.d], s[:, self.d:2*self.d]
        r = s[:, 2*self.d:2*self.d+self.m]
        h = s[:, 2*self.d+self.m:self.real_dim]
        g = expit(s[:, self.real_dim:])
        if self.mode == "independent": g = np.zeros_like(g)
        if self.mode == "shared": g = np.ones_like(g)
        v = a @ phi.T + r @ psi.T
        t = (g*a + (1-g)*b) @ phi.T + h @ chi.T
        if self.mode == "single": t = v.copy()
        return v, t

    def comparison(self, delta):
        a = expit((delta-self.delta)/self.tau)
        b = expit((-delta-self.delta)/self.tau)
        return np.stack((a, b, np.maximum(0, 1-a-b)), axis=-1)

    def outcome_probs(self, phi, psi, chi, i, j, targets=("V", "T"), skip=.15):
        if not 0 <= skip <= 1: raise ValueError("skip outside [0,1]")
        u = self.utilities(phi, psi, chi)
        cols = []
        for target in targets:
            p = self.comparison(u[target == "T"][:, i]-u[target == "T"][:, j])
            cols.append(np.column_stack(((1-skip)*p, np.full(len(p), skip))))
        return np.stack([np.prod([p[:, k] for p,k in zip(cols,ks)], axis=0)
                         for ks in product(range(4), repeat=len(targets))], axis=1)

    def log_likelihood(self, obs, phi, psi, context_fn, state=None):
        n = len(self.state if state is None else state)
        if obs.answer == "skip": return np.zeros(n)
        if obs.target not in {"V", "T"}: raise ValueError("Unknown target")
        chi = context_fn(obs.context)  # Never overwrite historic context.
        u = self.utilities(phi, psi, chi, state)[obs.target == "T"]
        if obs.j is None:
            if obs.target != "V" or obs.answer not in {"A", "B"}:
                raise ValueError("Only explicit binary viewing responses allowed")
            p = expit(u[:, obs.i])
            if obs.answer == "B": p = 1-p
        else:
            if obs.answer not in ANSWERS: raise ValueError("Unknown answer")
            p = self.comparison(u[:, obs.i]-u[:, obs.j])[:, ANSWERS.index(obs.answer)]
        return np.log(np.maximum(p, np.finfo(float).tiny))

    def log_prior(self, state):
        # Beta prior on g with logit-coordinate Jacobian: alpha*log(g)+beta*log(1-g).
        z = state[:, self.real_dim:]
        return (-.5*np.sum((state[:, :self.real_dim]/self.sigma)**2, axis=1)
                - self.alpha*np.logaddexp(0, -z).sum(axis=1)
                - self.beta*np.logaddexp(0, z).sum(axis=1))

    def log_density(self, state, phi, psi, context_fn):
        lp = self.log_prior(state)
        for obs in self.history:
            lp += self.log_likelihood(obs, phi, psi, context_fn, state)
        return lp

    @property
    def ess(self): return float(1/np.square(self.weights).sum())

    def update(self, obs, phi, psi, context_fn):
        if obs.event_id in self.accepted:
            previous=next(o for o in self.history if o.event_id==obs.event_id)
            if previous!=obs: raise ValueError("Conflicting reuse of an evidence event")
            return False
        ll = self.log_likelihood(obs, phi, psi, context_fn)
        self.accepted.add(obs.event_id)
        self.history.append(obs)
        lw = np.log(np.maximum(self.weights, np.finfo(float).tiny)) + ll
        self.weights = np.exp(lw-logsumexp(lw))
        if self.ess < self.ess_fraction*len(self.weights):
            self.resample_move(phi, psi, context_fn)
        return True

    def resample_move(self, phi, psi, context_fn):
        n = len(self.weights)
        positions = (self.rng.random()+np.arange(n))/n
        indices = np.searchsorted(np.cumsum(self.weights), positions).clip(max=n-1)
        self.state = self.state[indices].copy()
        self.weights.fill(1/n)
        self.resamples += 1
        lp = self.log_density(self.state, phi, psi, context_fn)
        for _ in range(self.rejuvenation):
            proposal = self.state + self.rng.normal(0, .15, self.state.shape)
            plp = self.log_density(proposal, phi, psi, context_fn)
            accept = np.log(self.rng.random(n)) < plp-lp
            self.state[accept], lp[accept] = proposal[accept], plp[accept]
            self.mh_acceptances += int(accept.sum()); self.mh_proposals += n


def list_scores(v, t, actions, distance, weights):
    w = np.asarray(weights, dtype=float)
    if len(w) != 3 or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError("Three confirmed simplex weights required")
    if not actions: raise ValueError("Empty action space")
    k = len(actions[0]); rho = 1/np.log2(np.arange(k)+2); rho /= rho.sum()
    a = np.asarray(actions)
    fv, ft = expit(v)[:, a] @ rho, expit(t)[:, a] @ rho
    div = np.array([np.mean([distance[i,j] for i,j in combinations(row,2)])
                    if k > 1 else 0. for row in actions])
    return w[0]*fv+w[1]*ft+w[2]*div, fv, ft, div


def regret(j, weights):
    return float(weights @ j.max(axis=1) - (weights @ j).max())


def query_value(j, weights, likelihood, cost=0.):
    """Exact enumeration on FIXED particle support, actions and objective weights.

    The expected statewise oracle term cancels; the expression below is the
    reduction in Bayes regret, without unstable subtraction of two regrets.
    """
    if cost < 0: raise ValueError("Negative burden")
    if not np.allclose(likelihood.sum(axis=1), 1, atol=1e-10):
        raise ValueError("Incomplete outcome distribution")
    joint = weights[:, None]*likelihood
    gross = float(np.max(j.T @ joint, axis=0).sum() - (weights @ j).max())
    if gross < -1e-9: raise ArithmeticError("Inconsistent EVSI")
    return max(0., gross)-cost


def shortlist(posterior, phi, psi, chi, pairs, limit):
    v,t = posterior.utilities(phi, psi, chi)
    scores = [posterior.weights @ ((v[:,i]-v[:,j])*(t[:,i]-t[:,j]) < 0)
              for i,j in pairs]
    order = sorted(range(len(pairs)), key=lambda k: (-scores[k], pairs[k]))
    return [pairs[k] for k in order[:limit]]


def build_actions(phi, psi, chi, destinations, distance, feasible, k=5, count=64, seed=19):
    """Frozen bounded catalog-only search, shared across all methods.

    Random projections are seeded once and never depend on labels or posterior.
    Approximation quality is reported against exhaustive lists on tiny catalogs.
    """
    feasible = list(map(int, feasible))
    if not feasible: return []
    k = min(k, len({destinations[i] for i in feasible}))
    rng = np.random.default_rng(seed)
    features = np.column_stack((phi, psi, chi))
    actions = set()
    for trial in range(count*5):
        relevance = features @ rng.normal(size=features.shape[1])
        lam = (trial % 5)/4
        chosen = []
        for _ in range(k):
            available = [i for i in feasible if destinations[i] not in {destinations[j] for j in chosen}]
            best = max(available, key=lambda i: (relevance[i]+lam*sum(distance[i,j] for j in chosen), -i))
            chosen.append(best)
        actions.add(tuple(chosen))
        if len(actions) >= count: break
    return sorted(actions)
