"""Conventional baselines, kept distinct from adaptations of published systems."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def map_bradley_terry(posterior,catalog,context):
    """Separate L2-regularized ordered-logistic fits; same observations/features.

    MAP point estimate, no transfer or particle uncertainty. All historical
    observations retain their original context; skips have no likelihood.
    """
    features=(np.column_stack((catalog.phi,catalog.psi)),
              np.column_stack((catalog.phi,catalog.context(context))))
    outputs=[]
    for target,current in zip(('V','T'),features):
        history=[o for o in posterior.history if o.target==target and o.answer!='skip']
        rows=[];labels=[]
        for o in history:
            f=(np.column_stack((catalog.phi,catalog.psi)) if target=='V' else
               np.column_stack((catalog.phi,catalog.context(o.context))))
            rows.append(f[o.i]-f[o.j]);labels.append(('A','B','tie').index(o.answer))
        if not rows: outputs.append(np.zeros((len(posterior.weights),len(current))));continue
        x=np.asarray(rows);y=np.asarray(labels)
        def loss(beta):
            delta=x@beta
            a=expit((delta-posterior.delta)/posterior.tau)
            b=expit((-delta-posterior.delta)/posterior.tau)
            probs=np.column_stack((a,b,np.maximum(0,1-a-b)))
            return -np.log(np.maximum(probs[np.arange(len(y)),y],1e-15)).sum()+.5*np.square(beta/posterior.sigma).sum()
        fit=minimize(loss,np.zeros(current.shape[1]),method='L-BFGS-B')
        if not fit.success and not np.isfinite(fit.fun):raise RuntimeError('MAP optimizer failed')
        outputs.append(np.broadcast_to(current@fit.x,(len(posterior.weights),len(current))).copy())
    return tuple(outputs)
