"""Planning sensitivity for a two-sided paired t test; not post-hoc power."""
import argparse
from scipy.stats import t, nct
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=36);p.add_argument('--effect',type=float,default=.5);p.add_argument('--alpha',type=float,default=.05);a=p.parse_args()
critical=t.ppf(1-a.alpha/2,a.n-1);nc=a.effect*a.n**.5
power=nct.cdf(-critical,a.n-1,nc)+nct.sf(critical,a.n-1,nc)
print({'n_participants':a.n,'standardized_paired_effect':a.effect,'alpha':a.alpha,'approximate_power':float(power),'note':'Sensitivity only; multiplicity-adjusted thresholds reduce power'})
