import numpy as np
# from math import gamma
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy import stats
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import beta

data = np.random.normal(0,2,size=30)
def gaussian(params):
    mean = params[0]
    sd = params[1]
    # Calculate negative log likelihood
    nll = -np.sum(stats.norm.logpdf(data, loc=mean, scale=sd))
    return nll


initParams = [1, 1]

results = minimize(gaussian, initParams, method='L-BFGS-B')
print(results.x)

def student(params):
    mean = params[0]
    sd = params[1]
    df = params[2]
    # Calculate negative log likelihood
    nll = -np.sum(stats.t.logpdf(data, df=df, loc=mean, scale=sd))
    return nll


initParams = [1, 1, 1]

results = minimize(student, initParams, method='L-BFGS-B')
print(results.x)

from scipy.stats import norm, t
x = np.arange (-10, 10, 0.01)
y = norm.pdf(x,0.32852566,2.0401025)
plt.plot(x,y, label="Gaussian")
y = t.pdf(x, loc=0.32932744, scale=2.01161381, df=70.2)
plt.plot(x,y, '--', label="Students' t")
plt.title("Compare Distributions")
plt.legend()

data = np.append(data,8)
data = np.append(data,9)
data = np.append(data,10)

def gaussian(params):
    mean = params[0]
    sd = params[1]
    # Calculate negative log likelihood
    nll = -np.sum(stats.norm.logpdf(data, loc=mean, scale=sd))
    return nll


initParams = [1, 1]

results = minimize(gaussian, initParams, method='L-BFGS-B')
print(results.x)

def student(params):
    mean = params[0]
    sd = params[1]
    df = params[2]
    # Calculate negative log likelihood
    nll = -np.sum(stats.t.logpdf(data, df=df, loc=mean, scale=sd))
    return nll


initParams = [1, 1, 1]

results = minimize(student, initParams, method='L-BFGS-B')
print(results.x)

from scipy.stats import norm, t
x = np.arange (-10, 10, 0.01)
y = norm.pdf(x,1.11684099, 3.17154201)
plt.plot(x,y, label="Gaussian")
y = t.pdf(x, loc=0.44950388, scale=1.52333538, df=1.70111095)
plt.plot(x,y, '--', label="Students' t")
plt.scatter(data,np.zeros(len(data)), label="Data")
plt.title("Compare Distributions")
plt.legend()
