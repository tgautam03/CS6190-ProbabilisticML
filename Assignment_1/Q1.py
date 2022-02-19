import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy import stats
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def students_t(x, mu, prec, df):
    return gamma(df/2+1/2)/gamma(df/2) * np.sqrt(prec/np.pi/df) * (1+prec/df*(x-mu)**2)**(-df/2-1/2)

def normal(x, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((x-mu)**2)/2/(sigma**2))

x = np.arange(-10, 10, 0.001)
# Student t
dfs = [0.1, 1, 10, 100, 1000000]
ys = []
for df in dfs:
    y = students_t(x, 0, 1, df)
    ys.append(y)

# Gaussian
y = normal(x, 0, 1)

# Plotting
for (i, df) in enumerate(dfs):
    plt.plot(x, ys[i], label="df={}".format(df))
plt.plot(x, y, label="Gaussian")
plt.legend()
plt.title("Student's t distribution")
