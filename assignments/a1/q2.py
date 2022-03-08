import numpy as np
# from math import gamma
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy import stats
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import beta

x = np.arange (0.01, 1, 0.01)
y = beta.pdf(x,1,1)
plt.plot(x,y, label="alpha=1, beta=1")
y = beta.pdf(x,5,5)
plt.plot(x,y, label="alpha=5, beta=5")
y = beta.pdf(x,10,10)
plt.plot(x,y, label="alpha=10, beta=10")
plt.title("Beta Distributions")
plt.legend()

from scipy.stats import beta
x = np.arange (0.01, 1, 0.01)
y = beta.pdf(x,1,2)
plt.plot(x,y, label="alpha=1, beta=2")
y = beta.pdf(x,5,5)
plt.plot(x,y, label="alpha=5, beta=5")
y = beta.pdf(x,10,11)
plt.plot(x,y, label="alpha=10, beta=11")
plt.title("Beta Distributions")
plt.legend()
