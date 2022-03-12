import pandas as pd
import numpy as np
import scipy.stats
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm

# Loading Training data
df = pd.read_csv("bank-note/bank-note/train.csv", header=None)
d = df.to_numpy()
X = d[:,:-1]
Y = d[:,-1]

# Loading Test data
df = pd.read_csv("bank-note/bank-note/test.csv", header=None)
d = df.to_numpy()
Xtest = d[:,:-1]
Ytest = d[:,-1]

##########################################################################################
###################################### PART A ############################################
##########################################################################################
def initialise_w(initialise):
    if(initialise == 'random'):
        w = np.random.randn(d,1)
        print("w is initialised from N[0,1]")
    elif(initialise == 'zeros'):
        w = np.zeros((d,1))
        print("w is initialised as a zero vector")
    else:
        print("Method unknown")
    return w

def compute_mu(X, w):
    mu = expit(np.dot(X,w))
    mu = mu.reshape(X.shape[0],1)
    return mu

def first_derivative(w):
    mu = compute_mu(X, w)
    epsilon = 1e-12

    grad = np.matmul(np.transpose(X), (mu-Y)) + w.reshape(d,1)
    grad = grad.squeeze()
    return(grad)

def second_deivative(w,X,y):
    mu = compute_mu(X, w)
    R = np.eye(n)
    for i in range(n):
        R[i,i] = mu[i,0] * (1-mu[i,0])
    return(np.dot(np.dot(np.transpose(X),R),X) + np.eye(d))

def test(w, X, y):
    n,d = X.shape
    mu = compute_mu(X, w)
    yhat = np.zeros((n,1)).astype(np.float64)
    yhat[mu>0.5]=1
    correct = np.sum(yhat==y)
    return(correct,n)

def train(initialise):

    np.random.seed(0)
    w = initialise_w(initialise)
    for j in range(100):

        grad1 = first_derivative(w.squeeze()).reshape(d,1)
        H = second_deivative(w, X, Y)
        delta_w = np.dot(np.linalg.inv(H),grad1)
        w = w - delta_w
        diff = np.linalg.norm(delta_w)

        correct,n = test(w, Xtest, Ytest)
        print("Iteration : {} \t Accuracy : {}%".format(j,correct/n*100))

        if(diff < 1e-5):
            print("tolerance reached at the iteration : ",j)
            break
    print("Training done...")
    print("Model weights : ", np.transpose(w))

n,d = X.shape
n1,d1 = Xtest.shape

Y = Y.reshape(n,1)
Ytest = Ytest.reshape(n1,1)

train('zeros')
train('random')


##########################################################################################
###################################### PART B ############################################
##########################################################################################
def compute_mu(X, w):
    phi=np.dot(X,w)
    mu = norm.cdf(phi)
    mu = mu.reshape(X.shape[0],1)
    return mu

def first_derivative(w):
    mu = compute_mu(X, w)
    epsilon = 1e-12
    phi=np.dot(X,w)
    grad_mu = X*(scipy.stats.norm.pdf(phi,0,1).reshape(-1,1))
    return(np.sum((- Y*(1/(mu)) + (1-Y)*(1/(1+epsilon-mu)))*grad_mu,0) + w).squeeze()

def second_deivative(w,X,y):
    mu = compute_mu(X, w)
    R = np.eye(n)

    phi=np.dot(X,w)
    for i in range(n):
        t1 = (y[i] - mu[i,0])/(mu[i,0] * (1-mu[i,0]))
        t2 = scipy.stats.norm.pdf(phi[i,0],0,1)
        t3 = (1-y[i])/np.power(1-mu[i,0],2) + y[i]/np.power(mu[i,0],2)
        R[i,i] = t1*t2*np.dot(X[i],w) + t3*t2*t2

    return(np.dot(np.dot(np.transpose(X),R),X) + np.eye(d))

def neg_log_posterior(w):
    w=w.reshape(-1,1)
    epsilon = 1e-12
    mu = compute_mu(X, w)
    prob_1 = Y*np.log(mu+epsilon)
    prob_0 = (1-Y)*np.log(1-mu+epsilon)
    log_like = np.sum(prob_1) + np.sum(prob_0)
    w_norm = np.power(np.linalg.norm(w),2)
    neg_log_pos = -log_like+w_norm/2
    print("neg_log_posterior = {:.4f} \tlog_like = {:.4f} \tw_norm = {:.4f}".format(neg_log_pos, log_like, w_norm))
    return(neg_log_pos)

def test(w, X, y):
    n,d = X.shape
    mu = compute_mu(X, w)
    #print(mu.shape, n, d)
    yhat = np.zeros((n,1)).astype(np.float64)
    yhat[mu>0.5]=1
    correct = np.sum(yhat==y)
    return(correct,n)

res = minimize(neg_log_posterior, initialise_w('zeros'), method='BFGS', jac=first_derivative,
               tol= 1e-5, options={'maxiter': 100})
correct,n = test(res.x, Xtest, Ytest)
print("\n_____________Model trained______________\n")
print("\nModel weights : ", res.x)
print("\n_____________Test Accuracy______________\n")

print("Accuracy : {}% ".format(correct/n*100))

res = minimize(neg_log_posterior, initialise_w('random'), method='BFGS', jac=first_derivative,
               tol= 1e-5, options={'maxiter': 100})
correct,n = test(res.x, Xtest, Ytest)
print("\n_____________Model trained______________\n")
print("\nModel weights : ", res.x)
print("\n_____________Test Accuracy______________\n")

print("Accuracy : {}% ".format(correct/n*100))


##########################################################################################
###################################### PART C ############################################
##########################################################################################
def compute_mu(X, w):
    phi=np.dot(X,w)
    mu = norm.cdf(phi)
    mu = mu.reshape(X.shape[0],1)
    return mu

def first_derivative(w):
    mu = compute_mu(X, w)
    epsilon = 1e-12
    phi=np.dot(X,w)
    grad_mu = X*(scipy.stats.norm.pdf(phi,0,1).reshape(-1,1))
    return(np.sum((- Y*(1/(mu)) + (1-Y)*(1/(1+epsilon-mu)))*grad_mu,0) + w).squeeze()

def second_deivative(w,X,y):
    mu = compute_mu(X, w)
    R = np.eye(n)

    phi=np.dot(X,w)
    for i in range(n):
        t1 = (y[i] - mu[i,0])/(mu[i,0] * (1-mu[i,0]))
        t2 = scipy.stats.norm.pdf(phi[i,0],0,1)
        t3 = (1-y[i])/np.power(1-mu[i,0],2) + y[i]/np.power(mu[i,0],2)
        R[i,i] = t1*t2*np.dot(X[i],w) + t3*t2*t2

    return(np.dot(np.dot(np.transpose(X),R),X) + np.eye(d))

def neg_log_posterior(w):
    w=w.reshape(-1,1)
    epsilon = 1e-12
    mu = compute_mu(X, w)
    prob_1 = Y*np.log(mu+epsilon)
    prob_0 = (1-Y)*np.log(1-mu+epsilon)
    log_like = np.sum(prob_1) + np.sum(prob_0)
    w_norm = np.power(np.linalg.norm(w),2)
    neg_log_pos = -log_like+w_norm/2
    print("neg_log_posterior = {:.4f} \tlog_like = {:.4f} \tw_norm = {:.4f}".format(neg_log_pos, log_like, w_norm))
    return(neg_log_pos)

def test(w, X, y):
    n,d = X.shape
    mu = compute_mu(X, w)
    #print(mu.shape, n, d)
    yhat = np.zeros((n,1)).astype(np.float64)
    yhat[mu>0.5]=1
    correct = np.sum(yhat==y)
    return(correct,n)

def train(initialise):

    np.random.seed(0)
    w = initialise_w(initialise)
    for j in range(100):

        grad1 = first_derivative(w.squeeze()).reshape(d,1)
        H = second_deivative(w, X, Y)
        delta_w = np.dot(np.linalg.inv(H),grad1)
        w = w - delta_w
        diff = np.linalg.norm(delta_w)

        correct,n = test(w, Xtest, Ytest)
        print("Iteration : {} \t Accuracy : {}%".format(j,correct/n*100))

        if(diff < 1e-5):
            print("tolerance reached at the iteration : ",j)
            break
    print("Training done...")
    print("Model weights : ", np.transpose(w))

n,d = X.shape
n1,d1 = Xtest.shape

Y = Y.reshape(n,1)
Ytest = Ytest.reshape(n1,1)

train('zeros')
train('random')
