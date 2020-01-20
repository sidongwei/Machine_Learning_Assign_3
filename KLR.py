import numpy as np
import csv
#import matplotlib.pyplot as plt

def linear_kernel (X, Y):       #make all kernel accept matrix input
    return np.dot(X,Y.T)
def poly_kernel (X, Y):
    return (np.dot(X,Y.T)+1)**5
def Gaussian_kernel (X, Y, sigma=2e7):      #tricky here to expand L-2 norm and use meshgrid
    X2 = np.linalg.norm(X,axis=1)**2
    Y2 = np.linalg.norm(Y,axis=1)**2
    YV,XV = np.meshgrid(Y2,X2)
    return np.exp(-(XV-2*np.dot(X,Y.T)+YV)/sigma)

def partial(a, kernel, y, s, lam):      #compute partial derivative on a_k with set s input
    pd = 2*lam*np.dot(a, kernel)
    for elm in s:       #must use for loop cuz different denominator for different data
        ker = kernel[:,elm]
        e = float(np.exp(-y[elm]*np.dot(a,ker)))
        pd += -y.T*ker*e/(1+e)
    return pd

def error(a, y, kernel, lam):   #compute error/loss function value
    n = len(y)
    err = 0
    for i in range(n):
        power = float(np.dot(a,kernel[:,i]))
        err += np.log(1+np.exp(-y[i]*power))+lam*a[0,i]*power
    return err/n

def SGD (y, kernel, lam):       #stochastic gradient descent
    n = len(y)
    a = np.zeros((1,n))
    eta = 4e-4
    epoch = 3       #times going throung the entire data
    iteration = 100         #batch_size = training set/ iteration
    while (epoch > 0):
        perm = np.random.permutation(n)     #randomly permute the data
        batch = np.array_split(perm, iteration)
        for i in range(iteration):
            a = a-eta*partial(a, kernel, y, batch[i], lam)
            print(error(a, y, kernel, lam))
        epoch -= 1
    return a


def predict (a,ker_test):       #predict on test set
    y = np.dot(a,ker_test)
    y = np.sign(y)
    return y

def kernel_logistic_regression (X, y, X_test, y_test, lam):  #compare different kernel
    kernel = {0:linear_kernel, 1:poly_kernel, 2:Gaussian_kernel}
    lam_num = len(lam)
    accuracy = np.zeros((3,lam_num))
    n = len(X)
    m = len(X_test)
    for i in range(3):
        Ker = kernel[i](X,X)     #compute kernel matrix only once to save time
        Ker_test = kernel[i](X,X_test)
        max = float(np.amax(Ker))
        Ker = Ker/max      #normalize the kernel matrix
        Ker_test = Ker_test/max
        for j in range(lam_num):    #iterate by lambda
            a = SGD(y,Ker,lam[j])
            y_pred = predict(a, Ker_test).reshape((m,1))
            acc = 0.0       #number of correct prediction
            for k in range(n):
                if y_pred[k]==y_test[k]:
                    acc += 1
            accuracy[i,j] = acc/n
    return accuracy

def compare_lambda(X, y, X_test, y_test):       #change the range of lambda
    lam_min = 0.2
    lam_leap = 0.2
    lam_max = 3
    lam_num = 15        #make sure lam_num=(lam_max-lam_min)/lam_leap
    lam = np.arange(lam_min,lam_max+lam_leap,lam_leap)
    accuracy= kernel_logistic_regression(X, y, X_test, y_test,lam)
    for i in range(lam_num):        #avoid loss of precision
        lam[i] = '%2f'%lam[i]
    header = ["lambda", "linear kernel", "poly kernel", "Gaussian kernel"]
    DATA = np.concatenate((lam.reshape(lam_num,1),accuracy.T),axis=1)
    CSV = open("result_lambda.csv","w")        #output result
    writer = csv.writer(CSV)
    writer.writerow(header)
    for row in DATA:
        writer.writerow(row)
    CSV.close()

def compare_sigma(X, y, X_test, y_test):        #compare against sigma
    lam = 1
    sig_min = 1e7
    sig_leap = 1e7
    sig_max = 1e8
    sig_num = 10        #make sure sig_num=(sig_max-sig_min)/sig_leap
    sigma = np.arange(sig_min,sig_max+sig_leap,sig_leap)    #values for sigma
    n = len(X)
    m = len(X_test)
    accuracy = np.zeros(sig_num)
    for i in range(sig_num):    #iterate by sigma
        Ker = Gaussian_kernel(X,X,sigma[i])     #compute kernel matrix only once to save time
        Ker = Ker/float(np.amax(Ker))      #scale the kernel matrix to [0,1]
        Ker_test = Gaussian_kernel(X,X_test,sigma[i])
        Ker_test = Ker_test/float(np.amax(Ker_test))
        a = SGD(y, Ker, lam)
        y_pred = predict(a, Ker_test).reshape((m, 1))
        acc = 0.0   #number of correct prediction
        for k in range(n):
            if y_pred[k] == y_test[k]:
                acc += 1
        accuracy[i] = acc / n
    header = ["sigma", "Gaussian kernel accuracy"]
    DATA = np.concatenate((sigma.reshape(sig_num,1),accuracy.reshape(sig_num,1)),axis=1)
    CSV = open("result_sigma.csv","w")        #output result
    writer = csv.writer(CSV)
    writer.writerow(header)
    for row in DATA:
        writer.writerow(row)
    CSV.close()

if __name__ == "__main__":
    X = np.genfromtxt("train_X_dog_cat.csv", delimiter = ',')
    y = np.genfromtxt("train_y_dog_cat.csv", delimiter = ',')
    X_test = np.genfromtxt("test_X_dog_cat.csv", delimiter = ',')
    y_test = np.genfromtxt("test_y_dog_cat.csv", delimiter = ',')
    compare_lambda(X, y, X_test, y_test)    #compare lambda and output "result_lambda.csv"
    compare_sigma(X, y, X_test, y_test)     #compare sigma and output "result_sigma.csv"