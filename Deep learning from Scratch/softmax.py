import numpy as np

def naive_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def modified_softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y

print(naive_softmax([1010, 1000, 990])) # [nan nan nan]
print(modified_softmax([0.3, 2.9, 4.0])) # [0.01821127 0.24519181 0.73659691]