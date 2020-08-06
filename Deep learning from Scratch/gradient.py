import numpy as np 

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    print("grad.shape:", grad.shape) # (2,)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def function_2(x):
    return np.sum(x**2)


arr = np.array([3.0, 4.0])

print("x.shape :", arr.shape) # (2,)
print(numerical_gradient(function_2, arr)) # [6. 8.]