import math

def linear(x):
    return x

def sigmoid(x):
    try:
        return (1.0 / (1.0 + math.exp(-x)))
    except OverflowError:
        return 0 if x < 0 else 1

def clamped(x):
    return max(-1, min(1, linear(x)))

def relu(x):
    return max(0, x)

def lelu(x):
    return 0.01 * x if x < 0 else x

def softplus(x):
    try:
        return math.log(1 + math.exp(x))
    except OverflowError:
        return 0 if x < 0 else 1

def step(x):
    return 0 if x < 0 else 1

def tanh(x):
    return 2 * sigmoid(2 * x) - 1