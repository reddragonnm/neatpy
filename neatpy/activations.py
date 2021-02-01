import math

def linear(x):
    """Linear activation function
    Returns the value given

    Args:
        x (float): input value

    Returns:
        float: output value
    """
    return x

def sigmoid(x):
    """Sigmoid activation function
    Logistic activation with a range of 0 to 1

    Args:
        x (float): input value

    Returns:
        float: output value
    """
    try:
        return (1.0 / (1.0 + math.exp(-x)))
    except OverflowError:
        return 0 if x < 0 else 1

def clamped(x):
    """Clamped activation function
    Clamps the input between -1 and 1

    Args:
        x (float): input value

    Returns:
        float: output value
    """
    return max(-1, min(1, linear(x)))

def relu(x):
    """ReLu activation function
    Limits the lower range of the input to 0

    Args:
        x (float): input value

    Returns:
        float: output value
    """
    return max(0, x)

def lelu(x):
    """LeLu activation function
    Basically ReLu but the lower range leaks slightly and has a range from -inf to inf

    Args:
        x (float): input value

    Returns:
        float: output value
    """
    return 0.01 * x if x < 0 else x

def softplus(x):
    """Softplus activation function

    Args:
        x (float): input value

    Returns:
        float: output value
    """
    try:
        return math.log(1 + math.exp(x))
    except OverflowError:
        return 0 if x < 0 else 1

def step(x):
    """Binary step activation function
    Returns 0 if less than 0 else returns 1

    Args:
        x (float): input value

    Returns:
        float: output value
    """
    return 0 if x < 0 else 1

def tanh(x):
    """Tanh activation function
    Logistic activation with a range of -1 to 1

    Args:
        x (float): input value

    Returns:
        float: output value
    """
    return 2 * sigmoid(2 * x) - 1