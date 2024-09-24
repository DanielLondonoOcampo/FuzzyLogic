import numpy as np
import matplotlib.pyplot as plt

def trapmf(x, abcd):
    """
    Trapezoidal membership function generator.
    """
    a, b, c, d = abcd
    y = np.zeros_like(x)
    
    # Left slope
    mask = (a < x) & (x < b)
    if b != a:
        y[mask] = (x[mask] - a) / (b - a)
    elif b == a:
        y[x == a] = 1
    
    # Flat top
    y[(b <= x) & (x <= c)] = 1
    
    # Right slope
    mask = (c < x) & (x < d)
    if d != c:
        y[mask] = (d - x[mask]) / (d - c)
    elif d == c:
        y[x == d] = 1
    
    return y

def trimf(x, abc):
    """
    Triangular membership function generator.
    """
    a, b, c = abc
    y = np.zeros_like(x)
    
    # Left slope
    mask = (a < x) & (x <= b)
    if b != a:
        y[mask] = (x[mask] - a) / (b - a)
    elif b == a:
        y[x == a] = 1
    
    # Right slope
    mask = (b < x) & (x < c)
    if c != b:
        y[mask] = (c - x[mask]) / (c - b)
    elif c == b:
        y[x == c] = 1
    
    return y