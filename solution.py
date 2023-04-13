import pandas as pd
import numpy as np
from scipy.stats import norm

chat_id = 776430833

def solution(sample1: np.ndarray, sample2: np.ndarray = None) -> bool:
    if sample2 is None:
        # one-sample t-test
        x = sample1
        n = len(x)
        se = np.std(x, ddof=1) / np.sqrt(n)
        t = np.mean(x) / se
        p_value = 2 * (1 - norm.cdf(abs(t)))
        return p_value < 0.1
    else:
        # two-sample t-test
        x = sample1
        y = sample2
        n1, n2 = len(x), len(y)
        s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
        se = np.sqrt(s1/n1 + s2/n2)
        t = (np.mean(x) - np.mean(y)) / se
        df = n1 + n2 - 2
        p_value = 2 * (1 - norm.cdf(abs(t)))
        return p_value < 0.1

