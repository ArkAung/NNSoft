import random as rnd
import numpy as np
import csv

def beta_noise(h1, h2):
    b1 = 0.01 * np.random.beta(0.5, 10, size=np.array(h1).size)
    b2 = 0.01 * np.random.beta(0.1, 20, size=np.array(h2).size)
    return b1, b2

def bernoulli(p):
    prob = round(rnd.uniform(0, 1), 3)
    if prob < p:
        return 1
    else:
        return 0

def classifier(h1, h2):
    if h1 < 0.5 and h2 < 0.5:
        return 1
    elif h1 < 0.5 and h2 > 0.5:
        return 2
    elif h1 > 0.5 and h2 < 0.5:
        return 3
    elif h1 > 0.5 and h2 > 0.5:
        return 4

def categorizeClass(h1, h2):
    """Greeble classification with respect to horns [0 down, 1 up]
        h1  |h2     |class
        0   |0      |1
        0   |1      |2
        1   |0      |3
        1   |1      |4"""
    str_class = ""
    if h1 == 0 and h2 == 0:
        str_class = 0
    elif h1 == 0 and h2 == 1:
        str_class = 1
    elif h1 == 1 and h2 == 0:
        str_class = 2
    elif h1 == 1 and h2 == 1:
        str_class = 3
    return str_class

def a(h1, h2):
    return (1 - h1) * (1 - h2)

def b(h1, h2):
    return (1 - h1) * h2

def c(h1, h2):
    return h1 * (1 - h2)

def d(h1, h2):
    return h1 * h2

p1 = [round(rnd.uniform(0, 1), 3) for p in range(0, 10000)]
p2 = [round(rnd.uniform(0, 1), 3) for p in range(0, 10000)]
l1 = map(bernoulli, p1)
l2 = map(bernoulli, p2)
hard_classes = map(categorizeClass, l1, l2)
soft_a = map(a, p1, p2)
soft_b = map(b, p1, p2)
soft_c = map(c, p1, p2)
soft_d = map(d, p1, p2)
soft_classes = map(classifier, p1, p2)
e1, e2 = beta_noise(p1, p2)
p1 = map(lambda x,y: x+y, np.array(p1), e1)
p2 = map(lambda x,y: x+y, np.array(p2), e2)

with open('beta_classes.csv', 'wb') as f:
    writer = csv.DictWriter(f, fieldnames=["p1", "p2", "l1", "l2", "hard_classes", "soft_a", "soft_b", "soft_c", "soft_d"], delimiter=',')
    writer.writeheader()
    writer = csv.writer(f, quoting = csv.QUOTE_NONE)
    writer.writerows(zip(p1, p2, l1, l2, hard_classes, soft_a, soft_b, soft_c, soft_d))
