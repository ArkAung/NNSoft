import random as rnd
import numpy as np
import csv


def bernoulli(p):
    prob = round(rnd.uniform(0, 1), 3)
    if prob < p:
        return 1
    else:
        return 0

def classifier(p1, p2):
    if p1 < 0.5 and p2 < 0.5:
        return 1
    elif p1 < 0.5 and p2 > 0.5:
        return 2
    elif p1 > 0.5 and p2 < 0.5:
        return 3
    elif p1 > 0.5 and p2 > 0.5:
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


def hardOneHot(hlbl):
    if hlbl == 1:
        return np.array([1, 0, 0, 0])
    elif hlbl == 2:
        return np.array([0, 1, 0, 0])
    elif hlbl == 3:
        return np.array([0, 0, 1, 0])
    elif hlbl == 4:
        return np.array([0, 0, 0, 1])


def a(p1, p2):
    return (1-p1) * (1-p2)

def b(p1, p2):
    return (1-p1) * p2

def c(p1, p2):
    return p1 * (1-p2)

def d(p1, p2):
    return p1 * p2

def ha(hlbl):
    if hlbl == 1:
        return 1
    else:
        return 0

def hb(hlbl):
    if hlbl == 2:
        return 1
    else:
        return 0

def hc(hlbl):
    if hlbl == 3:
        return 1
    else:
        return 0

def hd(hlbl):
    if hlbl == 4:
        return 1
    else:
        return 0


p1 = [round(rnd.uniform(0, 1), 3) for p in range(0, 100000)]
p2 = [round(rnd.uniform(0, 1), 3) for p in range(0, 100000)]
l1 = map(bernoulli, p1)
l2 = map(bernoulli, p2)
hard_classes = map(categorizeClass, l1, l2)
hard_one_hot = map(hardOneHot, hard_classes)
soft_a = map(a, p1, p2)
soft_b = map(b, p1, p2)
soft_c = map(c, p1, p2)
soft_d = map(d, p1, p2)

hard_a = map(ha, hard_classes)
hard_b = map(hb, hard_classes)
hard_c = map(hc, hard_classes)
hard_d = map(hd, hard_classes)
# soft_list = zip(soft_one_hot[:,0], soft_one_hot[:,1], soft_one_hot[:,2], soft_one_hot[:,3])

classes = map(classifier, p1, p2)

# with open('all_classes.csv', 'wb') as f:
#     writer = csv.writer(f)
#     # writer.writerows(zip(p1, p2, l1, l2, hard_classes, hard_one_hot, soft_one_hot))
#     writer.writerows(zip(p1, p2, soft_a, soft_b, soft_c, soft_d))
# p1,p2,l1,l2,h1,h2,h3,h4,a,b,c,d
with open('combined_classes2.csv', 'wb') as f:
    writer = csv.DictWriter(f, fieldnames=["p1", "p2", "l1", "l2", "hard_classes", "hard_a", "hard_b", "hard_c", "hard_d", "soft_a", "soft_b", "soft_c", "soft_d"], delimiter=',')
    writer.writeheader()
    writer = csv.writer(f, quoting = csv.QUOTE_NONE)
    writer.writerows(zip(p1, p2, l1, l2, hard_classes, hard_a, hard_b, hard_c, hard_d, soft_a, soft_b, soft_c, soft_d))
