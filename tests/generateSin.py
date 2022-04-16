from math import sin, pi
from random import random

N = 10000
print(N)
for i in range(N):
    x = int(random() * 1000) / 1000.
    y = int(random() * 1000) / 1000.
    z = int(y <= 1./2 + 1./2*sin(2.*x/(7.*pi)))
    print(x, y, z)
