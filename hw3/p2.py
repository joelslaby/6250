import numpy as np

def derivative_matrix(m):
    d = np.zeros(shape=(m, m+1))
    for i in range(m):
        d[i, i] = m - i 
    return d

xt = np.array([5.8, -9.2, 3.1, -6.2, 7.2, 1])

xpt = derivative_matrix(len(xt)-1) @ xt
print(xpt)

for t in [0.01, 0.45, 0.63, 0.81]:
    print(f"x\'({t}) = {np.polyval(xpt, t)}")