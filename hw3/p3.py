import numpy as np
import matplotlib.pyplot as plt

#########################
# PART A
#########################
plt.figure(figsize=(10,4))

phi = lambda z: np.exp(-z**2)
t = np.linspace(0, 1, 1000)

N = 10
plt.subplot(1, 2, 1)
for k in range(1, N+1):
    plt.plot(t, phi(N*t - k + .5))
plt.title(f"N = {N}")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("t")
plt.ylabel("$\phi_k$")

N = 25
plt.subplot(1, 2, 2)
for k in range(1, N+1):
    plt.plot(t, phi(N*t - k + .5))
plt.title(f"N = {N}")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("t")

plt.tight_layout()
plt.show()
plt.savefig('p2a.png', dpi=300)

#########################
# PART B
#########################
plt.figure(figsize=(5,4))
N=4
y = np.zeros(np.shape(t))
a = [1, -1, 1, -1]
for j in range(1, N+1):
    y += a[j-1] * phi(N*t - j + .5)
plt.plot(t, y, linewidth=3)
plt.xlabel("t")
plt.ylabel("y")
plt.xlim(0, 1)
plt.ylim(-.8, 0.8)
plt.tight_layout()
plt.show()
# plt.savefig('p2b.png', dpi=300)
# plt.close()

#########################
# PART C
#########################

x = lambda z: (z < 1/4) * 4 * z + (z >= 1/4) * (z < 1/2) * (-4 * z + 2) - (z >= 1/2) * np.sin(20*np.pi*z)

def fit_phi_basis(N):
    x_phik = lambda z, j: x(z) * phi(N*z - j + .5)

    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i, j] = np.trapz(phi(N*t - (i+1) + .5) * phi(N*t - (j+1) + .5), t)

    b = np.zeros((N, 1))
    for i in range(N):
        b[i] = np.trapz(x_phik(t, i+1), t)

    xhat_coeff = np.linalg.inv(A) @ b

    xhat = np.zeros(np.shape(t))
    for i, phi_k in enumerate(xhat_coeff):
        xhat += phi_k * phi(N*t - (i+1) + .5)
    
    return xhat

plt.figure(figsize=(10, 8))
for i, N in enumerate([5, 10, 20, 50]):
    plt.subplot(2, 2, i+1)
    plt.plot(t, x(t), '--')
    plt.plot(t, fit_phi_basis(N))
    plt.title(f"N = {N}")
plt.tight_layout()
plt.show()
# plt.savefig('p2c.png', dpi=300)