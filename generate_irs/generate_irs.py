# %%
import numpy as np
import matplotlib.pyplot as plt
import rim

# %%
xs = [1, 1, 1]
# xr = np.random.uniform(0, 2, size=(6, 3))
xr = [
    [0.92781265, 1.13393288, 0.26736836],
    [1.43182171, 1.56376992, 1.59195482],
    [1.93766398, 0.46898242, 0.1320927],
    [0.47049847, 1.36370444, 0.56008512],
    [0.18023421, 0.99715069, 1.75508388],
    [0.33782243, 0.99060236, 1.48623386],
]
Nt = 256
L = [2, 2, 2]
beta = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
Fs = 8000
N = [20, 20, 20]
xr_dirs = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]]
xr_types = ["o", "o", "o", "o", "o", "o"]

h_gen, seed = rim.rim(
    xs,
    xr,
    Nt,
    L,
    beta,
    Fs,
    Rd=0.0,
    N=N,
    Tw=20,
    Fc=0.9,
    MicDirs=xr_dirs,
    MicTypes=xr_types,
)

# %%
L = 128
fig, axes = plt.subplots(h_gen.shape[1], 1, figsize=(8, 10))
h_gen /= np.max(h_gen)
h = np.zeros((L * 6,))
for i, row in enumerate(axes):
    h_ = h_gen[:, i]
    h_ /= np.max(h_)
    start = np.argmax(h_ == 1)
    h[i * L : (i + 1) * L] = h_[start : start + L]
    row.plot(h_[start : start + L])
    row.set_ylabel("Amplitude [1]")
    row.set_xlabel("Samples [1]")
    row.spines["right"].set_visible(False)
    row.spines["top"].set_visible(False)
    row.set_ylim((-1, 1))
    row.grid()
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 1, figsize=(8, 3))
for i in range(6):
    x = np.arange(i * L, (i + 1) * L)
    plt.plot(x, h[i * L : (i + 1) * L])
plt.xlim(-1, L * 6)
plt.grid()
plt.show()

# %%
with open("h.npy", "wb") as f:
    np.save(f, h)

# %%
