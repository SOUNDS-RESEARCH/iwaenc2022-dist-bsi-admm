# %%
import utils
import numpy as np
import scipy.signal as signal
import admm_r1_td as admm
import simo_bsi_central as cent
import matplotlib.pyplot as plt

# %%
L = 64
N_f = 1024
N_sens = 3

# %%
fs = 16000
l = 120
N_s = 8000

# %%
with open("h.npy", "rb") as f:
    h = np.load(f)

# %%
h_f = np.zeros((N_sens, N_f), dtype=np.complex128)
fig, ax1 = plt.subplots()
ax1.set_title("Digital filter frequency response")
for n in range(N_sens):
    h_ = h[n * L : (n + 1) * L]
    w, hh = signal.freqz(h_, worN=N_f)
    h_f[n, :] = hh
    ax1.plot(w, 20 * np.log10(abs(hh)))
    # plt.legend()
ax1.set_ylabel("Amplitude [dB]")
ax1.set_xlabel("Frequency [rad/sample]")
ax1.grid()
plt.show()

# %%
plt.stem(h)
plt.title("filter taps of stacked imp resp")
plt.show()

# %%
network_td = admm.Network(L)
rho = 1
network_td.addNode("node1", rho)
network_td.addNode("node2", rho)
network_td.addNode("node3", rho)

# node 1 sends to node 2 and 3
network_td.setConnection("node1", ["node2"])
# network_td.setConnection("node2", ["node1"])
network_td.setConnection("node2", ["node3"])
network_td.setConnection("node3", ["node1"])
network_td.setBufferSize(1)

# %% GENERATE AR SIGNAL
N_ff = 1025
b1 = signal.firwin(N_ff, 0.25, pass_zero=False)
M = 10
Ah = np.zeros((M, N_ff - M - 1))
for l in range(M):
    Ah[l, :] = b1[M - l : M - l + N_ff - M - 1]
y = b1[M + 1 :]
a_ = np.linalg.solve(Ah @ Ah.T, Ah @ y)
b = np.concatenate([np.ones((1,)), -a_])

# %%
p0 = 1e-2
u = np.random.normal(size=(N_s, 1)) * np.sqrt(p0)
# %%
s = u / u.max()
# %%
# s = signal.lfilter([1], b, u, axis=0)

# %%
hopsize = L
s_ = np.concatenate([np.zeros(shape=(L - 1, 1)), s])
var_s = np.var(s)

x = np.zeros(shape=(N_s, N_sens))
SNR = 10
n_var = 10 ** (-SNR / 20) * var_s * np.linalg.norm(h) ** 2 / N_sens
H = np.reshape(h, (N_sens, L))
for k in range(N_s - L):
    x[k, :, None] = H @ s_[k : k + L][::-1] + n_var * np.random.normal(size=(N_sens, 1))

# %%
hopsize = L

# %% CENTRALIZED TIME DOMAIN QUASI NEWTON METHOD
print("CENTRALIZED TIME DOMAIN QUASI NEWTON METHOD")
err_MCQN = []
rho = 0.5  # step size
lambd = 1e-5  # regularization
eta = 0.98  # forgetting factor
buffer_size = L

x_vars = np.ones_like(np.var(x, 0))
bsi_mcqn = cent.MCQN(rho, lambd, eta, L, N_sens, x_vars, buffer_size)
for k in range(0, N_s - 2 * hopsize, hopsize):
    bsi_mcqn.step(x[k : k + L, :])
    err_MCQN.append(bsi_mcqn.get_error(h))
err_MCQN = np.asarray(err_MCQN)

# %% CENTRALIZED NORMALIZED FREQUENCY DOMAIN LMS METHOD
print("CENTRALIZED TIME DOMAIN QUASI NEWTON METHOD")
err_NMCFLMS = []
rho = 0.5  # step size
lambd = 0.98**L  # forgetting factor
sigma = 0.01  # regularization
bsi_nmcflms = cent.NMCFLMS(rho, lambd, sigma, L, N_sens)
for k_fq in range(0, N_s - 2 * L, L):
    bsi_nmcflms.step(x[k_fq : k_fq + 2 * L, :])
    err_NMCFLMS.append(bsi_nmcflms.get_error(h))
err_NMCFLMS = np.asarray(err_NMCFLMS)

# %% CENTRALIZED ROBUST NORMALIZED FREQUENCY DOMAIN LMS METHOD
print("CENTRALIZED TIME DOMAIN QUASI NEWTON METHOD")
err_RNMCFLMS = []
rho = 0.5  # step size
lambd = 0.98**L  # forgetting factor
sigma = 0.01  # regularization
eta = 0.5  # coupling factor
bsi_rnmcflms = cent.RNMCFLMS(rho, lambd, sigma, eta, L, N_sens)
for k_fq in range(0, N_s - 2 * L, L):
    bsi_rnmcflms.step(x[k_fq : k_fq + 2 * L, :])
    err_RNMCFLMS.append(bsi_rnmcflms.get_error(h))
err_RNMCFLMS = np.asarray(err_RNMCFLMS)

# %% CENTRALIZED l_p NORM CONSTRAINED ROBUST NORMALIZED FREQUENCY DOMAIN LMS METHOD
print("CENTRALIZED TIME DOMAIN QUASI NEWTON METHOD")
err_LPRNMCFLMS = []
rho = 0.5  # step size
lambd = 0.98**L  # forgetting factor
sigma = 0.01  # regularization
eta = 0.5  # coupling factor
p = 1.6  # p-norm
bsi_lprnmcflms = cent.LPRNMCFLMS(rho, lambd, sigma, eta, p, L, N_sens)
for k_fq in range(0, N_s - 2 * L, L):
    bsi_lprnmcflms.step(x[k_fq : k_fq + 2 * L, :])
    err_LPRNMCFLMS.append(bsi_lprnmcflms.get_error(h))
err_LPRNMCFLMS = np.asarray(err_LPRNMCFLMS)

# %% DISTRIBUTED NEWTON RANK-1 TIME DOMAIN METHOD
print("DISTRIBUTED NEWTON RANK-1 TIME DOMAIN METHOD")
rho_admm = 1  # penalty parameter / step size
mu = 0.5  # newton step size
eta = 0.95  # smoothing factor R
zeta = 0.95  # smoothing factor H
scaling = 1  # signal scaling
err_td = []
h_test = np.zeros(shape=h.shape)
network_td.setBufferSize(buffer_size)
network_td.reset()
network_td.setRho(rho_admm, mu, eta, zeta, scaling)
for k in range(0, N_s - 2 * hopsize, hopsize):
    network_td.step(x[k : k + L, :])
    e1 = h - (h.T @ network_td.z) / (network_td.z.T @ network_td.z) * network_td.z
    error = np.linalg.norm(e1) / np.linalg.norm(h)
    err_td.append(error)
err_td = np.asarray(err_td)

# %%
