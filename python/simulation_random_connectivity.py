# %%
import sys
import os.path
import utils
import numpy as np
import scipy.signal as signal
import admm_r1_td as admm
import admm_newton_fq as sb_newton_fq
import admm_newton_fq_diag as sb_newton_fq_diag
import simo_bsi_central as cent
import matplotlib.pyplot as plt

# %%
process_id = int(sys.argv[1])
SNR = int(sys.argv[2])
N_sens = int(sys.argv[3])
density = float(sys.argv[4])
print("Job: %d, SNR: %d, N_sens: %d, density: %f" % (process_id, SNR, N_sens, density))

# %%
runs = 30
seed = 12345
plot = False

# %%
L = 16
N_f = 1024

# %%
N_s = 16000

# %%
rng = np.random.RandomState()
rng.seed(seed)

# %%
for run in range(runs):
    if os.path.isfile(
        "data/simulation_random_connectivity/%d_%d.npy" % (process_id, run)
    ):
        print("run %d is already there" % (run))
        continue
    # %%
    h = np.array([]).reshape(0, 1)
    h_f = np.zeros((N_sens, N_f), dtype=np.complex128)
    for n in range(N_sens):
        h_ = rng.normal(size=(L, 1))
        h_ = h_ / np.linalg.norm(h_)
        w, hh = signal.freqz(h_, worN=N_f)
        h_f[n, :] = hh
        h = np.concatenate([h, h_])

    # %%
    if plot:
        fig, ax1 = plt.subplots()
        ax1.set_title("Digital filter frequency response")
        for n in range(N_sens):
            ax1.plot(w, 20 * np.log10(abs(h_f[n, :])))
        ax1.set_ylabel("Amplitude [dB]")
        ax1.set_xlabel("Frequency [rad/sample]")
        ax1.grid()
        plt.show()

    # %%
    if plot:
        plt.stem(h)
        plt.title("filter taps of stacked imp resp")
        plt.show()

    # %%
    # connectivity: ring + random inbetween ones
    G = utils.generateRandomConnectionMatrixWithRing(N_sens, density, rng)

    network_td = admm.Network(L)
    network_newton_fq = sb_newton_fq.Network(L)
    network_newton_fq_diag = sb_newton_fq_diag.Network(L)

    rho = 1
    for n in range(N_sens):
        name = "node%d" % (n)
        network_td.addNode(name, rho)
        network_newton_fq.addNode(name, rho)
        network_newton_fq_diag.addNode(name, rho)
    for n in range(N_sens):
        name = "node%d" % (n)
        connections = []
        for j in range(N_sens):
            name1 = "node%d" % (j)
            if G[n, j] > 0:
                connections.append(name1)
        network_td.setConnection(name, connections)
        network_newton_fq.setConnection(name, connections)
        network_newton_fq_diag.setConnection(name, connections)

    # %%

    # %% GENERATE AR SIGNAL
    N_ff = 1025
    b1 = signal.firwin(N_ff, 0.25, pass_zero=False)
    M = 20
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
    # SNR = 20
    n_var = 10 ** (-SNR / 20) * var_s * np.linalg.norm(h) ** 2 / N_sens
    H = np.reshape(h, (N_sens, L))
    for k in range(N_s - L):
        x[k, :, None] = H @ s_[k : k + L][::-1] + n_var * rng.normal(size=(N_sens, 1))

    # %%
    hopsize = 1

    # %% CENTRALIZED TIME DOMAIN QUASI NEWTON METHOD
    print("CENTRALIZED TIME DOMAIN QUASI NEWTON METHOD")
    err_MCQN = []
    rho = 0.5  # step size
    lambd = 1e-5  # regularization
    eta = 0.98  # forgetting factor
    buffer_size = L

    x_vars = np.ones_like(np.var(x, 0))
    bsi_mcqn = cent.MCQN(rho, lambd, eta, L, N_sens, x_vars, buffer_size)
    for k in range(0, N_s - 2 * L, hopsize):
        bsi_mcqn.step(x[k : k + L, :])
        err_MCQN.append(bsi_mcqn.get_error(h))
    err_MCQN = np.asarray(err_MCQN)

    # %% CENTRALIZED NORMALIZED FREQUENCY DOMAIN LMS METHOD
    print("CENTRALIZED NORMALIZED FREQUENCY DOMAIN LMS METHOD")
    err_NMCFLMS = []
    rho = 0.4  # step size
    lambd = 0.98  # forgetting factor
    sigma = 0.01  # regularization
    bsi_nmcflms = cent.NMCFLMS(rho, lambd, sigma, L, N_sens)
    for k_fq in range(0, N_s - 2 * L, hopsize):
        bsi_nmcflms.step(x[k_fq : k_fq + 2 * L, :])
        err_NMCFLMS.append(bsi_nmcflms.get_error(h))
    err_NMCFLMS = np.asarray(err_NMCFLMS)

    # %% CENTRALIZED ROBUST NORMALIZED FREQUENCY DOMAIN LMS METHOD
    print("CENTRALIZED ROBUST NORMALIZED FREQUENCY DOMAIN LMS METHOD")
    err_RNMCFLMS = []
    rho = 0.25  # step size
    lambd = 0.98  # forgetting factor
    sigma = 0.01  # regularization
    eta = 0.4  # coupling factor
    bsi_rnmcflms = cent.RNMCFLMS(rho, lambd, sigma, eta, L, N_sens)
    for k_fq in range(0, N_s - 2 * L, hopsize):
        bsi_rnmcflms.step(x[k_fq : k_fq + 2 * L, :])
        err_RNMCFLMS.append(bsi_rnmcflms.get_error(h))
    err_RNMCFLMS = np.asarray(err_RNMCFLMS)

    # %% CENTRALIZED l_p NORM CONSTRAINED ROBUST NORMALIZED FREQUENCY DOMAIN LMS METHOD
    print(
        "CENTRALIZED l_p NORM CONSTRAINED ROBUST NORMALIZED FREQUENCY DOMAIN LMS METHOD"
    )
    err_LPRNMCFLMS = []
    rho = 0.3  # step size
    lambd = 0.98  # forgetting factor
    sigma = 0.01  # regularization
    eta = 0.4  # coupling factor
    p = 1.6  # p-norm
    bsi_lprnmcflms = cent.LPRNMCFLMS(rho, lambd, sigma, eta, p, L, N_sens)
    for k_fq in range(0, N_s - 2 * L, hopsize):
        bsi_lprnmcflms.step(x[k_fq : k_fq + 2 * L, :])
        err_LPRNMCFLMS.append(bsi_lprnmcflms.get_error(h))
    err_LPRNMCFLMS = np.asarray(err_LPRNMCFLMS)

    # %% DISTRIBUTED NEWTON RANK-1 TIME DOMAIN METHOD
    print("DISTRIBUTED NEWTON RANK-1 TIME DOMAIN METHOD")
    rho_admm = 0.05  # penalty parameter / step size
    mu = 0.05  # newton step size
    eta = 0.98  # smoothing factor R
    zeta = 0.98  # smoothing factor H
    scaling = 1  # signal scaling
    err_td = []
    h_test = np.zeros(shape=h.shape)
    network_td.setBufferSize(buffer_size)
    network_td.reset()
    network_td.setRho(rho_admm, mu, eta, zeta, scaling)
    for k in range(0, N_s - 2 * L, hopsize):
        network_td.step(x[k : k + L, :])
        e1 = h - (h.T @ network_td.z) / (network_td.z.T @ network_td.z) * network_td.z
        error = np.linalg.norm(e1) / np.linalg.norm(h)
        err_td.append(error)
    err_td = np.asarray(err_td)

    # %% DISTRIBUTED NEWTON FREQUENCY DOMAIN METHOD
    print("DISTRIBUTED NEWTON FREQUENCY DOMAIN METHOD")
    rho_admm = 0.1
    mu = 0.2
    eta = 0.0
    err_ADMM_newton_fq = []
    h_test_newton_fq = np.zeros(shape=h.shape)
    network_newton_fq.setBufferSize(1)
    network_newton_fq.reset()
    network_newton_fq.setRho(rho_admm, mu, eta)
    for k_admm_fq in range(0, N_s - 2 * L, hopsize):
        network_newton_fq.step(x[k_admm_fq : k_admm_fq + 2 * L, :])
        for n in range(N_sens):
            h_test_newton_fq[(n * L) : (n + 1) * L] = np.real(
                np.fft.ifft(network_newton_fq.z[(n * L) : (n + 1) * L], axis=0)
            )
        e1 = (
            h
            - (h.T @ h_test_newton_fq)
            / (h_test_newton_fq.T @ h_test_newton_fq)
            * h_test_newton_fq
        )
        error = np.linalg.norm(e1) / np.linalg.norm(h)
        err_ADMM_newton_fq.append(error)
    err_ADMM_newton_fq = np.asarray(err_ADMM_newton_fq)

    # %% DISTRIBUTED DIAGONALIZED NEWTON FREQUENCY DOMAIN METHOD
    print("DISTRIBUTED DIAGONALIZED NEWTON FREQUENCY DOMAIN METHOD")
    rho_admm = 1
    mu = 0.5
    eta = 0.98
    err_ADMM_newton_fq_diag = []
    h_test_newton_fq_diag = np.zeros(shape=h.shape)
    network_newton_fq_diag.setBufferSize(1)
    network_newton_fq_diag.reset()
    network_newton_fq_diag.setRho(rho_admm, mu, eta)
    for k_admm_fq in range(0, N_s - 2 * L, hopsize):
        network_newton_fq_diag.step(x[k_admm_fq : k_admm_fq + 2 * L, :])
        for n in range(N_sens):
            h_test_newton_fq_diag[(n * L) : (n + 1) * L] = np.real(
                np.fft.ifft(network_newton_fq_diag.z[(n * L) : (n + 1) * L], axis=0)
            )
        e1 = (
            h
            - (h.T @ h_test_newton_fq_diag)
            / (h_test_newton_fq_diag.T @ h_test_newton_fq_diag)
            * h_test_newton_fq_diag
        )
        error = np.linalg.norm(e1) / np.linalg.norm(h)
        err_ADMM_newton_fq_diag.append(error)
    err_ADMM_newton_fq_diag = np.asarray(err_ADMM_newton_fq_diag)

    # %%
    with open(
        "data/simulation_random_connectivity/%d_%d.npy" % (process_id, run), "wb"
    ) as f:
        np.save(f, runs)
        np.save(f, run)
        np.save(f, L)
        np.save(f, N_sens)
        np.save(f, SNR)
        np.save(f, G)
        np.save(f, h)
        np.save(f, err_MCQN)
        np.save(f, err_NMCFLMS)
        np.save(f, err_RNMCFLMS)
        np.save(f, err_LPRNMCFLMS)
        np.save(f, err_td)
        np.save(f, err_ADMM_newton_fq)
        np.save(f, err_ADMM_newton_fq_diag)

# %%
if plot:
    fig = plt.figure(figsize=(8, 6))
    plt.plot(20 * np.log10(err_MCQN), label="MCQN")
    plt.plot(20 * np.log10(err_NMCFLMS), label="NMCFLMS")
    plt.plot(20 * np.log10(err_RNMCFLMS), label="RNMCFLMS")
    plt.plot(20 * np.log10(err_LPRNMCFLMS), label="LPRNMCFLMS")
    plt.plot(20 * np.log10(err_td), label="ADMM")
    plt.plot(20 * np.log10(err_ADMM_newton_fq), label="ADMM FQ")
    plt.plot(20 * np.log10(err_ADMM_newton_fq_diag), label="ADMM FQ DIAG")
    plt.plot(np.ones_like(err_MCQN) * (-SNR), "--k", label="Noise floor")
    plt.title("Misalignment")
    plt.xlabel("Frame [1]")
    plt.ylabel("Normalized Misalignment NPM [dB]")
    plt.grid()
    plt.legend()
    plt.show()
# %%
