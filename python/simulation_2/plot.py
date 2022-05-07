# %%
import os
import numpy as np
import time
import utils

# import scipy.signal as signal
import matplotlib.pyplot as plt

# %% random connectivity
SNRs = [20]
N_senses = [4, 6, 8]
densities = [0.0, 0.25, 0.5, 0.75]

err_ADMM_newton_td_ = {}
err_ADMM_newton_fq_ = {}
err_ADMM_newton_fq_diag_ = {}
for i in range(len(densities)):
    err_ADMM_newton_td_[i] = {}
    err_ADMM_newton_fq_[i] = {}
    err_ADMM_newton_fq_diag_[i] = {}
    for N in N_senses:
        err_ADMM_newton_td_[i][N] = None
        err_ADMM_newton_fq_[i][N] = None
        err_ADMM_newton_fq_diag_[i][N] = None

dir = "data/"
files = sorted(os.listdir(dir))
for filename in files:
    print(filename)
    with open(dir + filename, "rb") as f:
        runs = np.load(f)
        run = np.load(f)
        L = np.load(f)
        N_sens = int(np.load(f))
        SNR = int(np.load(f))
        G = np.load(f)
        h = np.load(f)
        err_MCQN = np.load(f)
        err_NMCFLMS = np.load(f)
        err_RNMCFLMS = np.load(f)
        err_LPRNMCFLMS = np.load(f)
        err_ADMM_newton_td = np.load(f)
        err_ADMM_newton_fq = np.load(f)
        err_ADMM_newton_fq_diag = np.load(f)

        pid = int(filename.split("_")[0])
        if pid < 3:
            did = 0
            density = 0.0
        elif pid < 6:
            did = 1
            density = 0.25
        elif pid < 9:
            did = 2
            density = 0.5
        else:
            did = 3
            density = 0.75

        if not np.isnan(err_ADMM_newton_td).any():
            err_ADMM_newton_td_[did][N_sens] = (
                err_ADMM_newton_td
                if err_ADMM_newton_td_[did][N_sens] is None
                else np.vstack((err_ADMM_newton_td_[did][N_sens], err_ADMM_newton_td))
            )
        if not np.isnan(err_ADMM_newton_fq).any():
            err_ADMM_newton_fq_[did][N_sens] = (
                err_ADMM_newton_fq
                if err_ADMM_newton_fq_[did][N_sens] is None
                else np.vstack((err_ADMM_newton_fq_[did][N_sens], err_ADMM_newton_fq))
            )
        if not np.isnan(err_ADMM_newton_fq_diag).any():
            err_ADMM_newton_fq_diag_[did][N_sens] = (
                err_ADMM_newton_fq_diag
                if err_ADMM_newton_fq_diag_[did][N_sens] is None
                else np.vstack(
                    (err_ADMM_newton_fq_diag_[did][N_sens], err_ADMM_newton_fq_diag)
                )
            )

# %%
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
lim = 2500
id = 0
for N_sens in [4, 8]:
    for i in range(len(densities)):
        if err_ADMM_newton_fq_diag_[i][N_sens] is not None:
            axes[id].plot(
                20
                * np.log10(
                    np.nanmedian(err_ADMM_newton_fq_diag_[i][N_sens], axis=0)[:lim]
                ),
                label=r"$\zeta = $" + "%1.2f" % (densities[i]),
            )
    # axes[id].set_title(r"$M=$")
    axes[id].set_xlabel("Frame [1]")
    axes[id].set_ylabel("NPM [dB]")
    axes[id].set_xlim(0, lim)
    axes[id].set_ylim(-40, 0)
    axes[id].grid()
    axes[id].legend()
    id += 1
plt.tight_layout()
plt.show()

# %%
lim = 2500
for N_sens in [4, 8]:
    lines = ["-", "--", "-.", ":"]
    fig = plt.figure(figsize=(2.75, 2.5))
    for i in range(len(densities)):
        if err_ADMM_newton_fq_diag_[i][N_sens] is not None:
            plt.plot(
                20
                * np.log10(
                    np.nanmedian(err_ADMM_newton_fq_diag_[i][N_sens], axis=0)[:lim]
                ),
                lines.pop(),
                label=r"$\zeta = $" + "%1.2f" % (densities[i]),
            )
    # plt.title(r"$M=$")
    plt.xlabel("Frame [1]")
    plt.ylabel("NPM [dB]")
    plt.xlim(0, lim)
    plt.ylim(-40, 0)
    plt.grid()
    if N_sens == 8:
        plt.legend()
    plt.tight_layout()
    plt.show()
    utils.savefig(fig, "NPM_over_time_M%d" % (N_sens))

# %%
