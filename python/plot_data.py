# %%
import os
import numpy as np
import time
import utils

# import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib import rc

# %%
class SimulationData:
    variable: str
    variable_range: list


# %% random IRS
SNRs = [5, 10, 15, 20, 30, 40, 50]
id = 0
runs_ = []
run_ = []
L_ = []
N_sens_ = []
SNR_ = []
h_ = []
err_MCQN_ = {}
err_NMCFLMS_ = {}
err_RNMCFLMS_ = {}
err_LPRNMCFLMS_ = {}
err_ADMM_newton_td_ = {}
err_ADMM_newton_fq_ = {}
err_ADMM_newton_fq_diag_ = {}
for SNR in SNRs:
    err_MCQN_[SNR] = None
    err_NMCFLMS_[SNR] = None
    err_RNMCFLMS_[SNR] = None
    err_LPRNMCFLMS_[SNR] = None
    err_ADMM_newton_td_[SNR] = None
    err_ADMM_newton_fq_[SNR] = None
    err_ADMM_newton_fq_diag_[SNR] = None

L_p = 64
N_p = 5

dir = "data/simulation_random/"
files = (file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file)))
for filename in files:
    print(filename)
    with open(dir + filename, "rb") as f:
        runs = np.load(f)
        run = np.load(f)
        L = np.load(f)
        N_sens = np.load(f)
        SNR = int(np.load(f))
        h = np.load(f)
        err_MCQN = np.load(f)
        err_NMCFLMS = np.load(f)
        err_RNMCFLMS = np.load(f)
        err_LPRNMCFLMS = np.load(f)
        err_ADMM_newton_td = np.load(f)
        err_ADMM_newton_fq = np.load(f)
        err_ADMM_newton_fq_diag = np.load(f)

        if L != L_p or N_sens != N_p:
            continue

        if not np.isnan(err_MCQN).any():
            err_MCQN_[SNR] = (
                err_MCQN
                if err_MCQN_[SNR] is None
                else np.vstack((err_MCQN_[SNR], err_MCQN))
            )
        if not np.isnan(err_NMCFLMS).any():
            err_NMCFLMS_[SNR] = (
                err_NMCFLMS
                if err_NMCFLMS_[SNR] is None
                else np.vstack((err_NMCFLMS_[SNR], err_NMCFLMS))
            )
        if not np.isnan(err_RNMCFLMS).any():
            err_RNMCFLMS_[SNR] = (
                err_RNMCFLMS
                if err_RNMCFLMS_[SNR] is None
                else np.vstack((err_RNMCFLMS_[SNR], err_RNMCFLMS))
            )
        if not np.isnan(err_LPRNMCFLMS).any():
            err_LPRNMCFLMS_[SNR] = (
                err_LPRNMCFLMS
                if err_LPRNMCFLMS_[SNR] is None
                else np.vstack((err_LPRNMCFLMS_[SNR], err_LPRNMCFLMS))
            )
        if not np.isnan(err_ADMM_newton_td).any():
            err_ADMM_newton_td_[SNR] = (
                err_ADMM_newton_td
                if err_ADMM_newton_td_[SNR] is None
                else np.vstack((err_ADMM_newton_td_[SNR], err_ADMM_newton_td))
            )
        if not np.isnan(err_ADMM_newton_fq).any():
            err_ADMM_newton_fq_[SNR] = (
                err_ADMM_newton_fq
                if err_ADMM_newton_fq_[SNR] is None
                else np.vstack((err_ADMM_newton_fq_[SNR], err_ADMM_newton_fq))
            )
        if not np.isnan(err_ADMM_newton_fq_diag).any():
            err_ADMM_newton_fq_diag_[SNR] = (
                err_ADMM_newton_fq_diag
                if err_ADMM_newton_fq_diag_[SNR] is None
                else np.vstack((err_ADMM_newton_fq_diag_[SNR], err_ADMM_newton_fq_diag))
            )

        runs_.append(runs)
        run_.append(run)
        L_.append(L)
        N_sens_.append(N_sens)
        SNR_.append(SNR)
        h_.append(h)

    id += 1
# %%
for SNR in SNRs:
    fig = plt.figure(figsize=(8, 6))

    if err_MCQN_[SNR] is not None:
        plt.plot(20 * np.log10(np.nanmedian(err_MCQN_[SNR], axis=0)[:-L]), label="MCQN")
    if err_NMCFLMS_[SNR] is not None:
        plt.plot(
            20 * np.log10(np.nanmedian(err_NMCFLMS_[SNR], axis=0)[:-L]), label="NMCFLMS"
        )
    if err_RNMCFLMS_[SNR] is not None:
        plt.plot(
            20 * np.log10(np.nanmedian(err_RNMCFLMS_[SNR], axis=0)[:-L]),
            label="RNMCFLMS",
        )
    if err_LPRNMCFLMS_[SNR] is not None:
        plt.plot(
            20 * np.log10(np.nanmedian(err_LPRNMCFLMS_[SNR], axis=0)[:-L]),
            label="LPRNMCFLMS",
        )
    if err_ADMM_newton_td_[SNR] is not None:
        plt.plot(
            20 * np.log10(np.nanmedian(err_ADMM_newton_td_[SNR], axis=0)[:-L]),
            label="ADMM",
        )
    if err_ADMM_newton_fq_[SNR] is not None:
        plt.plot(
            20 * np.log10(np.nanmedian(err_ADMM_newton_fq_[SNR], axis=0)[:-L]),
            label="ADMM FQ",
        )
    if err_ADMM_newton_fq_diag_[SNR] is not None:
        plt.plot(
            20 * np.log10(np.nanmedian(err_ADMM_newton_fq_diag_[SNR], axis=0)[:-L]),
            label="ADMM FQ DIAG",
        )
    plt.plot(np.ones_like(err_MCQN) * (-SNR), "--k", label="Noise floor")
    plt.title("Misalignment")
    plt.xlabel("Frame [1]")
    plt.ylabel("Normalized Misalignment NPM [dB]")
    plt.grid()
    plt.legend()
    plt.show()
# %%
avg_range = 100
err_MCQN_avg = []
err_NMCFLMS_avg = []
err_RNMCFLMS_avg = []
err_LPRNMCFLMS_avg = []
err_ADMM_newton_td_avg = []
err_ADMM_newton_fq_avg = []
err_ADMM_newton_fq_diag_avg = []
for SNR in SNRs:
    err_MCQN_avg.append(
        np.nanmedian(err_MCQN_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    )
    err_NMCFLMS_avg.append(
        np.nanmedian(err_NMCFLMS_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    )
    err_RNMCFLMS_avg.append(
        np.nanmedian(err_RNMCFLMS_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    )
    err_LPRNMCFLMS_avg.append(
        np.nanmedian(err_LPRNMCFLMS_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    )
    err_ADMM_newton_td_avg.append(
        np.nanmedian(err_ADMM_newton_td_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    )
    err_ADMM_newton_fq_avg.append(
        np.nanmedian(err_ADMM_newton_fq_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    )
    err_ADMM_newton_fq_diag_avg.append(
        np.nanmedian(err_ADMM_newton_fq_diag_[SNR], axis=0)[
            -(avg_range + L) : -L
        ].mean()
    )

# %%
fig = plt.figure(figsize=(5, 2.5))
# plt.plot(SNRs, -np.asarray(SNRs), "k--", label="SNR")
plt.plot(SNRs, 20 * np.log10(err_MCQN_avg), marker="o", label="MCQN")
plt.plot(SNRs, 20 * np.log10(err_NMCFLMS_avg), marker=">", label="NMCFLMS")
# plt.plot(SNRs, 20 * np.log10(err_RNMCFLMS_avg), marker="o", label="RNMCFLMS")
plt.plot(SNRs, 20 * np.log10(err_LPRNMCFLMS_avg), marker="^", label=r"$l_p$-RNMCFLMS")
# plt.plot(SNRs, 20 * np.log10(err_ADMM_newton_td_avg), marker="v", label="ADMM")
# plt.plot(SNRs, 20 * np.log10(err_ADMM_newton_fq_avg), marker="o", label="ADMM FQ")
plt.plot(
    SNRs,
    20 * np.log10(err_ADMM_newton_fq_diag_avg),
    marker="s",
    label="ADMM BSI",
)
# plt.title("Median misalignment after 8000 Samples")
plt.xlabel("SNR [dB]")
plt.ylabel("Average NPM [dB]")
plt.xticks(SNRs)
plt.xlim(3, 52)
plt.ylim(-85, 0)
# plt.yticks(-np.asarray(SNRs))
plt.grid()
plt.legend(fontsize="small")
plt.tight_layout()
plt.show()

# %%
utils.savefig(fig, "NPM_over_SNR_L%d_M%d" % (L_p, N_p))

# %%
fig = plt.figure(figsize=(5, 2.5))
SNR = 10
if err_MCQN_[SNR] is not None:
    plt.plot(20 * np.log10(np.nanmedian(err_MCQN_[SNR], axis=0)[:-L]), label="MCQN")
if err_NMCFLMS_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_NMCFLMS_[SNR], axis=0)[:-L]), label="NMCFLMS"
    )
if err_LPRNMCFLMS_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_LPRNMCFLMS_[SNR], axis=0)[:-L]),
        label=r"$l_p$-RNMCFLMS",
    )
if err_ADMM_newton_fq_diag_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_ADMM_newton_fq_diag_[SNR], axis=0)[:-L]),
        label="ADMM BSI",
    )
plt.xlabel("Frame [1]")
plt.ylabel("NPM [dB]")
plt.xlim(0, 8000)
plt.xticks(np.arange(0, 8000, 1000))
plt.ylim(-40, 0)
plt.grid()
plt.legend(fontsize="small")
plt.tight_layout()
plt.show()
utils.savefig(fig, "NPM_over_time_SNR%d" % (SNR))

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

dir = "data/simulation_random_connectivity/"
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

    # id += 1
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
    fig = plt.figure(figsize=(3, 2.5))
    for i in range(len(densities)):
        if err_ADMM_newton_fq_diag_[i][N_sens] is not None:
            plt.plot(
                20
                * np.log10(
                    np.nanmedian(err_ADMM_newton_fq_diag_[i][N_sens], axis=0)[:lim]
                ),
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
# %% random IRS
SNRs = [5, 10, 15, 20, 30, 40, 50]
id = 0
runs_ = []
run_ = []
L_ = []
N_sens_ = []
SNR_ = []
h_ = []
err_MCQN_ = {}
err_NMCFLMS_ = {}
err_RNMCFLMS_ = {}
err_LPRNMCFLMS_ = {}
err_ADMM_newton_td_ = {}
err_ADMM_newton_fq_ = {}
err_ADMM_newton_fq_diag_ = {}
for SNR in SNRs:
    err_MCQN_[SNR] = None
    err_NMCFLMS_[SNR] = None
    err_RNMCFLMS_[SNR] = None
    err_LPRNMCFLMS_[SNR] = None
    err_ADMM_newton_td_[SNR] = None
    err_ADMM_newton_fq_[SNR] = None
    err_ADMM_newton_fq_diag_[SNR] = None

L_p = 128
N_p = 6

dir = "data/simulation_rim/"
files = sorted(os.listdir(dir))
for filename in files:
    print(filename)
    with open(dir + filename, "rb") as f:
        runs = np.load(f)
        run = np.load(f)
        L = np.load(f)
        N_sens = np.load(f)
        SNR = int(np.load(f))
        h = np.load(f)
        err_MCQN = np.load(f)
        err_NMCFLMS = np.load(f)
        err_RNMCFLMS = np.load(f)
        err_LPRNMCFLMS = np.load(f)
        err_ADMM_newton_td = np.load(f)
        err_ADMM_newton_fq = np.load(f)
        err_ADMM_newton_fq_diag = np.load(f)

        if L != L_p or N_sens != N_p:
            continue

        if not np.isnan(err_MCQN).any():
            err_MCQN_[SNR] = (
                err_MCQN
                if err_MCQN_[SNR] is None
                else np.vstack((err_MCQN_[SNR], err_MCQN))
            )
        if not np.isnan(err_NMCFLMS).any():
            err_NMCFLMS_[SNR] = (
                err_NMCFLMS
                if err_NMCFLMS_[SNR] is None
                else np.vstack((err_NMCFLMS_[SNR], err_NMCFLMS))
            )
        if not np.isnan(err_RNMCFLMS).any():
            err_RNMCFLMS_[SNR] = (
                err_RNMCFLMS
                if err_RNMCFLMS_[SNR] is None
                else np.vstack((err_RNMCFLMS_[SNR], err_RNMCFLMS))
            )
        if not np.isnan(err_LPRNMCFLMS).any():
            err_LPRNMCFLMS_[SNR] = (
                err_LPRNMCFLMS
                if err_LPRNMCFLMS_[SNR] is None
                else np.vstack((err_LPRNMCFLMS_[SNR], err_LPRNMCFLMS))
            )
        if not np.isnan(err_ADMM_newton_td).any():
            err_ADMM_newton_td_[SNR] = (
                err_ADMM_newton_td
                if err_ADMM_newton_td_[SNR] is None
                else np.vstack((err_ADMM_newton_td_[SNR], err_ADMM_newton_td))
            )
        if not np.isnan(err_ADMM_newton_fq).any():
            err_ADMM_newton_fq_[SNR] = (
                err_ADMM_newton_fq
                if err_ADMM_newton_fq_[SNR] is None
                else np.vstack((err_ADMM_newton_fq_[SNR], err_ADMM_newton_fq))
            )
        if not np.isnan(err_ADMM_newton_fq_diag).any():
            err_ADMM_newton_fq_diag_[SNR] = (
                err_ADMM_newton_fq_diag
                if err_ADMM_newton_fq_diag_[SNR] is None
                else np.vstack((err_ADMM_newton_fq_diag_[SNR], err_ADMM_newton_fq_diag))
            )

        runs_.append(runs)
        run_.append(run)
        L_.append(L)
        N_sens_.append(N_sens)
        SNR_.append(SNR)
        h_.append(h)

    id += 1
# %%
# for SNR in SNRs:
SNR = 40
fig = plt.figure(figsize=(8, 6))

if err_MCQN_[SNR] is not None:
    plt.plot(20 * np.log10(np.nanmedian(err_MCQN_[SNR], axis=0)[:-L]), label="MCQN")
if err_NMCFLMS_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_NMCFLMS_[SNR], axis=0)[:-L]), label="NMCFLMS"
    )
if err_RNMCFLMS_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_RNMCFLMS_[SNR], axis=0)[:-L]),
        label="RNMCFLMS",
    )
if err_LPRNMCFLMS_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_LPRNMCFLMS_[SNR], axis=0)[:-L]),
        label="LPRNMCFLMS",
    )
if err_ADMM_newton_td_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_ADMM_newton_td_[SNR], axis=0)[:-L]),
        label="ADMM",
    )
if err_ADMM_newton_fq_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_ADMM_newton_fq_[SNR], axis=0)[:-L]),
        label="ADMM FQ",
    )
if err_ADMM_newton_fq_diag_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_ADMM_newton_fq_diag_[SNR], axis=0)[:-L]),
        label="ADMM FQ DIAG",
    )
plt.plot(np.ones_like(err_MCQN) * (-SNR), "--k", label="Noise floor")
plt.title("Misalignment")
plt.xlabel("Frame [1]")
plt.ylabel("Normalized Misalignment NPM [dB]")
plt.grid()
plt.legend()
plt.show()

# %%
