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


# %%
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
dir = "data/"
files = sorted(os.listdir(dir))

# %%
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
    if err_MCQN_[SNR] is None:
        continue
    fig = plt.figure(figsize=(8, 6))
    plt.plot(20 * np.log10(np.nanmedian(err_MCQN_[SNR], axis=0)[:-L]), label="MCQN")
    plt.plot(
        20 * np.log10(np.nanmedian(err_NMCFLMS_[SNR], axis=0)[:-L]), label="NMCFLMS"
    )
    plt.plot(
        20 * np.log10(np.nanmedian(err_RNMCFLMS_[SNR], axis=0)[:-L]), label="RNMCFLMS"
    )
    plt.plot(
        20 * np.log10(np.nanmedian(err_LPRNMCFLMS_[SNR], axis=0)[:-L]),
        label="LPRNMCFLMS",
    )
    plt.plot(
        20 * np.log10(np.nanmedian(err_ADMM_newton_td_[SNR], axis=0)[:-L]),
        label="ADMM",
    )
    plt.plot(
        20 * np.log10(np.nanmedian(err_ADMM_newton_fq_[SNR], axis=0)[:-L]),
        label="ADMM FQ",
    )
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
    data = 20 * np.log10(
        np.mean(err_MCQN_[SNR][:, -(avg_range + L) : -L], axis=1).squeeze()
    )
    data = np.vstack(
        (
            data,
            20
            * np.log10(
                np.mean(err_NMCFLMS_[SNR][:, -(avg_range + L) : -L], axis=1).squeeze()
            ),
        )
    )
    data = np.vstack(
        (
            data,
            20
            * np.log10(
                np.mean(err_RNMCFLMS_[SNR][:, -(avg_range + L) : -L], axis=1).squeeze()
            ),
        )
    )
    data = np.vstack(
        (
            data,
            20
            * np.log10(
                np.mean(
                    err_LPRNMCFLMS_[SNR][:, -(avg_range + L) : -L], axis=1
                ).squeeze()
            ),
        )
    )
    data = np.vstack(
        (
            data,
            20
            * np.log10(
                np.mean(
                    err_ADMM_newton_td_[SNR][:, -(avg_range + L) : -L], axis=1
                ).squeeze()
            ),
        )
    )
    data = np.vstack(
        (
            data,
            20
            * np.log10(
                np.mean(
                    err_ADMM_newton_fq_[SNR][:, -(avg_range + L) : -L], axis=1
                ).squeeze()
            ),
        )
    )
    data = np.vstack(
        (
            data,
            20
            * np.log10(
                np.mean(
                    err_ADMM_newton_fq_diag_[SNR][:, -(avg_range + L) : -L], axis=1
                ).squeeze()
            ),
        )
    )
    plt.boxplot(data.T)
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
fig = plt.figure(figsize=(5, 3))
# plt.plot(SNRs, -np.asarray(SNRs), "k--", label="SNR")
plt.plot(SNRs, 20 * np.log10(err_MCQN_avg), marker="o", label="MCQN")
plt.plot(SNRs, 20 * np.log10(err_NMCFLMS_avg), marker=">", label="NMCFLMS")
# plt.plot(SNRs, 20 * np.log10(err_RNMCFLMS_avg), marker="o", label="RNMCFLMS")
plt.plot(SNRs, 20 * np.log10(err_LPRNMCFLMS_avg), marker="^", label="LPRNMCFLMS")
plt.plot(SNRs, 20 * np.log10(err_ADMM_newton_td_avg), marker="v", label="ADMM")
plt.plot(SNRs, 20 * np.log10(err_ADMM_newton_fq_avg), marker="o", label="ADMM FQ")
plt.plot(
    SNRs,
    20 * np.log10(err_ADMM_newton_fq_diag_avg),
    marker="s",
    label="ADMM FQ DIAG",
)
# plt.title("Median misalignment after 8000 Samples")
plt.xlabel("SNR [dB]")
plt.ylabel("Average NPM [dB]")
plt.xticks(SNRs)
# plt.yticks(-SNRs)
plt.grid()
plt.legend(fontsize="small", ncol=2)
plt.tight_layout()
plt.show()

# %%
utils.savefig(fig, "NPM_over_SNR_L%d_M5" % (L))

# %%
