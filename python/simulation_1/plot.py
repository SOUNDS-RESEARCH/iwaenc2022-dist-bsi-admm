# %%
import os
import numpy as np
import time
import utils
import matplotlib.pyplot as plt

# %% random IRS
SNRs = [5, 10, 15, 20, 30, 40, 50]
err_MCQN_ = {}
err_NMCFLMS_ = {}
# err_RNMCFLMS_ = {}
err_LPRNMCFLMS_ = {}
# err_ADMM_newton_td_ = {}
# err_ADMM_newton_fq_ = {}
err_ADMM_newton_fq_diag_ = {}
for SNR in SNRs:
    err_MCQN_[SNR] = None
    err_NMCFLMS_[SNR] = None
    # err_RNMCFLMS_[SNR] = None
    err_LPRNMCFLMS_[SNR] = None
    # err_ADMM_newton_td_[SNR] = None
    # err_ADMM_newton_fq_[SNR] = None
    err_ADMM_newton_fq_diag_[SNR] = None

dir = "data/"
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
        # err_RNMCFLMS = np.load(f)
        err_LPRNMCFLMS = np.load(f)
        # err_ADMM_newton_td = np.load(f)
        # err_ADMM_newton_fq = np.load(f)
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
        # if not np.isnan(err_RNMCFLMS).any():
        #     err_RNMCFLMS_[SNR] = (
        #         err_RNMCFLMS
        #         if err_RNMCFLMS_[SNR] is None
        #         else np.vstack((err_RNMCFLMS_[SNR], err_RNMCFLMS))
        #     )
        if not np.isnan(err_LPRNMCFLMS).any():
            err_LPRNMCFLMS_[SNR] = (
                err_LPRNMCFLMS
                if err_LPRNMCFLMS_[SNR] is None
                else np.vstack((err_LPRNMCFLMS_[SNR], err_LPRNMCFLMS))
            )
        # if not np.isnan(err_ADMM_newton_td).any():
        #     err_ADMM_newton_td_[SNR] = (
        #         err_ADMM_newton_td
        #         if err_ADMM_newton_td_[SNR] is None
        #         else np.vstack((err_ADMM_newton_td_[SNR], err_ADMM_newton_td))
        #     )
        # if not np.isnan(err_ADMM_newton_fq).any():
        #     err_ADMM_newton_fq_[SNR] = (
        #         err_ADMM_newton_fq
        #         if err_ADMM_newton_fq_[SNR] is None
        #         else np.vstack((err_ADMM_newton_fq_[SNR], err_ADMM_newton_fq))
        #     )
        if not np.isnan(err_ADMM_newton_fq_diag).any():
            err_ADMM_newton_fq_diag_[SNR] = (
                err_ADMM_newton_fq_diag
                if err_ADMM_newton_fq_diag_[SNR] is None
                else np.vstack((err_ADMM_newton_fq_diag_[SNR], err_ADMM_newton_fq_diag))
            )

# %%
for SNR in SNRs:
    fig = plt.figure(figsize=(8, 6))

    if err_MCQN_[SNR] is not None:
        plt.plot(20 * np.log10(np.nanmedian(err_MCQN_[SNR], axis=0)[:-L]), label="MCQN")
    if err_NMCFLMS_[SNR] is not None:
        plt.plot(
            20 * np.log10(np.nanmedian(err_NMCFLMS_[SNR], axis=0)[:-L]), label="NMCFLMS"
        )
    # if err_RNMCFLMS_[SNR] is not None:
    #     plt.plot(
    #         20 * np.log10(np.nanmedian(err_RNMCFLMS_[SNR], axis=0)[:-L]),
    #         label="RNMCFLMS",
    #     )
    if err_LPRNMCFLMS_[SNR] is not None:
        plt.plot(
            20 * np.log10(np.nanmedian(err_LPRNMCFLMS_[SNR], axis=0)[:-L]),
            label="LPRNMCFLMS",
        )
    # if err_ADMM_newton_td_[SNR] is not None:
    #     plt.plot(
    #         20 * np.log10(np.nanmedian(err_ADMM_newton_td_[SNR], axis=0)[:-L]),
    #         label="ADMM",
    #     )
    # if err_ADMM_newton_fq_[SNR] is not None:
    #     plt.plot(
    #         20 * np.log10(np.nanmedian(err_ADMM_newton_fq_[SNR], axis=0)[:-L]),
    #         label="ADMM FQ",
    #     )
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
# err_RNMCFLMS_avg = []
err_LPRNMCFLMS_avg = []
# err_ADMM_newton_td_avg = []
# err_ADMM_newton_fq_avg = []
err_ADMM_newton_fq_diag_avg = []
for SNR in SNRs:
    err_MCQN_avg.append(
        np.nanmedian(err_MCQN_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    )
    err_NMCFLMS_avg.append(
        np.nanmedian(err_NMCFLMS_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    )
    # err_RNMCFLMS_avg.append(
    #     np.nanmedian(err_RNMCFLMS_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    # )
    err_LPRNMCFLMS_avg.append(
        np.nanmedian(err_LPRNMCFLMS_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    )
    # err_ADMM_newton_td_avg.append(
    #     np.nanmedian(err_ADMM_newton_td_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    # )
    # err_ADMM_newton_fq_avg.append(
    #     np.nanmedian(err_ADMM_newton_fq_[SNR], axis=0)[-(avg_range + L) : -L].mean()
    # )
    err_ADMM_newton_fq_diag_avg.append(
        np.nanmedian(err_ADMM_newton_fq_diag_[SNR], axis=0)[
            -(avg_range + L) : -L
        ].mean()
    )

# %%
fig = plt.figure(figsize=(5, 2.5))
plt.plot(SNRs, 20 * np.log10(err_MCQN_avg), ":", marker="o", label="MCQN")
plt.plot(SNRs, 20 * np.log10(err_NMCFLMS_avg), "--", marker=">", label="NMCFLMS")
plt.plot(
    SNRs, 20 * np.log10(err_LPRNMCFLMS_avg), "-.", marker="^", label=r"$l_p$-RNMCFLMS"
)
plt.plot(
    SNRs,
    20 * np.log10(err_ADMM_newton_fq_diag_avg),
    marker="s",
    label="proposed",
)
plt.xlabel("SNR [dB]")
plt.ylabel("Average NPM [dB]")
plt.xticks(SNRs)
plt.xlim(3, 52)
plt.ylim(-85, 0)
plt.grid()
plt.legend(fontsize="small")
plt.tight_layout()
plt.show()
utils.savefig(fig, "NPM_over_SNR_L%d_M%d" % (L, N_sens))

# %%
fig = plt.figure(figsize=(5, 2.5))
SNR = 20
if err_MCQN_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_MCQN_[SNR], axis=0)[:-L]), "--", label="MCQN"
    )
if err_NMCFLMS_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_NMCFLMS_[SNR], axis=0)[:-L]),
        ":",
        label="NMCFLMS",
    )
if err_LPRNMCFLMS_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_LPRNMCFLMS_[SNR], axis=0)[:-L]),
        "-.",
        label=r"$l_p$-RNMCFLMS",
    )
if err_ADMM_newton_fq_diag_[SNR] is not None:
    plt.plot(
        20 * np.log10(np.nanmedian(err_ADMM_newton_fq_diag_[SNR], axis=0)[:-L]),
        label="proposed",
    )
plt.xlabel("Frame [1]")
plt.ylabel("NPM [dB]")
plt.xlim(0, 10000)
plt.xticks(np.arange(0, 10000, 1000))
plt.ylim(-50, 0)
plt.grid()
plt.legend(fontsize="small")
plt.tight_layout()
plt.show()
utils.savefig(fig, "NPM_over_time_SNR%d" % (SNR))

# %%
