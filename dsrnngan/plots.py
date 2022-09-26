import os
import pickle
from string import ascii_lowercase

import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec

import data
from noise import NoiseGenerator
from rapsd import plot_spectrum1d, rapsd
from thresholded_ranks import findthresh
# from cmcrameri import cm

path = os.path.dirname(os.path.abspath(__file__))


def plot_img(img, value_range=(np.log10(0.1), np.log10(100)), extent=None):
    plt.imshow(img,
               interpolation='nearest',
               norm=colors.Normalize(*value_range),
               origin='lower',
               extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)


def plot_img_log(img, value_range=(0.01, 5), extent=None):
    plt.imshow(img,
               interpolation='nearest',
               norm=colors.LogNorm(*value_range),
               origin='lower',
               extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)


def plot_img_log_coastlines(img, value_range_precip=(0.01, 5), cmap='viridis', extent=None, alpha=0.8):
    plt.imshow(img,
               interpolation='nearest',
               norm=colors.LogNorm(*value_range_precip),
               cmap=cmap,
               origin='lower',
               extent=extent,
               transform=ccrs.PlateCarree(),
               alpha=alpha)
    plt.gca().tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)


def truncate_colourmap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_sequences(gen,
                   mode,
                   batch_gen,
                   checkpoint,
                   noise_channels,
                   latent_variables,
                   num_samples=8,
                   num_instances=4,
                   out_fn=None):

    for cond, const, seq_real in batch_gen.as_numpy_iterator():
        batch_size = cond.shape[0]

    seq_gen = []
    if mode == 'GAN':
        for i in range(num_instances):
            noise_shape = cond[0, ..., 0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            seq_gen.append(gen.predict([cond, const, noise_gen()]))
    elif mode == 'det':
        for i in range(num_instances):
            seq_gen.append(gen.predict([cond, const]))
    elif mode == 'VAEGAN':
        # call encoder
        (mean, logvar) = gen.encoder([cond, const])
        # run decoder n times
        for i in range(num_instances):
            noise_shape = cond[0, ..., 0].shape + (latent_variables,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            seq_gen.append(gen.decoder.predict([mean, logvar, noise_gen(), const]))

    seq_real = data.denormalise(seq_real)
    cond = data.denormalise(cond)
    seq_gen = [data.denormalise(seq) for seq in seq_gen]

    num_rows = num_samples
    num_cols = 2+num_instances

    figsize = (num_cols*1.5, num_rows*1.5)
    plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(num_rows, num_cols,
                           wspace=0.05, hspace=0.05)

    value_range = (0, 5)  # batch_gen.decoder.value_range

    for s in range(num_samples):
        i = s
        plt.subplot(gs[i, 0])
        plot_img(seq_real[s, :, :, 0], value_range=value_range)
        plt.subplot(gs[i, 1])
        plot_img(cond[s, :, :, 0], value_range=value_range)
        for k in range(num_instances):
            j = 2+k
            plt.subplot(gs[i, j])
            plot_img(seq_gen[k][s, :, :, 0], value_range=value_range)

    plt.suptitle('Checkpoint ' + str(checkpoint))

    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()


def plot_rank_metrics_by_samples(metrics_fn,
                                 ax=None,
                                 plot_metrics=["KS", "DKL", "OP", "mean"],
                                 value_range=(-0.1, 0.2),
                                 linestyles=['solid', 'dashed', 'dashdot', ':'],
                                 opt_switch_point=350000,
                                 plot_switch_text=True):

    if ax is None:
        ax = plt.gca()

    df = pd.read_csv(metrics_fn, delimiter=" ")

    x = df["N"]
    for (metric, linestyle) in zip(plot_metrics, linestyles):
        y = df[metric]
        label = metric
        if metric == "DKL":
            label = "$D_\\mathrm{KL}$"
        if metric == "OP":
            label = "OF"
        if metric == "mean":
            y = y-0.5
            label = "mean - $\\frac{1}{2}$"
        ax.plot(x, y, label=label, linestyle=linestyle)

    ax.set_xlim((0, x.max()))
    ax.set_ylim(value_range)
    ax.axhline(0, linestyle='--', color=(0.75, 0.75, 0.75), zorder=-10)
    ax.axvline(opt_switch_point, linestyle='--', color=(0.75, 0.75, 0.75), zorder=-10)
    if plot_switch_text:
        text_x = opt_switch_point*0.98
        text_y = value_range[1]-(value_range[1]-value_range[0])*0.02
        ax.text(text_x, text_y, "Adam\u2192SGD", horizontalalignment='right',
                verticalalignment='top', color=(0.5, 0.5, 0.5))
    plt.grid(axis='y')


def plot_rank_metrics_by_samples_multiple(metrics_files,
                                          value_ranges=[(-0.025, 0.075),
                                                        (-0.1, 0.2)]):
    (fig, axes) = plt.subplots(len(metrics_files),
                               1,
                               sharex=True,
                               squeeze=True)
    plt.subplots_adjust(hspace=0.1)

    for (i, (ax, fn, vr)) in enumerate(zip(axes, metrics_files, value_ranges)):
        plot_rank_metrics_by_samples(fn,
                                     ax,
                                     plot_switch_text=(i == 0),
                                     value_range=vr)
        if i == len(metrics_files) - 1:
            ax.legend(ncol=5)
            ax.set_xlabel("Training sequences")
        ax.text(0.04, 0.97, "({})".format(ascii_lowercase[i]),
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)
        ax.set_ylabel("Rank metric")
        ax.grid(axis='y')


def plot_quality_metrics_by_samples(quality_metrics_fn,
                                    rank_metrics_fn,
                                    ax=None,
                                    plot_metrics=["RMSE", "MSSSIM", "LSD", "CRPS"],
                                    value_range=(0, 0.7),
                                    linestyles=['-', '--', ':', '-.']):

    if ax is None:
        ax = plt.gca()

    df = pd.read_csv(quality_metrics_fn, delimiter=" ")
    df_r = pd.read_csv(rank_metrics_fn, delimiter=" ")
    df["CRPS"] = df_r["CRPS"]

    x = df["N"]
    for (metric, linestyle) in zip(plot_metrics, linestyles):
        y = df[metric]
        label = metric
        if metric == "MSSSIM":
            y = 1-y
            label = "$1 - $MS-SSIM"
        if metric == "LSD":
            label = "LSD [dB] / 50"
            y = y/50
        if metric == "CRPS":
            y = y*10
            label = "CRPS $\\times$ 10"
        ax.plot(x, y, label=label, linestyle=linestyle)

    ax.set_xlim((0, x.max()))
    ax.set_ylim(value_range)
    ax.axhline(0, linestyle='--', color=(0.75, 0.75, 0.75), zorder=-10)


def plot_quality_metrics_by_samples_multiple(quality_metrics_files,
                                             rank_metrics_files):

    (fig, axes) = plt.subplots(len(quality_metrics_files), 1, sharex=True,
                               squeeze=True)
    plt.subplots_adjust(hspace=0.1)
    value_ranges = [(0, 0.4), (0, 0.8)]

    for (i, (ax, fn_q, fn_r, vr)) in enumerate(zip(axes, quality_metrics_files, rank_metrics_files, value_ranges)):
        plot_quality_metrics_by_samples(fn_q,
                                        fn_r,
                                        ax,
                                        plot_switch_text=(i == 0),
                                        value_range=vr)
        if i == 0:
            ax.legend(mode='expand', ncol=4, loc='lower left')
        if i == 1:
            ax.set_xlabel("Training sequences")
        ax.text(0.04, 0.97, "({})".format(ascii_lowercase[i]),
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)
        ax.set_ylabel("Quality metric")
        ax.grid(axis='y')


def plot_rank_histogram(ax, ranks, N_ranks=101, **plot_params):

    bc = np.linspace(0, 1, N_ranks)
    db = (bc[1] - bc[0])
    bins = bc - db/2
    bins = np.hstack((bins, bins[-1]+db))
    (h, _) = np.histogram(ranks, bins=bins)
    h = h / h.sum()

    ax.plot(bc, h, **plot_params)


def plot_rank_cdf(ax, ranks, N_ranks=101, **plot_params):

    bc = np.linspace(0, 1, N_ranks)
    db = (bc[1] - bc[0])
    bins = bc - db/2
    bins = np.hstack((bins, bins[-1]+db))
    (h, _) = np.histogram(ranks, bins=bins)
    h = h.cumsum()
    h = h / h[-1]

    ax.plot(bc, h, **plot_params)


def plot_rank_histogram_all(rank_files, 
                            labels, 
                            log_path, 
                            N_ranks=101,
                            threshold=False, 
                            freq=0.0001, 
                            lead_time=None,
                            model=None,
                            ablation=False):
    (fig, axes) = plt.subplots(2, 1, sharex=True, figsize=(6, 3))
    if lead_time is not None:
        if threshold:
            fig.suptitle('Rank histograms {} - top {:.2%}'.format(model, freq))
        else:
            fig.suptitle('Rank histograms {} - all'.format(model))
    else:
        if threshold:
            fig.suptitle('Rank histograms - top {:.2%}'.format(freq))
        else:
            fig.suptitle('Rank histograms - all')
    plt.subplots_adjust(hspace=0.15)
    plt.rcParams['font.size'] = '12'
    ls = "solid"

    for (fn_valid, label) in zip(rank_files, labels):
        with np.load(fn_valid) as f:
            ranks = f['ranks']
            if threshold:
                lowres = f['lowres']
                assert ranks.shape == lowres.shape
                thresh = findthresh(lowres, freq).root  # find IFS tp threshold
                ranks = ranks[np.where(lowres > thresh)]  # restrict to these events

        plot_rank_histogram(axes[0], ranks, N_ranks=N_ranks, label=label, linestyle=ls, linewidth=0.75, zorder=2)
        plot_rank_cdf(axes[1], ranks, N_ranks=N_ranks, label=label, linestyle=ls, linewidth=0.75, zorder=2)

    bc = np.linspace(0, 1, N_ranks)
    axes[0].plot(bc, [1./N_ranks]*len(bc), linestyle=':', label="Uniform", c='dimgrey', zorder=0)
    axes[0].set_ylabel("Norm. occurrence")
    ylim = axes[0].get_ylim()
    if threshold and lead_time:
        axes[0].set_ylim((0, 0.1))
    elif lead_time: 
        axes[0].set_ylim((0, 0.05))
    else:
        axes[0].set_ylim((0, ylim[1]))  
    axes[0].set_xlim((0, 1))
    axes[0].text(0.01, 0.97, "(a)",
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=axes[0].transAxes)

    axes[1].plot(bc, bc, linestyle=':', label="Ideal", c='dimgrey', zorder=0)
    axes[1].set_ylabel("CDF")
    axes[1].set_xlabel("Normalized rank")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlim((0, 1))
    axes[1].text(0.01, 0.97, "(b)",
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=axes[1].transAxes)
    axes[1].legend(bbox_to_anchor=(1.05, 1.05))

    if lead_time is not None:
        if threshold:
            plt.savefig("{}/rank-distribution-{}-{}-{}.pdf".format(log_path, 'lead_time', model, freq), bbox_inches='tight')
        else:
            plt.savefig("{}/rank-distribution-{}-{}.pdf".format(log_path, 'lead_time', model), bbox_inches='tight')
    elif ablation:
        if threshold:
            plt.savefig("{}/rank-distribution-ablation-{}.pdf".format(log_path, freq), bbox_inches='tight')
        else:
            plt.savefig("{}/rank-distribution-ablation.pdf".format(log_path), bbox_inches='tight')
    else:    
        if threshold:
            plt.savefig("{}/rank-distribution-{}.pdf".format(log_path, freq), bbox_inches='tight')
        else:
            plt.savefig("{}/rank-distribution.pdf".format(log_path), bbox_inches='tight')
    plt.close()


def plot_histograms(log_folder, val_year, ranks, N_ranks):
    rank_metrics_files = [os.path.join(log_folder, f"ranksnew-{val_year}-{rk}.npz") for rk in ranks]
    labels = [f"{rk}" for rk in ranks]
    plot_rank_histogram_all(rank_metrics_files, labels, log_folder, N_ranks=N_ranks)


def gridplot(models, model_labels=None,
             vmin=0, vmax=1):
    nx = models[0].shape[0]
    ny = len(models)
    fig = plt.figure(dpi=200, figsize=(nx, ny))
    gs1 = gridspec.GridSpec(ny, nx)
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.

    for i in range(nx):
        for j in range(ny):
            # print(i,j)
            ax = plt.subplot(gs1[i+j*nx])  # plt.subplot(ny,nx,i+1+j*nx)
            ax.pcolormesh(models[j][i, :, :], vmin=vmin, vmax=vmax)
            # ax.axis('off')
            ax.set(xticks=[], yticks=[])
            if i == 0 and (model_labels is not None):
                ax.set_ylabel(model_labels[j])
            ax.axis('equal')
    fig.text(0.5, 0.9, 'Dates', ha='center')
    fig.text(0.04, 0.5, 'Models', va='center', rotation='vertical')
    return


def plot_roc_curves(roc_files,
                    labels,
                    log_path,
                    precip_values,
                    pooling_methods,
                    ecpoint_methods,
                    lw=2):

    roc_data = {}
    for (fn, label) in zip(roc_files, labels):
        with open(fn, 'rb') as handle:
            roc_data[label] = pickle.load(handle)

    for method in pooling_methods:
        for i in range(len(precip_values)):
            plt.plot(roc_data['IFS_fpr'][method][i],
                     roc_data['IFS_tpr'][method][i],
                     linestyle="",
                     marker="x",
                     label="IFS")
            plt.plot(roc_data['GAN_fpr'][method][i],
                     roc_data['GAN_tpr'][method][i],
                     lw=lw,
                     label="GAN (area = %0.2f)" % roc_data['GAN_auc'][method][i])
            plt.plot(roc_data['VAEGAN_fpr'][method][i],
                     roc_data['VAEGAN_tpr'][method][i],
                     lw=lw,
                     label="VAEGAN (area = %0.2f)" % roc_data['VAEGAN_auc'][method][i])
            for ecpoint_method in ecpoint_methods:
                ecpoint_label = 'ecPoint_' + ecpoint_method
                if method == 'no_pooling' and ecpoint_method == 'partcorr':
                    continue
                if method == 'no_pooling' and ecpoint_method == 'nocorr':
                    ecpoint_plot_label = 'ecPoint'
                else:
                    ecpoint_plot_label = ecpoint_label
                plt.plot(roc_data[ecpoint_label + '_' + 'fpr'][method][i],
                         roc_data[ecpoint_label + '_' + 'tpr'][method][i],
                         lw=lw,
                         label=f"{ecpoint_plot_label} (area = %0.2f)" % roc_data[ecpoint_label + '_' + 'auc'][method][i])

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # no-skill
            plt.plot([], [], ' ',
                     label="event frequency %0.3f" % roc_data['GAN_base'][method][i])  # plot base rate
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate,  FP/(FP+TN)')
            plt.ylabel('True Positive Rate, TP/(TP+FN)')
            plt.title(f'ROC curve for >{precip_values[i]}mm, {method}')
            plt.legend(loc="lower right")
            plt.savefig("{}/ROC-{}-{}.pdf".format(log_path, precip_values[i], method), bbox_inches='tight')
            plt.close()


def plot_prc_curves(prc_files,
                    labels,
                    log_path,
                    precip_values,
                    pooling_methods,
                    ecpoint_methods,
                    lw=2):

    prc_data = {}
    for (fn, label) in zip(prc_files, labels):
        with open(fn, 'rb') as handle:
            prc_data[label] = pickle.load(handle)

    for method in pooling_methods:
        for i in range(len(precip_values)):
            plt.plot(prc_data['IFS_rec'][method][i],
                     prc_data['IFS_pre'][method][i],
                     linestyle="",
                     marker="x",
                     label="IFS")
            plt.plot(prc_data['GAN_rec'][method][i],
                     prc_data['GAN_pre'][method][i],
                     lw=lw,
                     label="GAN (area = %0.2f)" % prc_data['GAN_auc'][method][i])
            plt.plot(prc_data['VAEGAN_rec'][method][i],
                     prc_data['VAEGAN_pre'][method][i],
                     lw=lw,
                     label="VAEGAN (area = %0.2f)" % prc_data['VAEGAN_auc'][method][i])
            for ecpoint_method in ecpoint_methods:
                ecpoint_label = 'ecPoint_' + ecpoint_method
                if method == 'no_pooling' and ecpoint_method == 'partcorr':
                    continue
                if method == 'no_pooling' and ecpoint_method == 'nocorr':
                    ecpoint_plot_label = 'ecPoint'
                else:
                    ecpoint_plot_label = ecpoint_label
                plt.plot(prc_data[ecpoint_label + '_' + 'rec'][method][i],
                         prc_data[ecpoint_label + '_' + 'pre'][method][i],
                         lw=lw,
                         label=f"{ecpoint_plot_label} (area = %0.2f)" % prc_data[ecpoint_label + '_' + 'auc'][method][i])
            plt.plot([0, 1],
                     [prc_data['GAN_base'][method][i], prc_data['GAN_base'][method][i]],
                     '--',
                     lw=0.5,
                     color='gray',
                     label="event frequency %0.3f" % prc_data['GAN_base'][method][i])  # no skill
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall TP/(TP+FN)')
            plt.ylabel('Precision TP/(TP+FP)')
            plt.title(f'Precision-recall curve for >{precip_values[i]}mm, {method}')
            plt.legend()
            plt.savefig("{}/PR-{}-{}.pdf".format(log_path, precip_values[i], method), bbox_inches='tight')
            plt.close()


def plot_fss(fss_files,
             labels,
             log_path,
             nimg,
             precip_values,
             spatial_scales,
             ecpoint_methods,
             lw=2):

    for i in range(len(precip_values)):
        baserate_first = None
        plt.figure(figsize=(7, 5), dpi=200)
        plt.gcf().set_facecolor("white")
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # load data
        j = 0
        for (fn, label, color) in zip(fss_files, labels, colors):
            f1 = fn + '-1.pickle'
            f2 = fn + '-2.pickle'
            with open(f1, 'rb') as f:
                m1data = pickle.load(f)[precip_values[i]]
            with open(f2, 'rb') as f:
                m2data = pickle.load(f)[precip_values[i]]

            assert spatial_scales == list(m1data.keys()), "spatial scales do not match"

            y1 = [m1data[spasc]["score"] for spasc in spatial_scales]
            y2 = [m2data[spasc]["score"] for spasc in spatial_scales]
            plt.semilogx(spatial_scales, y1, '-', color=color, lw=lw,
                         label=f"{labels[j]}")
            plt.semilogx(spatial_scales, y2, ':', color=color, lw=lw)

            # obtain base frequency for no-skill and target-skill lines
            baserate = m1data[1]['fssobj']['sum_obs_sq']/(nimg*940*940)
            # sanity check that the truth base rate is the same for
            # each model tested -- if not, bug / different batches etc
            if baserate_first is None:
                baserate_first = baserate
            else:
                assert np.isclose(baserate, baserate_first)
            j = j + 1

        target_skill = 0.5 + baserate_first/2
        plt.semilogx([1.0, spatial_scales[-1]], [baserate, baserate],
                     '-', color='0.9', lw=lw)
        plt.semilogx([1.0, spatial_scales[-1]], [target_skill, target_skill],
                     '-', color='0.8', lw=lw)
        # plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # no-skill
        # plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Spatial scale (km)')
        plt.ylabel('Fractions skill score (FSS)')
        plt.title(f'FSS curve for precip threshold {precip_values[i]}')
        plt.legend(loc="best")
        pltsave = os.path.join(log_path, f"FSS-{precip_values[i]}.pdf")
        plt.savefig(pltsave, bbox_inches='tight')
        plt.close()


def plot_rapsd(rapsd_files,
               num_samples,
               labels,
               log_path,
               spatial_scales):

    rapsd_data = {}
    for (fn, label) in zip(rapsd_files, labels):
        with open(fn, 'rb') as handle:
            rapsd_data[label] = pickle.load(handle)

    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colours += ['#56B4E9', '#009E73', '#F0E442']  # add some
    for k in range(num_samples):
        for model in labels:
            # sniff data labels
            rapsd_data_labels = list(rapsd_data[model][0].keys())
            fig, ax = plt.subplots()
            for (i, colour) in zip(range(len(rapsd_data_labels)), colours):
                if rapsd_data_labels[i] == 'IFS':
                    # skip the input data b/c the resolution is different
                    pass
                elif rapsd_data_labels[i] == 'ecPoint mean':
                    # skip b/c it's not spatially coherent
                    pass
                elif len(rapsd_data[model][k][rapsd_data_labels[i]].shape) != 2:
                    # if preds weren't made for this method don't plot
                    # if preds were made shape will be (940, 940)
                    pass
                else:
                    if rapsd_data_labels[i] == 'TRUTH':
                        lw = 2.0
                    else:
                        lw = 1.0
                    R_1, freq_1 = rapsd(rapsd_data[model][k][rapsd_data_labels[i]],
                                        fft_method=np.fft, return_freq=True)
                    # Plot the observed power spectrum and the model
                    plot_spectrum1d(freq_1,
                                    R_1,
                                    x_units="km",
                                    y_units="dBR",
                                    color=colour,
                                    ax=ax,
                                    lw=lw,
                                    label=rapsd_data_labels[i],
                                    wavelength_ticks=spatial_scales)
            ax.set_title(f"Radially averaged log-power spectrum - {k+1}")
            ax.legend(loc="best")
            pltsave = os.path.join(log_path, f"RAPSD-{model}-{k+1}.pdf")
            plt.savefig(pltsave, bbox_inches='tight')
            plt.close()


def plot_preds(pred_files,
               num_samples,
               labels,
               log_path,
               preds_to_plot,
               value_range_precip=(0.1, 30),
               palette="YlGnBu"):
    pred_data = {}
    for (fn, label) in zip(pred_files, labels):
        with open(fn, 'rb') as handle:
            pred_data[label] = pickle.load(handle)
    num_cols = num_samples
    num_rows = len(preds_to_plot)
    plt.figure(figsize=(1.5*num_cols, 1.5*num_rows), dpi=300)
    linewidth = 0.4
    cmap = ListedColormap(sns.color_palette(palette, 256))
    cmap.set_under('white')
    extent = [-7.5, 2, 49.5, 59]  # (lon, lat)
    alpha = 0.8
    spacing = 10

    # colorbar
    units = "Rain rate [mm h$^{-1}$]"
    cb_tick_loc = np.array([0.1, 0.5, 1, 2, 5, 15, 30, 50])
    cb_tick_labels = [0.1, 0.5, 1, 2, 5, 15, 30, 50]

    gs = gridspec.GridSpec(spacing*num_rows+1, spacing*num_cols, wspace=0.5, hspace=0.5)

    for k in range(num_samples):
        for i in range(num_rows):
            if i < (num_rows)/2:
                label = 'GAN'
            else:
                label = 'VAEGAN'
            plt.subplot(gs[(spacing*i):(spacing+spacing*i), (spacing*k):(spacing+spacing*k)],
                        projection=ccrs.PlateCarree())
            ax = plt.gca()
            ax.coastlines(resolution='10m', color='black', linewidth=linewidth)
            if i < (len(preds_to_plot)):
                plot_img_log_coastlines(pred_data[label][k][preds_to_plot[i]],
                                        value_range_precip=value_range_precip,
                                        cmap=cmap,
                                        extent=extent,
                                        alpha=alpha)
            if i == 0:
                title = pred_data[label][k]['dates'][:4] + '-' + pred_data[label][k]['dates'][4:6] + '-' + pred_data[label][k]['dates'][6:8] + ' ' + str(pred_data[label][k]['hours']) + 'Z'
                plt.title(title, fontsize=9)

            if k == 0:
                if i < (len(preds_to_plot)):
                    ax.set_ylabel(preds_to_plot[i], fontsize=8)  # cartopy takes over the xlabel and ylabel
                    ax.set_yticks([])  # this weird hack restores them. WHY?!?!

    plt.suptitle('Example predictions for different input conditions')

    cax = plt.subplot(gs[-1, 1:-1]).axes
    cb = colorbar.ColorbarBase(cax, norm=colors.LogNorm(*value_range_precip), cmap=cmap, orientation='horizontal')
    cb.set_ticks(cb_tick_loc)
    cb.set_ticklabels(cb_tick_labels)
    cax.tick_params(labelsize=12)
    cb.set_label(units, size=12)
    # cannot save as pdf - will produce artefacts
    plt.savefig("{}/predictions-{}.png".format(log_path,
                                               num_samples), bbox_inches='tight')
    plt.close()


def plot_comparison(files,
                    num_samples,
                    labels,
                    log_path,
                    comp_to_plot,
                    value_range_precip=(0.1, 30),
                    palette="mako_r"):

    pred_data = {}
    for (fn, label) in zip(files, labels):
        with open(fn, 'rb') as handle:
            pred_data[label] = pickle.load(handle)

    num_cols = num_samples
    num_rows = len(comp_to_plot)
    plt.figure(figsize=(1.5*num_cols, 1.5*num_rows), dpi=300)
    linewidth = 0.4
    cmap = ListedColormap(sns.color_palette(palette, 256))
    cmap.set_under('white')
    extent = [-7.5, 2, 49.5, 59]  # (lon, lat)
    alpha = 0.8
    spacing = 10

    # colorbar
    units = "Rain rate [mm h$^{-1}$]"
    cb_tick_loc = np.array([0.1, 0.5, 1, 2, 5, 15, 30, 50])
    cb_tick_labels = [0.1, 0.5, 1, 2, 5, 15, 30, 50]

    gs = gridspec.GridSpec(spacing*num_rows+1, spacing*num_cols, wspace=0.5, hspace=0.5)

    for k in range(num_samples):
        for i in range(num_rows):
            plt.subplot(gs[(spacing*i):(spacing+spacing*i), (spacing*k):(spacing+spacing*k)],
                        projection=ccrs.PlateCarree())
            ax = plt.gca()
            ax.coastlines(resolution='10m', color='black', linewidth=linewidth)
            if i == 0:  # plot IFS
                plot_img_log_coastlines(pred_data['4x'][k]['IFS'],
                                        value_range_precip=value_range_precip,
                                        cmap=cmap,
                                        extent=extent,
                                        alpha=alpha)
                title = pred_data['4x'][k]['dates'][:4] + '-' + pred_data['4x'][k]['dates'][4:6] + '-' + pred_data['4x'][k]['dates'][6:8] + ' ' + str(pred_data['4x'][k]['hours']) + 'Z'
                plt.title(title, fontsize=9)
            if i == 1:  # plot TRUTH
                plot_img_log_coastlines(pred_data['4x'][k]['TRUTH'],
                                        value_range_precip=value_range_precip,
                                        cmap=cmap,
                                        extent=extent,
                                        alpha=alpha)
            if i > 1:  # plot different flavours of prediction
                plot_img_log_coastlines(pred_data[comp_to_plot[i]][k]['GAN pred 1'],
                                        value_range_precip=value_range_precip,
                                        cmap=cmap,
                                        extent=extent,
                                        alpha=alpha)
            if k == 0:
                if i < (len(comp_to_plot)):
                    ax.set_ylabel(comp_to_plot[i], fontsize=8)  # cartopy takes over the xlabel and ylabel
                    ax.set_yticks([])  # this weird hack restores them. WHY?!?!

    plt.suptitle('Example predictions for different input conditions')

    cax = plt.subplot(gs[-1, 1:-1]).axes
    cb = colorbar.ColorbarBase(cax, norm=colors.LogNorm(*value_range_precip), cmap=cmap, orientation='horizontal')
    cb.set_ticks(cb_tick_loc)
    cb.set_ticklabels(cb_tick_labels)
    cax.tick_params(labelsize=12)
    cb.set_label(units, size=12)
    # cannot save as pdf - will produce artefacts
    plt.savefig("{}/comparison-{}.png".format(log_path,
                                              num_samples), bbox_inches='tight')
    plt.close()
