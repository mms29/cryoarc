


import matplotlib.pyplot as plt
import numpy as np
import torch
from cryodrgn.utils import load_pkl
from umap import UMAP
from sklearn.decomposition import PCA
import os 

# base_dirs=[
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"]

# noise_levels = [0,-1,-2,-2.5,-3]

# methods=["avg",
#          "run_cryodrgn", "run_cryodrgn_sgd", 
#          "run_drgnai",
#          "run","run_sgd", "run_conv_sgd", "run_fourier","run_fourier_sgd","run_conv_fourier_sgd", "run_conv_pair_fourier", "run_conv_pair_fourier_sgd",
#          "run_conv_sgd_beta", "run_conv_sgd_highlr", "run_conv_sgd_highwd", "run_conv_sgd_low_struct_loss", "run_conv_sgd_table", "run_conv_pair_sgd_4blocks_lowlr"]
# names=["Homogeneous",
#        "CryoDRGN", "CryoDRGN-Pose", 
#        "DRGN-AI",
#        "MLP-MLP","MLP-MLP-Pose", "CNN-MLP-Pose", "MLP-MLP-FT","MLP-MLP-FT-Pose","CNN-MLP-FT-Pose","CNN-Pair-FT","CNN-Pair-FT-Pose",
#        "Beta", "HighLR", "HighWD",  "LowStructLoss", "Table", "PairLowLR4Blocks"
#        ]
# prefix = "/home/vuillemr/flexfold/data/cryofold/AKMD/"
# methods_ff =[ m for m in names if not ("DRGN" in m or "Homogeneous" in m) ]

# colors =   [plt.cm.Greens(v) for v in np.linspace(0.7, 0.8, 1) ] + [plt.cm.Reds(v) for v in np.linspace(0.6, 0.8,3) ] + [plt.cm.Blues(v) for v in np.linspace(0.5, 0.8, len(methods_ff)) ]

base_dirs=[
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr1",
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1",
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01",
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005",
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"]

noise_levels = [0,-1,-2,-2.5,-3]

methods=["avg",
         "run_cryodrgn", "run_cryodrgn_sgd", 
         "run_drgnai",
         "run_table_pair","run_table_pair_sgd", "run_conv_sgd", "run_conv_pair_new_sgd", "run_conv_mlp_new_sgd"]
names=["Homogeneous",
       "CryoDRGN", "CryoDRGN-Pose", 
       "DRGN-AI",
       "run_table_pair", "run_table_pair_sgd", "run_conv_sgd", "run_conv_pair_new_sgd", "run_conv_mlp_new_sgd"
       ]
prefix = "/home/vuillemr/flexfold/data/cryofold/AKMD/table"
methods_ff =[ m for m in names if not ("DRGN" in m or "Homogeneous" in m) ]

colors =   [plt.cm.Greens(v) for v in np.linspace(0.7, 0.8, 1) ] + [plt.cm.Reds(v) for v in np.linspace(0.6, 0.8,3) ] + [plt.cm.Blues(v) for v in np.linspace(0.5, 0.8, len(methods_ff)) ]



# base_dirs=[
# # "/home/vuillemr/flexfold/data/cryofold/AKMD/snr1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"]

# noise_levels = [-1, -2,-2.5,-3]

# methods=["avg","run_cryodrgn_sgd","run_drgnai","run_conv_sgd"]
# names=["Homogeneous","CryoDRGN","DRGN-AI","Ours"]
# prefix = "/home/vuillemr/flexfold/data/cryofold/AKMD/essential_"


# colors = ["tab:green", "tab:orange", "tab:red", "tab:blue"]


##################### 
# COnformatinoal space
#####################

data_pca = []
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, m in enumerate(methods):
        if "drgnai" in m:
            filename = basedir + "/" + m + "/out/conf.99.pkl"
        else:
            filename = basedir + "/" + m + "/z.99.pkl"
        if not os.path.isfile(filename):
            data_pca.append(None)
            print("missing %s "%filename)
        else:
            z = load_pkl(filename)
            print(i)
            # dimred = UMAP(n_components=2, n_neighbors=50, min_dist=0.1)
            # data_umap.append(dimred.fit_transform(z)[:,:2])
            dimred = PCA(n_components=2)
            data_pca.append(dimred.fit_transform(z)[:,:2])

data=data_pca
nrows = len(base_dirs)
ncols = len(methods)
cmap = "plasma"
c = np.repeat(np.arange(100), 100)
fig, ax = plt.subplots(nrows, (ncols-1), figsize=((ncols-1)*4, nrows*3), layout="constrained")  
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, (m,n) in enumerate(zip(methods[1:], names[1:])):
        ind = ncols*i + j+1
        if data[ind] is not None:
            ax[i,j].scatter(data[ind][:,0],data[ind][:,1], c=c, cmap=cmap, s=2.0, alpha=0.5)
            ax[i,j].set_title(f"{n} - $SNR=10^{{{noise:g}}}$")
        else:
            print("missing %s "%n)

fig.savefig(prefix+"summary_pca.png", dpi=150)


# data_umap = []
# for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
#     for j, m in enumerate(methods):
#         filename = basedir + "/" + m + "/z.99.pkl"
#         if not os.path.isfile(filename):
#             data_umap.append(None)
#         else:
#             z = load_pkl(filename)
#             print(i)
#             dimred = UMAP(n_components=2, )#n_neighbors=50, min_dist=0.1)
#             data_umap.append(dimred.fit_transform(z)[:,:2])
# data=data_umap
# nrows = len(base_dirs)
# ncols = len(methods)
# cmap = "plasma"
# c = np.repeat(np.arange(100), 100)
# fig, ax = plt.subplots(nrows, (ncols-1), figsize=((ncols-1)*4, nrows*3), layout="constrained")  
# for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
#     for j, (m,n) in enumerate(zip(methods[1:], names[1:])):
#         ind = ncols*i + j +1
#         if data[ind] is not None:
#             ax[i,j].scatter(data[ind][:,0],data[ind][:,1], c=c, cmap=cmap, s=2.0, alpha=0.5)
#             ax[i,j].set_title(f"{n} - $SNR=10^{{{noise:g}}}$")
# fig.savefig(prefix+"summary_umap.png", dpi=300)

##################### 
# FSC
#####################
from matplotlib.ticker import FuncFormatter
def fraction_formatter(x, pos):
    if x == 0:
        return "0"
    return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places
# tab_colors = list(plt.cm.tab10.colors)

methods_ff =[ m for m in names if not ("CryoDRGN" in m or "Homogeneous" in m) ]

fig, ax = plt.subplots(1, len(noise_levels), figsize=(len(noise_levels)*4,5), layout="constrained")
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, (m,n) in enumerate(zip(methods, names)):
        if "cryodrgn" in m :
            loc = "analyze.100"
        elif "drgnai" in m:
            loc = 'out/analysis_99'
        else:
            loc = "analysis"
        filename = basedir + "/" + m + "/" +loc  + "/assert.pkl"
        if os.path.isfile(filename):
            assert_dict = load_pkl(filename)
            # ax[i].errorbar(x=assert_dict["pixres"], y=np.mean(assert_dict["fsc"],axis=0), yerr=np.std(assert_dict["fsc"],axis=0),
            #                label=m)
            ax[i].plot(assert_dict["pixres"], np.mean(assert_dict["fsc"],axis=0), label=n, color=colors[j])
    ax[i].set_title(f"$SNR=10^{{{noise:g}}}$")
    # ax[i].set_xscale("log")
    ax[i].xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
    ax[i].set_xlabel("Resolution ($1/\AA$)")
    ax[i].set_ylabel("Fourier Shell Correlation")
    ax[i].set_ylim(0,1.05)
    ax[i].axhline(0.143, ls="--", color="grey", alpha=0.5)
    ax[i].legend(loc="upper right")
fig.savefig(prefix+"summary_fsc.png", dpi=300)


fscauc = [[] for i in noise_levels]
fig, ax = plt.subplots(1, len(noise_levels), figsize=(int(len(methods)/4 * len(noise_levels))+8,5), layout="constrained")
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, (m,n) in enumerate(zip(methods, names)):
        if "cryodrgn" in m :
            loc = "analyze.100"
        elif "drgnai" in m:
            loc = 'out/analysis_99'
        else:
            loc = "analysis"
        filename = basedir + "/" + m + "/" +loc  + "/assert.pkl"
        if os.path.isfile(filename):
            assert_dict = load_pkl(filename)
            fscauc[i].append([ np.mean(assert_dict["auc"]), np.std(assert_dict["auc"])])
            ax[i].bar(j,fscauc[i][-1][0],yerr=fscauc[i][-1][1], label=n, error_kw={"capsize":6}, color=colors[j])
    ax[i].set_title(f"$SNR=10^{{{noise:g}}}$")
    ax[i].set_xticks(np.arange(len(methods)))
    ax[i].set_xticklabels(names, rotation=-45,ha='left', rotation_mode='anchor')
    ax[i].set_ylabel("FSC AUC")
    ax[i].set_ylim(0.05,0.35)
fig.savefig(prefix+"summary_fscauc.png", dpi=300)

fscauc =np.array(fscauc)
fig, ax = plt.subplots(1,1, figsize=(10,5), layout="constrained")
for j in range(fscauc.shape[1]):
    ax.errorbar(x=noise_levels, y=fscauc[:, j, 0], yerr=fscauc[:, j, 1],capsize=6, color=colors[j], label=names[j])
ax.legend()
fig.savefig(prefix+"summary_fscauc_noise.png", dpi=300)



fig, ax = plt.subplots(1, len(noise_levels), figsize=(int((len(methods_ff)+1)/4 * len(noise_levels))+10,5), layout="constrained")
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    skip=0
    for j, (m,n) in enumerate(zip(methods, names)):
        if "cryodrgn" in m :
            loc = "analyze.100"
        elif "drgnai" in m:
            loc = 'out/analysis_99'
        else:
            loc = "analysis"
        filename = basedir + "/" + m + "/" +loc  + "/assert.pkl"
        if os.path.isfile(filename):
            assert_dict = load_pkl(filename)
            if "rmsd" in assert_dict:
                ax[i].bar(j-skip, np.mean(assert_dict["rmsd"]),yerr=np.std(assert_dict["rmsd"]), label=n, error_kw={"capsize":6}, color=colors[j])
            else:
                skip+=1
    ax[i].set_title(f"$SNR=10^{{{noise:g}}}$")
    ax[i].set_xticks(np.arange(len(methods_ff) + 1 ))
    ax[i].set_xticklabels([names[0]] + methods_ff, rotation=-45,ha='left', rotation_mode='anchor')
    ax[i].set_ylabel("RMSD ($\AA$)")
    ax[i].set_ylim(0.0,5)
fig.savefig(prefix+"summary_rmsd.png", dpi=300)

