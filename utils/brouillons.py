import torch
from flexfold.core import struct_to_pdb
from openfold.utils.tensor_utils import tensor_tree_map
import numpy as np 
from openfold.np import residue_constants, protein
from openfold.config import model_config
from openfold.model.model import AlphaFold









struct = torch.load("/home/vuillemr/cryofold/cryobench_IgD/pred2/embeddings.pt")
struct = tensor_tree_map(lambda x: x.detach().cpu().numpy(), struct)
struct_to_pdb(struct,"/home/vuillemr/cryofold/cryobench_IgD/test.pdb" )


config = model_config(
    "model_3_multimer_v3", 
    train=True, 
    low_prec=False,
) 
model = AlphaFold(config)



structure_module = StructureModule(
            is_multimer=is_multimer,
            **self.config["structure_module"],
        )
structure_input = {
    "pair": struct["pair"],
    "single": struct["single"]
}
        # Predict 3D structure
outputs = {}
outputs["sm"] = self.structure_module(
    structure_input,
    embedding_expand["aatype"],
    mask=embedding_expand["seq_mask"],
    inplace_safe=inplace_safe,
    _offload_inference=self.globals.offload_inference,
)
outputs["final_atom_positions"] = atom14_to_atom37(
    outputs["sm"]["positions"][-1], embedding_expand
)
outputs["final_atom_mask"] = embedding_expand["atom37_atom_exists"]
outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]










from cryodrgn import utils, config
z = utils.load_pkl("data/cryofold/particlesSNR1.0/run_fourier_inf_baseline/z.6.pkl")
zdim = z.shape[1]


from cryodrgn import utils, config
from flexfold.models import HetOnlyVAE
from flexfold import dataset




dataset.ImageDataset(
            mrcfile="data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/099_particles_128.mrcs",
            lazy="False",
            norm=None,
            invert_data=False,
            ind=None,
            keepreal=False,
            window=True,
            datadir=None,
            window_r=0.85,
            max_threads=0,
        )


cfg = config.load("data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run/config.yaml")
D = cfg["lattice_args"]["D"]  # image size + 1
zdim = cfg["model_args"]["zdim"]
norm = [float(x) for x in cfg["dataset_args"]["norm"]]
model, lattice = HetOnlyVAE.load(cfg, "data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run/weights.0.pkl", device="cuda:0")
model.eval()







import torch
from openfold.np.residue_constants import restypes
from flexfold.core import output_single_pdb
from Bio.PDB.MMCIFParser import MMCIFParser
embeddings=  torch.load("data/cryofold/spike-md/embeddings.pt", map_location="cuda:0")

parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("pdb", "data/cryofold/spike-md/6VSB.cif")
model = next(structure.get_models())  # usually only 1 model
# Extract all residues (modeled)
modeled = set()
for chain in model:
    if chain.id == "A":
        for res in chain:
            if res.id[0] == " ":  # ignore hetero/water
                modeled.add(res.id[1])  # resseq number

modeled = torch.tensor(list(modeled), device="cuda:0")


resisd = embeddings["residue_index"]

crop = (resisd<=(1146-1)) * (resisd>=(27-1))
mask = torch.tensor([1 if i in modeled else 0 for i in embeddings["residue_index"]], device="cuda:0")


asymid_new = torch.zeros_like(mask)
ones_section = False
ind = 0
for i in range(len(mask)):
    m = mask[i]
    if m == 1:
        if not ones_section:
            ind+=1
        ones_section=True
    else:
        ones_section=False
    asymid_new[i] = ind
    

output_single_pdb(
    embeddings["final_atom_positions"][mask==True].cpu().numpy(),
    embeddings["aatype"][mask==True].cpu().numpy(),
    (mask[..., None]*  embeddings["final_atom_mask"])[mask==True].cpu().numpy(),
    "data/cryofold/spike-md/test.pdb",
    asymid_new[mask==True].cpu().numpy(),
    embeddings["residue_index"][mask==True].cpu().numpy(),

)

output_single_pdb(
    embeddings["final_atom_positions"][crop].cpu().numpy(),
    embeddings["aatype"][crop].cpu().numpy(),
    (embeddings["final_atom_mask"])[crop].cpu().numpy(),
    "data/cryofold/spike-md/test_unmask.pdb",
    embeddings["asym_id"][crop].cpu().numpy(),
    embeddings["residue_index"][crop].cpu().numpy(),

)

crop=mask==True
embeddings_new = {
    "aatype": embeddings["aatype"][crop],
    "seq_mask": mask[crop],
    "pair": embeddings["pair"][crop, :, :][:, crop, :],
    "single":  embeddings["single"][crop, :],
    "final_atom_mask": (mask[..., None]* embeddings["final_atom_mask"])[crop, :],
    "final_atom_positions":embeddings["final_atom_positions"][crop, :,:],
    "residx_atom37_to_atom14":embeddings["residx_atom37_to_atom14"][crop, :],
    "residx_atom14_to_atom37":embeddings["residx_atom14_to_atom37"][crop, :],
    "atom37_atom_exists":embeddings["atom37_atom_exists"][crop, :],
    "atom14_atom_exists":embeddings["atom14_atom_exists"][crop, :],
    "residue_index": embeddings["residue_index"][crop],
    "asym_id":  embeddings["asym_id"][crop],
}

output_single_pdb(
    embeddings_new["final_atom_positions"].cpu().numpy(),
    embeddings_new["aatype"].cpu().numpy(),
    (embeddings_new["final_atom_mask"]).cpu().numpy(),
    "data/cryofold/spike-md/test.pdb",
    embeddings_new["asym_id"].cpu().numpy(),
    embeddings_new["residue_index"].cpu().numpy(),

)

torch.save({k: v.cpu() for k, v in embeddings_new.items()}, "data/cryofold/spike-md/embeddings_crop_mask.pt")


import torch
from openfold.np.residue_constants import restypes
from flexfold.core import output_single_pdb
embeddings=  torch.load("data/cryofold/jillsData/pred2/embeddings.pt")
embeddings_new = {
    "aatype": embeddings["aatype"],
    "seq_mask": embeddings["seq_mask"],
    "pair": embeddings["pair"],
    "single":  embeddings["single"],
    "final_atom_mask": embeddings["final_atom_mask"],
    "final_atom_positions":embeddings["final_atom_positions"],
    "residx_atom37_to_atom14":embeddings["residx_atom37_to_atom14"],
    "residx_atom14_to_atom37":embeddings["residx_atom14_to_atom37"],
    "atom37_atom_exists":embeddings["atom37_atom_exists"],
    "atom14_atom_exists":embeddings["atom14_atom_exists"],
    "residue_index": embeddings["residue_index"],
    "asym_id":  embeddings["asym_id"],
}
torch.save({k: v.cpu() for k, v in embeddings_new.items()}, "data/cryofold/jillsData/pred2/embeddings_fixed.pt")





########################################################################""
import torch 
from flexfold.models import HetOnlyVAE
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt
from cryodrgn.commands_utils.fsc import get_fsc_curve, get_fsc_thresholds,calculate_cryosparc_fscs
from cryodrgn.commands_utils.plot_fsc import create_fsc_plot
from cryodrgn.mrcfile import parse_mrc, write_mrc
import numpy as np
from Bio.PDB import PDBParser, Superimposer, is_aa
from io import StringIO
from openfold.utils.tensor_utils import tensor_tree_map
from flexfold.core import struct_to_pdb
import pickle 

config_file = "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01/run/config.yaml"
weight_file = "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01/run/weights.54.pkl"
z_file =  "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01/run/z.54.pkl"
outfile = "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01/run/out.54.pkl"
##################################################""
# LOAD MODEL and WEIGHTS
##################################################""


def aligned_rmsd(structure1, structure2):
    def get_ca_atoms(structure):
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=False) and "CA" in residue:
                        atoms.append(residue["CA"])
        return atoms

    atoms1 = get_ca_atoms(structure1)
    atoms2 = get_ca_atoms(structure2)

    # Superimpose and calculate RMSD
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)  # This aligns atoms2 onto atoms1
    rmsd = sup.rms
    return rmsd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = config.load(config_file)
D = cfg["lattice_args"]["D"]  # image size + 1
zdim = cfg["model_args"]["zdim"]
norm = [float(x) for x in cfg["dataset_args"]["norm"]]
model, lattice = HetOnlyVAE.load(cfg,weight_file, device=device)
model.eval()
z = load_pkl(z_file)
zdim = z.shape[-1]
parser = PDBParser(QUIET=True)

n_gt = 100
n_expand = 100

outputs = {
    "fsc": [],
    "fscauc": [],
    "fscres": [],
    "rmsd": []
}

for ind in range(n_gt):

    # GT VOLUME
    gt_vol_file = "../cryofold/AKMD/vols/out%s.mrc"%str(ind+1).zfill(6)
    vol,header =parse_mrc(gt_vol_file)
    apix = header.apix

    #GT PDB
    gt_pdb_file =  "../cryofold/AKMD/pdbs/%s_df.pdb"%str(ind+1).zfill(5)
    gt_structure = parser.get_structure("ref", gt_pdb_file)

    for j in range(n_expand):
        i = ind*n_expand +j 
        print("============================== ITER %i ======================================="%i)

        v,s = model.decoder.eval_volume(torch.empty(0).to(device), D-1, None, None,torch.tensor(z[i]).to(device) )

        # Volume FSC
        fsc = get_fsc_curve(v.detach().cpu(), torch.tensor(vol))
        fscauc = np.trapz(np.array(fsc["fsc"]), np.array(fsc["pixres"]))
        fsc_res = get_fsc_thresholds(fsc, apix=apix)
        # create_fsc_plot(fsc, "../cryofold/AKMD/fsc.png", apix)
        # write_mrc("../cryofold/AKMD/test.mrc",v.detach().cpu() )

        # PDB RMSD

        s["final_atom_positions"] = s["final_atom_positions"] @ model.decoder.rot_init + model.decoder.trans_init
        pdb_string = StringIO(struct_to_pdb(tensor_tree_map(lambda x: x.detach().cpu().numpy()[-1], s),"", return_string=True))
        pred_structure = parser.get_structure("mobile",pdb_string)
        rmsd = aligned_rmsd(gt_structure, pred_structure)
        print(rmsd)

        if not "pixres" in outputs : 
            outputs["pixres"] = fsc["pixres"].to_numpy()
        outputs["fsc"].append(fsc["fsc"].to_numpy())
        outputs["fscauc"].append(fscauc)
        outputs["fscres"].append(fsc_res)
        outputs["rmsd"].append(rmsd)

    with open(outfile, "wb") as f:
        pickle.dump(outputs, f)






























import mrcfile
import torch
from flexfold.fsc import fourier_shell_correlation, fsc_auc,fsc_thresh
from cryodrgn.mrcfile import parse_mrc, write_mrc
import time

device = "cuda"

file1 = "../cryofold/AKMD/snr0.01/run/debug_pred.mrc"
file2 = "../cryofold/AKMD/snr0.01/run/debug_gt.mrc"
with mrcfile.open(file1, permissive=True) as mrc:
    vol1 = torch.tensor(mrc.data.copy()  ).to(device)
with mrcfile.open(file2, permissive=True) as mrc:
    vol2 = torch.tensor(mrc.data.copy()  ).to(device)

dt = time.time()
fsc, freqs  = fourier_shell_correlation(vol1, vol2, apix=1.0)
print(time.time()-dt)

file1 = "../cryofold/AKMD/snr0.01/run_cryodrgn/debug_pred.mrc"
file2 = "../cryofold/AKMD/snr0.01/run_cryodrgn/debug_gt.mrc"
with mrcfile.open(file1, permissive=True) as mrc:
    vol1 = torch.tensor(mrc.data.copy()  ).to(device)
with mrcfile.open(file2, permissive=True) as mrc:
    vol2 = torch.tensor(mrc.data.copy()  ).to(device)

dt = time.time()
fsc2, freqs2 = fourier_shell_correlation(vol1, vol2, apix=1.0)
print(time.time()-dt)


auc1 = fsc_auc(fsc, freqs)
auc2 = fsc_auc(fsc2, freqs2)
res_05_1, res_143_1 = fsc_thresh(fsc, freqs)
res_05_2, res_143_2 = fsc_thresh(fsc2, freqs2)

fsc = fsc.cpu().numpy()
fsc2 = fsc2.cpu().numpy()
freqs = freqs.cpu().numpy()
freqs2 = freqs2.cpu().numpy()

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1,1)
ax.plot(freqs, fsc)
ax.plot(freqs, fsc2)
def fraction_formatter(x, pos):
    if x == 0:
        return "0"
    return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places
ax.xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
ax.axhline(0.143, c="red")
ax.axhline(0.5, c="green")
ax.set_xlabel("Resolution ($1/\AA$)")
ax.set_ylabel("Fourier Shell Correlation")
ax.set_ylim(0,1)
fig.savefig("../cryofold/AKMD/snr0.01/run/test.png")


###################################################"
# "    
def get_atom_coords(structure):
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=False) and "CA" in residue:
                    atoms += [a.get_coord() for a in residue]
    return np.array(atoms)
def get_ca_atoms(structure):
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=False) and "CA" in residue:
                    atoms.append(residue["CA"])
    return atoms
def get_rmsd(arr1, arr2):
    return np.sqrt(np.mean(np.square(np.linalg.norm(arr1-arr2, axis=-1))))

def get_coef(structure):
    coef = []
    for atom in structure.get_atoms():
        coef.append(atomdefs[atom.element][0])
    return np.array(coef)

def vol_from_coords(coords, coef):
    crd = torch.tensor(coords)[None]
    pix_loc, pix_mask = get_voxel_mask(crd, grid_size=100, pixel_size=1.0, n_pix_cutoff=15)
    vol = vol_real_mask(crd=crd, pix_loc=pix_loc, pix_mask=pix_mask, grid_size=100, sigma=1.0, pixel_size=1.0, coef=coef)
    return vol[-1]





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
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01",
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005",
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"
]

noise_levels = [-2,-2.5,-3]

methods=["avg",
         "run_cryodrgn_sgd", 
         "run_drgnai",
         "run_conv_sgd"]
names=["Homogeneous",
       "CryoDRGN", 
       "DRGN-AI",
       "Ours"
       ]
prefix = "/home/vuillemr/flexfold/data/cryofold/AKMD/figure_"
methods_ff =[ m for m in names if not ("DRGN" in m or "Homogeneous" in m) ]

colors =   [plt.cm.Greens(v) for v in np.linspace(0.7, 0.8, 1) ] + [plt.cm.Reds(v) for v in np.linspace(0.6, 0.8,2) ] + [plt.cm.Blues(v) for v in np.linspace(0.5, 0.8, len(methods_ff)) ]



##################### 
# COnformatinoal space
#####################

data_pca = []
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, m in enumerate(methods):
        if "drgnai" in m:
            filename = basedir + "/" + m + "/out/conf.99.pkl"
        elif "avg" in m:
            filename="/home/vuillemr/flexfold/data/cryofold/AKMD/pdbs/pca.pkl"
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
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), layout="constrained")  
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, (m,n) in enumerate(zip(methods, names)):
        ind = ncols*i + j

        if data[ind] is not None:
            c = np.repeat(np.arange(100), data[ind].shape[0]//100)
            if data[ind].shape[0] == 100:
                s=20.0
                alpha = 1.0
                n="GT"
                pad = 2
                ax[i,j].set_xlim(data[ind][:,0].min()-pad, data[ind][:,0].max()+pad)
                ax[i,j].set_ylim(data[ind][:,1].min()-pad, data[ind][:,1].max()+pad)
            else:
                s = 10.0
                alpha=0.2

            ax[i,j].scatter(data[ind][:,0],data[ind][:,1], c=c, cmap=cmap, s=s, alpha=alpha)
            ax[i,j].set_title(f"{n} - $SNR=10^{{{noise:g}}}$", fontsize=14)
            ax[i,j].set_xlabel("PC1 ($\AA$)")
            ax[i,j].set_ylabel("PC2 ($\AA$)")
        else:
            print("missing %s "%n)

fig.savefig(prefix+"summary_pca.png", dpi=300)


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

fig, ax = plt.subplots(1, len(noise_levels), figsize=(len(noise_levels)*3.5,2.5), layout="constrained")
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
fig.savefig(prefix+"summary_fsc.svg", dpi=300)


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




from flexfold.scripts.rln2xmp import read_star_multi, write_star_from_dicts, print_summary

stack_file = "particles_1_100K.mrcs"
input_star = "data/cryofold/EMPIAR-12093/particles_1_100K.star"
output_star = "data/cryofold/EMPIAR-12093/particles_1_100K_downsampled.star"


stack_file = "particles_2_100K.mrcs"
input_star = "data/cryofold/EMPIAR-12093/particles_2_100K.star"
output_star = "data/cryofold/EMPIAR-12093/particles_2_100K_downsampled.star"

stack_file = "particles_1.mrcs"
input_star = "data/cryofold/EMPIAR-12093/particles_1.star"
output_star = "data/cryofold/EMPIAR-12093/particles_1_downsampled.star"

stack_file = "particles_2.mrcs"
input_star = "data/cryofold/EMPIAR-12093/particles_2.star"
output_star = "data/cryofold/EMPIAR-12093/particles_2_downsampled.star"

star = read_star_multi(input_star)
print_summary(star)
N = len(star["data_particles"]["_rlnImageName"])
star["data_particles"]["_rlnImageName"] = ["%s@%s"%(str(i+1).zfill(7), stack_file) for i in range(N)]
star["data_optics"]["_rlnImagePixelSize"] = [1.656, 1.656]
star["data_optics"]["_rlnImageSize"] = [200, 200]
write_star_from_dicts(star, output_star)








from flexfold.scripts.rln2xmp import read_star_multi, write_star_from_dicts, print_summary

input_star = "data/cryofold/EMPIAR-10330/FinalRefinement-OriginalParticles-PfCRT.star"
output_star = "data/cryofold/EMPIAR-10330/particles.star"

star = read_star_multi(input_star)
print_summary(star)
N = len(star["data_images"]["_rlnImageName"])
star["data_images"]["_rlnAmplitudeContrast"] = [0.07 for i in range(N)]

write_star_from_dicts(star, output_star)



from flexfold.models import mmcif_feats_from_file, map_sequences
import torch 
import openfold.np.residue_constants as rc
import itertools
import numpy as np

embeddings = torch.load("data/cryofold/EMPIAR-10330/embeddings.pt",map_location="cpu")

target_feats =  mmcif_feats_from_file("data/cryofold/EMPIAR-10330//6UKJ.cif")

def struct_to_seqs(struct):
    return {j.item():"".join([rc.restypes_with_x[i]   for i in struct["aatype"][struct["asym_id"] == j]]) for j in torch.unique(struct["asym_id"])}

def transpose_seqs(seqs, transpose):
    asym_id = t["asym_id"]
    order = np.unique(t["asym_id"])[[i for i in transpose]]
    order_ind = np.concatenate([np.where(asym_id==i)[0] for i in order])
    new_asym_id = np.concatenate([(i+1)*np.ones((asym_id==o).sum()) for i,o in enumerate(order)])

    out = {k:v for k,v in t.items()}
    out["all_atom_positions"] = t["all_atom_positions"][order_ind]
    out["aatype"] = t["aatype"][order_ind]
    out["asym_id"] = new_asym_id
    return out

seq1 = struct_to_seqs(target_feats)
seq2 = struct_to_seqs(embeddings)

len_res = len(target_feats["aatype"])
len_chain = len(seq1)
perms = list(itertools.permutations(range(len_chain)))

matched_seqs_perms = []
for p in perms:
    print("Mapping permutation :", p)
    matched_seqs = 0
    for i, pi in enumerate(p):
        s1 = list(seq1.values())[i]
        s2 = list(seq2.values())[pi]
        mapping = map_sequences(s1, s2) 
        matched = sum([m!=-1 for m in mapping])
        matched_seqs+= matched

    print("N res matched : ", matched_seqs, "/", len_res)
    matched_seqs_perms.append(matched_seqs)

best_perm = np.argmax(matched_seqs_perms)
mappings = []
for i, pi in enumerate(perms[best_perm]):
    s1 = list(seq1.values())[i]
    s2 = list(seq2.values())[pi]
    mapping = map_sequences(s1, s2) 
    mappings.append(mapping)

print("Best sequence alignemnt aligned ", matched_seqs_perms[best_perm],  "/", len_res, "residues")
        tmp = transpose_chains(target_feats, p)
        print((tmp["aatype"] == embeddings["aatype"]))
        if all(tmp["aatype"] == embeddings["aatype"]):
            target_feats = tmp
            break


        mapping = map_sequences(seq1, seq2)
        if -1 in mapping:
            raise NotImplementedError() #FIXME
        target_feats_mapped = {k:v.clone() for k,v in {k2:v2 for k2,v2 in embeddings.items() if k2 in target_keys}.items()}
        target_feats_mapped["all_atom_positions"] = target_feats_mapped["final_atom_positions"]
        del target_feats_mapped["final_atom_positions"]
        for k,v in target_feats_mapped.items():
            v[mapping] = target_feats[k] 
        target_feats = target_feats_mapped












import torch 
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt
from flexfold.fsc import fourier_shell_correlation, fsc_auc,fsc_thresh, spherical_soft_mask, fourier_mask
from cryodrgn.mrcfile import parse_mrc, write_mrc
import numpy as np
from Bio.PDB import PDBParser, Superimposer, is_aa
from io import StringIO
from openfold.utils.tensor_utils import tensor_tree_map
from flexfold.core import struct_to_pdb
import pickle 
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import os
import glob
import time
from scipy.ndimage import shift


def aligned_rmsd(structure1, structure2):
    def get_ca_atoms(structure):
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=False) and "CA" in residue:
                        atoms.append(residue["CA"])
        return atoms

    atoms1 = get_ca_atoms(structure1)
    atoms2 = get_ca_atoms(structure2)

    # Superimpose and calculate RMSD
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)  # This aligns atoms2 onto atoms1
    rmsd = sup.rms
    return rmsd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

snrs = ["snr1","snr0.1","snr0.01","snr0.005","snr0.001"]

for snr in snrs:
    gt_vols = list(glob.glob("/home/vuillemr/flexfold/data/cryofold/AKMD/vols128/*mrc"))
    gt_vols.sort() 
    gt_pdbs = list(glob.glob("/home/vuillemr/flexfold/data/cryofold/AKMD/pdbs/*pdb" ))
    gt_pdbs.sort() 
    outputs = {
        "fsc": [],
        "auc": [],
        "res_05": [],
        "res_0143": [],
        "rmsd" : []
    }


    mask_real = None
    mask_ft = None
    parser = PDBParser(QUIET=True)

    avg_vol,header = parse_mrc("/home/vuillemr/flexfold/data/cryofold/AKMD/%s/backproject/backproject.mrc"%snr)
    avg_struct = parser.get_structure("ref", "/home/vuillemr/flexfold/data/cryofold/AKMD/average_structure.pdb")
    avg_vol =  torch.tensor(avg_vol).to(device)
    n_gt = 100

    for ind in range(n_gt):
        print(ind)

        # GT VOLUME
        vol,header =parse_mrc(gt_vols[ind])
        vol =  torch.tensor(vol).to(device)
        if mask_real is not None:
            vol*= mask_real
        apix = header.apix

        #GT PDB
        gt_structure = parser.get_structure("ref", gt_pdbs[ind])


        # Volume FSC
        if mask_real is not None:
            avg_vol*= mask_real
        fsc_curve,freqs = fourier_shell_correlation(avg_vol,vol,mask_ft, apix=apix)
        auc = fsc_auc(fsc_curve,freqs )
        res_05, res_0143 = fsc_thresh(fsc_curve,freqs )

        rmsd = aligned_rmsd(gt_structure, avg_struct)
        print("RMSD = %.2f Ang"%rmsd)
        if not "pixres" in outputs : 
            outputs["pixres"] = freqs.cpu().numpy()
        outputs["fsc"].append(fsc_curve.cpu().numpy())
        outputs["auc"].append(auc)
        outputs["res_05"].append(res_05)
        outputs["res_0143"].append(res_0143)
        outputs["rmsd"].append(rmsd)

    outdir = "/home/vuillemr/flexfold/data/cryofold/AKMD/%s/avg/analysis"%snr
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    with open(outdir + "/assert.pkl", "wb") as f:
        pickle.dump(outputs, f)













import torch
from openfold.model.primitives import softmax_no_cast, permute_final_dims

def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)
    # a=torch.zeros_like(a_tmp)
    # a[..., mask[...,0], mask[...,1]] = a_tmp[[..., mask[...,0], mask[...,1]]]

    a = softmax_no_cast(a, -1)

    # # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a

def _attention_masked(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask) -> torch.Tensor:
    N,M,_ = mask.shape

    # [B, H, N, M, C]
    query_masked = query [..., mask[..., 0], :]
    key_masked = key [..., mask[..., 1], :]

    # [B, H, N, M]
    a = torch.sum(key_masked*query_masked, dim=-1)

    # [B, H, N, M] , [B, H, N, 1]
    a, a_fill = sparse_softmax(a,N-M, -1)

    # [B, H, N, M, C]
    value_masked = value [..., mask[..., 1], :]

    # [B, H, N, C]
    a = torch.sum(a[..., None] * value_masked, dim=-2)

    # [B, H, 1, C]
    value_sum = value.sum(dim=-2, keepdim=True)
    # [B, H, N, C]
    value_sum_masked = value_masked.sum(dim=-2)

    # [B, H, N, C]
    a =a +  (a_fill*value_sum)  -(a_fill*value_sum_masked)

    return a

def get_diag(N,M):
    half = M // 2  # half window

    # create all row indices
    rows = torch.arange(N).unsqueeze(1).repeat(1, M)  # shape (N, M)

    # create relative offsets for the window
    offsets = torch.arange(-half, half+1)  # e.g. [-2, -1, 0, 1, 2], length M

    # add offsets to rows
    cols = rows + offsets  # shape (N, M)

    # clamp to matrix bounds
    cols = (rows + offsets) % N  # shape (N, M)
    # stack to get (N, M, 2)
    indices = torch.stack([rows, cols], dim=-1)
    return indices

def sparse_softmax(scores, num_masked, dim):
    # scores: (N, M,)
    # pairs: (N, M, 2)

    # Compute max per row for stability
    max_per_row = torch.max(scores, dim=dim, keepdim=True).values
    max_per_row =0.0
    scores_exp = torch.exp(scores - max_per_row)

    # Sum exp over valid connections
    denom = torch.sum(scores_exp, dim=dim, keepdim=True) + num_masked

    return scores_exp / denom, 1/denom

import time

device="cuda"
B = 1
N = 40000
M = 21
H = 4
C = 32


torch.cuda.empty_cache() 
torch.cuda.reset_peak_memory_stats() 
torch.cuda.synchronize()  

mask =get_diag(N,M).to(device)

query = torch.normal(0,1, (B,H, N, C), device=device, dtype=torch.float32, requires_grad=True)
key= torch.normal(0,1, (B,H, N, C), device=device, dtype=torch.float32, requires_grad=True)
value = torch.normal(0,1, (B,H, N, C), device=device, dtype=torch.float32, requires_grad=True)



dt = time.time()
out = _attention(query, key, value, mask)
loss = out.sum()
loss.backward()
print("%.5f s"%(time.time()-dt))
torch.cuda.synchronize()
mem_op1 = torch.cuda.max_memory_allocated()
print(f"Memory used after op1: {mem_op1 / 1e6:.2f} MB")
del out, loss
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

dt = time.time()
out_mask = _attention_masked(query, key, value, mask)
loss = out_mask.sum()
loss.backward()
print("%.5f s"%(time.time()-dt))
torch.cuda.synchronize()
mem_op1 = torch.cuda.max_memory_allocated()
print(f"Memory used after op1: {mem_op1 / 1e6:.2f} MB")
del out_mask, loss
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# print(torch.max(torch.abs((out-out_mask))))











from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

data1 = load_pkl("../cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_new/analysis/assert.pkl")
data2 = load_pkl("../cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_cryodrgn/analyze.100/assert.pkl")

fig, ax = plt.subplots(1, 1, figsize=(10,5), layout="constrained")
ax.errorbar(data1["pixres"], y=np.mean(data1["fsc"],axis=0), yerr=np.std(data1["fsc"],axis=0), label="Ours")
ax.errorbar(data2["pixres"], y=np.mean(data2["fsc"],axis=0), yerr=np.std(data2["fsc"],axis=0), label="CryoDRGN")
def fraction_formatter(x, pos):
    if x == 0:
        return "0"
    return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places
ax.xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
ax.legend()
ax.set_xlabel("Resolution ($1/\AA$)")
ax.set_ylabel("Fourier Shell Correlation")
fig.savefig("../cryofold/cryobench_IgD/IgG-1D/images/snr0.01/summary_fsc.png", dpi=300)
plt.close(fig)























import torch 
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt
from flexfold.fsc import fourier_shell_correlation, fsc_auc,fsc_thresh, spherical_soft_mask, fourier_mask
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from flexfold import dataset
from flexfold.pose import PoseTracker
from flexfold.lattice import Lattice
from cryodrgn import __version__, ctf
from flexfold.core import ifft2_center, unsymmetrize_ht, fft3center, ifft3center, dcd2numpyArr, numpyArr2dcd, get_voxel_mask, vol_real_mask
from cryodrgn.mrcfile import parse_mrc, write_mrc
import torch.nn.functional as F
import argparse
import math
from mpmath import erfinv
import tqdm
import os


volume_half1, header = parse_mrc("/home/vuillemr/cryofold/HER2/data/run/backproject_half1.mrc")
volume_half2, header = parse_mrc("/home/vuillemr/cryofold/HER2/data/run/backproject_half2.mrc")
apix = header.apix

fsc_curve,freqs = fourier_shell_correlation(torch.tensor(volume_half1),
                                            torch.tensor(volume_half2),None, apix=1.1599)
res_05, res_0143 = fsc_thresh(fsc_curve,freqs )

volume_half1_2, header = parse_mrc("/home/vuillemr/cryofold/HER2/data/backproject_real/backproject_half1.mrc")
volume_half2_2, header = parse_mrc("/home/vuillemr/cryofold/HER2/data/backproject_real/backproject_half2.mrc")
apix = header.apix
fsc_curve_2,freqs_2 = fourier_shell_correlation(torch.tensor(volume_half1_2),
                                            torch.tensor(volume_half2_2),None, apix=1.1599)
res_05_2, res_0143_2 = fsc_thresh(fsc_curve_2,freqs_2 )

volume_half1_3, header = parse_mrc("/home/vuillemr/cryofold/HER2/data/backproject_100K/half_map_a.mrc")
volume_half2_3, header = parse_mrc("/home/vuillemr/cryofold/HER2/data/backproject_100K/half_map_b.mrc")
apix = header.apix
fsc_curve_3,freqs_3 = fourier_shell_correlation(torch.tensor(volume_half1_3),
                                            torch.tensor(volume_half2_3),None, apix=1.1599)
res_05_3, res_0143_3 = fsc_thresh(fsc_curve_3,freqs_3 )


label = "Warp ( %.2f $\AA$)" %res_05
label_2 = "Real ( %.2f $\AA$)" %res_05_2
label_3 = "Fourier ( %.2f $\AA$)" %res_05_3

fig, ax = plt.subplots(1,1)
ax.plot(freqs, fsc_curve, label=label)
ax.plot(freqs_2, fsc_curve_2, label=label_2)
ax.plot(freqs_3, fsc_curve_3, label=label_3)
def fraction_formatter(x, pos):
    if x == 0:
        return "0"
    return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places
ax.xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
ax.axhline(0.143, c="red")
ax.axhline(0.5, c="green")
ax.axvline(1/res_05, c="red")
ax.axvline(1/res_0143, c="green")
ax.legend()
ax.set_xlabel("Resolution ($1/\AA$)")
ax.set_ylabel("Fourier Shell Correlation")
ax.set_title("AVG FSC Resolution %.2f $\AA$ (%.2f $\AA$)"%(res_0143, res_05))
fig.savefig("/home/vuillemr/cryofold/HER2/data/run/fsc.png")
plt.close(fig)








































import matplotlib.pyplot as plt
import numpy as np
import torch
from cryodrgn.utils import load_pkl, save_pkl
from umap import UMAP
from sklearn.decomposition import PCA
import os 
import pickle

from flexfold.core import dcd2numpyArr, numpyArr2dcd

prefix = "/home/vuillemr/flexfold/data/cryofold/jillsData/particles/run_pair"
filename = prefix+ "/z.9.pkl"
with open(filename, "rb") as f:
    z_mu = pickle.load(f)
    z_logvar = pickle.load(f)
# dimred = UMAP(n_components=2, n_neighbors=50, min_dist=0.1)
# data_umap.append(dimred.fit_transform(z)[:,:2])
dimred = PCA(n_components=2)
data_pca = dimred.fit_transform(z_mu)[:,:2]
thresh = 10.8
mean_log_var = np.linalg.norm(z_logvar, axis=-1) 
data_pca_var = mean_log_var<thresh

cmap = "jet"

fig, ax = plt.subplots(3,3, figsize=(15,15), layout="constrained")  
ax[0,0].scatter(z_mu[:,0],z_mu[:,1], c=data_pca_var, cmap=cmap, s=.01, alpha=0.5)
ax[0,1].scatter(z_mu[:,0],z_mu[:,2], c=data_pca_var, cmap=cmap, s=.01, alpha=0.5)
ax[0,2].scatter(z_mu[:,0],z_mu[:,3], c=data_pca_var, cmap=cmap, s=.01, alpha=0.5)

ax[1,0].scatter(z_mu[:,1],z_mu[:,0], c=data_pca_var, cmap=cmap, s=.01, alpha=0.5)
ax[1,1].scatter(z_mu[:,1],z_mu[:,2], c=data_pca_var, cmap=cmap, s=.01, alpha=0.5)
ax[1,2].scatter(z_mu[:,1],z_mu[:,3], c=data_pca_var, cmap=cmap, s=.01, alpha=0.5)

ax[2,0].scatter(z_mu[:,2],z_mu[:,0], c=data_pca_var, cmap=cmap, s=.01, alpha=0.5)
ax[2,1].scatter(z_mu[:,2],z_mu[:,1], c=data_pca_var, cmap=cmap, s=.01, alpha=0.5)
ax[2,2].scatter(z_mu[:,2],z_mu[:,3], c=data_pca_var, cmap=cmap, s=.01, alpha=0.5)
fig.savefig(prefix+"/summary_pca.png", dpi=300)


with open(prefix+ "/z.999.pkl", "wb") as f:
    z_mu = pickle.dump(f, z_mu[data_pca_var])
    z_logvar = pickle.dump(f)

import glob
dcd_files = glob.glob("/home/vuillemr/flexfold/data/cryofold/jillsData/particles/run_pair/chunk_*dcd")
idx_files = glob.glob("/home/vuillemr/flexfold/data/cryofold/jillsData/particles/run_pair/chunk_*txt")
out_prefix = "/home/vuillemr/flexfold/data/cryofold/jillsData/particles/run_pair/filtered_"
dcd_files.sort()
idx_files.sort()

for i, (idx_file,dcd_file) in enumerate(zip(idx_files, dcd_files)):
    print(idx_file)
    idx = np.loadtxt(idx_file).astype(int)
    dcd = dcd2numpyArr(dcd_file)
    filter_idx = mean_log_var[idx]< thresh

    np.savetxt(
        out_prefix+"chunk_%i_indices.txt"%i,
        idx[filter_idx]
    )
    numpyArr2dcd(
        dcd[filter_idx],
        out_prefix+"chunk_%i_coordinates.dcd"%i,
    )





import torch
from flexfold.core import struct_to_pdb, output_single_pdb
from openfold.utils.tensor_utils import tensor_tree_map
import numpy as np 
from openfold.np import residue_constants, protein
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.model.structure_module import StructureModule
from flexfold.models import import_weights
import torch.nn as nn
from openfold.utils.feats import (
    atom14_to_atom37,
)
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.structure_module =StructureModule(
            **config
            )
    def forward(self, *args, **kwargs):
        return self.structure_module(*args, **kwargs)


struct = torch.load("/home/vuillemr/cryofold/newmd/2pbi_A_embeddings.pt")
struct_to_pdb( tensor_tree_map(lambda x: x.detach().cpu().numpy(), struct),"/home/vuillemr/cryofold/newmd/test.pdb" )

device="cuda"

config = model_config(
    "finetuning", 
    train=False, 
    low_prec=False,
) 
model = Model(config.model.structure_module).to(device)
model = import_weights(model, "../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt")


structure_input = {
    "pair": struct["pair"],
    "single": struct["single"]
}

outputs = {}
outputs["sm"] =  model(evoformer_output_dict=structure_input, aatype=struct["aatype"])


outputs["final_atom_positions"] = atom14_to_atom37(
    outputs["sm"]["positions"][-1], struct
)
outputs["final_atom_mask"] = struct["atom37_atom_exists"]
outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]


output_single_pdb(
    all_atom_positions= outputs["final_atom_positions"].cpu().detach().numpy(), 
    aatype=struct["aatype"].cpu().detach().numpy(), 
    all_atom_mask=outputs["final_atom_mask"].cpu().detach().numpy(), 
    file="/home/vuillemr/cryofold/newmd/test2.pdb" , 
    chain_index=None, 
    residue_index=struct["residue_index"].cpu().detach().numpy(), 
    b_factors=None
)













import matplotlib.pyplot as plt
import numpy as np
import torch
from cryodrgn.utils import load_pkl, save_pkl
from umap import UMAP
from sklearn.decomposition import PCA
import os 
import pickle
import numpy as np
import glob
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser, Superimposer, is_aa
from Bio.PDB import PDBIO

def get_coordinates(structure):
    coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinates.append(atom.coord)
    return np.array(coordinates)

def set_coordinates(structure, coordinates):
    i=0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                   atom.set_coord(coordinates[i])
                   i+=1
    return structure
prefix = "/home/vuillemr/flexfold/data/cryofold/AKMD/pdbs"
files = glob.glob(prefix+"/*.pdb")
files.sort()
parser = PDBParser(QUIET=True)

coordinates=[]
for f in files:
    print(f)
    structure = parser.get_structure("", f)
    coordinates.append(get_coordinates(structure))

coordinates = np.array(coordinates)

dimred = PCA(n_components=2)
data_pca = dimred.fit_transform(coordinates.reshape((100,-1)))
data_pca /= coordinates.shape[1]**0.5

start, end = np.percentile(data_pca[:, 0], (5, 95))
num_points=10
traj = np.linspace(start, end, num_points)

cmap = "jet"
fig, ax = plt.subplots(1,1, figsize=(15,15), layout="constrained")  
ax.scatter(data_pca[:,0],data_pca[:,1], c=np.arange(100), cmap=cmap, s=100, alpha=0.5)
ax.plot(traj,np.zeros(num_points), "--o", c="black")
fig.savefig(prefix+"/summary_pca.png", dpi=300)


with open("/home/vuillemr/flexfold/data/cryofold/AKMD/pdbs/pca.pkl", "wb") as f:
    import pickle
    pickle.dump(data_pca,f)

traj_coordinates = dimred.inverse_transform(np.stack((traj, np.zeros(num_points))).T).reshape(num_points,-1,3)
traj_coordinates -=  traj_coordinates.mean(axis=(0,1))
io = PDBIO()
for i in range(num_points):
    structure = set_coordinates(structure, traj_coordinates[i])
    io.set_structure(structure)
    io.save(prefix+"/pc1/%s.pdb"%str(i+1).zfill(3))


import torch

weights = torch.load("/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005/run_conv_sgd/checkpoints/epoch=99-step=7605.ckpt")



from flexfold.scripts.rln2xmp import read_star_multi, write_star_from_dicts, print_summary


star = read_star_multi("data/cryofold/AKMD/snr0.005/particles_009356.star")
print_summary(star)

N = len(next(iter(star["data_particles"].values())))
star["data_particles"]["_rlnRandomSubset"] = [1 if i%2 ==0 else 2 for i in range(N)]

write_star_from_dicts(star, "data/cryofold/AKMD/snr0.005/particles_half_maps.star")







############################################################################################
# empiar 10330


import torch
from flexfold.models import get_target_feats
from flexfold.models import mmcif_feats_from_file, map_sequences
import torch 
import openfold.np.residue_constants as rc
import itertools
import numpy as np
from flexfold.core import struct_to_pdb


from openfold.utils.tensor_utils import tensor_tree_map

embeddings = torch.load("data/cryofold/EMPIAR-10330/embeddings.pt",map_location="cpu")

def struct_to_seqs(struct):
    return {j.item():"".join([rc.restypes_with_x[i]   for i in struct["aatype"][struct["asym_id"] == j]]) for j in torch.unique(struct["asym_id"])}

embeddings = torch.load("data/cryofold/EMPIAR-10330/embeddings.pt", map_location="cpu")




embeddings["residue_index"]

target_feats = get_target_feats("data/cryofold/EMPIAR-10330/6uKJ_fixed.pdb", embeddings)
target_feats["asym_id"] = torch.tensor(target_feats["asym_id"])
target_feats["final_atom_positions"] = target_feats["all_atom_positions"]
target_feats["final_atom_mask"] = target_feats["all_atom_mask"]



struct_to_pdb(tensor_tree_map(lambda x: x.detach().cpu().numpy(), embeddings), "data/cryofold/EMPIAR-10330/embeddings.pdb")
struct_to_pdb(tensor_tree_map(lambda x: x.detach().cpu().numpy(), target_feats), "data/cryofold/EMPIAR-10330/target.pdb")

embeddings_seqs = struct_to_seqs(embeddings)
struct_to_seqs(target_feats)


indices = embeddings["asym_id"]!=3
indices = indices | ((embeddings["residue_index"] >= 49 ) &  (embeddings["residue_index"] <= 407) & (embeddings["asym_id"]==3))

embeddings["final_atom_mask"] = embeddings["final_atom_mask"]* indices.unsqueeze(-1)
embeddings["seq_mask"] = embeddings["seq_mask"] *indices


crop = indices

embeddings_new = {
    "aatype": embeddings["aatype"][crop],
    "seq_mask": embeddings["seq_mask"][crop],
    "pair": embeddings["pair"][crop, :, :][:, crop, :],
    "single":  embeddings["single"][crop, :],
    "final_atom_mask": (embeddings["final_atom_mask"])[crop, :],
    "final_atom_positions":embeddings["final_atom_positions"][crop, :,:],
    "residx_atom37_to_atom14":embeddings["residx_atom37_to_atom14"][crop, :],
    "residx_atom14_to_atom37":embeddings["residx_atom14_to_atom37"][crop, :],
    "atom37_atom_exists":embeddings["atom37_atom_exists"][crop, :],
    "atom14_atom_exists":embeddings["atom14_atom_exists"][crop, :],
    "residue_index": embeddings["residue_index"][crop],
    "asym_id":  embeddings["asym_id"][crop],
}

target_feats_new = {
    "aatype": target_feats["aatype"][crop],
    "final_atom_mask": (target_feats["final_atom_mask"])[crop, :],
    "final_atom_positions":target_feats["final_atom_positions"][crop, :,:],
    "residue_index": target_feats["residue_index"][crop],
    "asym_id":  target_feats["asym_id"][crop],
}

torch.save(embeddings_new, "data/cryofold/EMPIAR-10330/embeddings_masked.pt")
struct_to_pdb(tensor_tree_map(lambda x: x.detach().cpu().numpy(), embeddings_new), "data/cryofold/EMPIAR-10330/embeddings_masked.pdb")



###################################################################################################
for f in run2/analysis/z1/vol*pdb; do
  dir=${f%/*}
  base=${f##*/}
  { head -n 229 ss.pdb; cat "$f"; } > "$dir/new_$base"
done


for f in run2/analysis/z2/vol*pdb; do
  dir=${f%/*}
  base=${f##*/}
  { head -n 229 ss.pdb; cat "$f"; } > "$dir/new_$base"
done


for f in run2/analysis/z3/vol*pdb; do
  dir=${f%/*}
  base=${f##*/}
  { head -n 229 ss.pdb; cat "$f"; } > "$dir/new_$base"
done

for f in run2/analysis/z4/vol*pdb; do
  dir=${f%/*}
  base=${f##*/}
  { head -n 229 ss.pdb; cat "$f"; } > "$dir/new_$base"
done


for f in run_new/analysis/z1/vol*pdb; do
  dir=${f%/*}
  base=${f##*/}
  { head -n 137 ss.pdb; cat "$f"; } > "$dir/new_$base"
done

for f in run_new/analysis/z2/vol*pdb; do
  dir=${f%/*}
  base=${f##*/}
  { head -n 137 ss.pdb; cat "$f"; } > "$dir/new_$base"
done


for f in run_new/analysis/z3/vol*pdb; do
  dir=${f%/*}
  base=${f##*/}
  { head -n 137 ss.pdb; cat "$f"; } > "$dir/new_$base"
done


for f in run_new/analysis/z4/vol*pdb; do
  dir=${f%/*}
  base=${f##*/}
  { head -n 137 ss.pdb; cat "$f"; } > "$dir/new_$base"
done


for f in run_new/reference.pdb; do
  dir=${f%/*}
  base=${f##*/}
  { head -n 137 ss.pdb; cat "$f"; } > "$dir/new_$base"
done


########################################"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
res = np.array([214,920 ,1494, 1344 ])
names = ["AdK", "PfCRT", "HER2", "IgG"]
time_ = 1/np.array([1.2, 0.4, 0.24,0.27])
time = 1/np.array([4.3, 2.5, 0.72,1.17])
mem = np.array([2.5, 15.5, 55.5,38])
imsize = np.array([128, 300, 280,128])
plt.rcParams["font.size"] = 14

rmax = 1800
rmin = 100

p = 2.8
fig, ax = plt.subplots(1,2, figsize=(10,4.5), layout="constrained")
xfit = np.linspace(rmin, rmax, 200)

a, c = np.polyfit(res**3, time, 1)
yfit = a * xfit**3 +  c
ax[0].plot(xfit, yfit, "--", color="grey", label="$\mathcal{O}(N_\mathrm{res}^3)$")

ax[0].plot(res,time, "o")
for i in range(4):
    ax[0].annotate(names[i], (50+res[i],time[i]), fontsize=12)
# ax[0].annotate("$x^2$", (1750,5.5), color="grey")
ax[0].set_xlabel("Residue count")
ax[0].set_ylabel("Compute time (s)")


a, c = np.polyfit(res**2, mem, 1)
yfit = a * xfit**2 +  c
ax[1].plot(xfit, yfit, "--", color="grey", label="$\mathcal{O}(N_\mathrm{res}^2)$")


ax[1].plot(res,mem, "o")
for i in range(4):
    ax[1].annotate(names[i], (50+res[i],mem[i]), fontsize=12)
# ax[1].annotate("$x^2$", (1750,5.5), color="grey")
ax[1].set_xlabel("Residue count")
ax[1].set_ylabel("GPU Memory (GB)")
ax[1].set_title("GPU Memory (per particle, per GPU)")
ax[0].set_title("Compute time (per particle, per GPU)")

ax[0].legend(fontsize=14)
ax[1].legend(fontsize=14)
fig.show()
fig.savefig("/home/vuillemr/remote_bettik/openfold_data/data/cryofold/compute_stats.svg")





import numpy as np 

ind = np.concatenate([np.loadtxt("/home/vuillemr/flexfold/data/cryofold/jillsData/particles/run_mlp_new2/chunk_%i_indices.txt"%i) for i in range(3)])
indall = np.arange(392148)
ind4 = []
for i in indall:
    print(i)
    if i not in ind:
        ind4.append(i)


ind4 = np.array(ind4)
ind4.shape
np.savetxt( "/home/vuillemr/flexfold/data/cryofold/jillsData/particles/run_mlp_new2/chunk_3_indices.txt",ind4)

from flexfold.core import ifft2_center, unsymmetrize_ht, fft3center, ifft3center, dcd2numpyArr, numpyArr2dcd

dcd = dcd2numpyArr("/home/vuillemr/flexfold/data/cryofold/jillsData/particles/run_mlp_new2/chunk_3_coordinates.dcd")