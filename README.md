# CryoARC — Cryo-EM Atomic-Resolution Conformations

CryoARC is a continuous heterogeneity analysis framework for cryo-EM particle images that integrates sequence coevolutionary information to model atomic-resolution conformational variability.

---

# Installation

## 1. Create the Conda Environment

We recommend installing CryoARC with **Mamba**:

```bash
mamba env create -f environment.yml
mamba activate cryoarc
```

If you do not have Mamba installed, you can install it through Conda:

```bash
conda install mamba
```

## 2. Install OpenFold Dependencies

After activating the environment, run:
```
./utils/install_openfold_dependencies.sh
```

This installs required dependencies for OpenFold-based embedding generation.

## 3. Download Pretrained Model Parameters

CryoARC uses pretrained Alphafold-based sequence embedding. To download the pre-trained weights, you can use either weights from Deepmind or OpenFold. These parameters are required for generating sequence embeddings.

```
./utils/download_alphafold_params.sh
./utils/download_openfold_params.sh
```
## 4. Sequence Embedding Databases (Optional but Usually Required)

To generate embeddings with MSAs (Multiple Sequence Alignments), you need genetic databases.

You have two options:
### Option A — Provide Your Own MSAs

If you already have MSAs, you can skip database downloads.

### Option B — Download Genetic Databases (~1.5 TB)

⚠️ Requires approximately 1.5 TB of storage.
```
./utils/download_alphafold_params.sh
./utils/download_mgnify.sh
./utils/download_pdb_seqres.sh
./utils/download_uniref90.sh
./utils/download_uniref30.sh
./utils/download_pdb70.sh
```

If you plan to use structural templates, also download PDB mmCIF files:
```
./utils/download_pdb_mmcif.sh
```

---

# Usage

## Workflow Overview

1. Generate sequence embeddings  
2. Parse cryo-EM particle metadata  
3. Align embeddings with density  
4. Train CryoARC  
5. Analyze results

---

## 1. Generate Sequence Embeddings

Place your amino acid sequences in FASTA format inside a new directory `your/sequence/dir`. Next, select the model weight you want to use. For multimer, we recommend to use `model_1_multimer_v3`. You can refer to the weights description available in [Openfold documentation](https://openfold.readthedocs.io/en/latest/Inference.html) . Here is an example script to genereate sequence embeddings : 

```bash
python scripts/run_pretrained_openfold.py your/sequence/dir data/pdb_data/mmcifs/ \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta \
  --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa \
  --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt \
  --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03 \
  --uniprot_database_path data/alignment_data/uniprot/uniprot_trembl.fasta \
  --jackhmmer_binary_path $CONDA_PREFIX/bin/jackhmmer \
  --hhblits_binary_path $CONDA_PREFIX/bin/hhblits \
  --hmmsearch_binary_path $CONDA_PREFIX/bin/hmmsearch \
  --hmmbuild_binary_path $CONDA_PREFIX/bin/hmmbuild \
  --kalign_binary_path $CONDA_PREFIX/bin/kalign \
  --config_preset "model_1_multimer_v3" \
  --model_device "cuda:0" \
  --jax_param_path resources/params/params_model_1_multimer_v3.npz \
  --output_dir your/results/dir \
  --embeddings_output_path your/results/dir/embeddings.pt
```

**Outputs** 
- `embeddings.pt` containing the sequence embeddings
- `embeddings.pdb` the associated structure

## 2. Parsing particle images and metadata

You must provide the particle image, particle alignment and CTF parameters using Relion STAR format.

We use CryoDRGN subroutines for parsing STAR file.
```
cryodrgn parse_ctf_star particles.star -o ctf.pkl
cryodrgn parse_pose_star particles.star -o particles.pkl
```

To make sure the particles are well imported, we recommend to do a rigid backprojection using the following command. 
```
cryodrgn backproject_voxel particles.star --poses particles.pkl --ctf ctf.pkl -o backproject
```

This produce a `backproject/backproject.mrc` density. Open it with a 3D viewer to validate the reconstruction.

## 3. Align embeddings and particles
Align the predicted structure to the backprojected volume:
```
python scripts/compute_initial_pose.py \
    initial_pose --backproject_path backproject/backproject.mrc \
    --embedding_pdb_path your/results/dir/embeddings.pdb
```
**Output**

 - `initial_pose.pdb`

 - `initial_pose.pt`

Validate alignment in a 3D viewer (e.g., ChimeraX).

## 4. Training CryoARC

Below is an example of parameters to train cryoARC
```
python scripts/train.py \
  particles.star  \
  --poses particles.pkl \
  --ctf ctf.pkl \
  --lazy \
  -n N_EPOCHS \
  -o OUTPUT_DIR \
  --pixel_size PIXEL_SIZE \
  --all_atom \
  --embedding_path your/results/dir/embeddings.pt  \
  --initial_pose_path initial_pose.pt \
  --af_checkpoint_path  resources/params/params_model_1_multimer_v3.npz \
  --batch-size 4 \
  --no_blocks_sm 4 \
  --zdim 4  \
  --domain real \
  --encode-mode conv \
  --enc-dim 32 \
  --enc-layers 5 \
  --pair_stack  \
  --dec-dim 32 \
  --dec-layers 4 
  --frozen_angle \
  --multimer \
  --lr 8e-5 \
  --chi_loss_weight 0.01\
  --viol_loss_weight 0.01\
  --domain_loss fourier \
  --do-pose-sgd  \
  --pose-lr 1e-4  \
  --pretrain 0

```
**Output**
- `weights.*.pkl` model weights
- `z.*.pkl` latent variable

## 5. Analyze results

The following command goes through the latent variable at the given `EPOCH` and produces densities, atomic structures and corresponding plots. 
```bash
python scripts/analyze.py -o OUTPUT_DIR/analysis OUTPUT_DIR EPOCH --pc 2
```

## 6. Heterogeneous reconstruction

As a validation you can perform heterogeneous reconstruction using our back-projection algorithm.
The following command produces atomic structures for each latent coordinate.
```bash
python scripts/trajectory_from_model.py OUTPUT_DIR OUTPUT_DIR --epoch EPOCH  --batch_size 16 --num_nodes 1 --devices 4
```
The next command performs the backprojection.
```bash
python scripts/flexible_backprojection.py  --reference OUTPUT_DIR/reference.pdb -o OUTPUT_DIR --chunk   \
#     --coordinates "OUTPUT_DIR/chunk_*_coordinates.dcd" --indices "OUTPUT_DIR/chunk_*_indices.txt" --coefs OUTPUT_DIR/coefs.pt    \
#     --batch_size 16 --particles particles.star --poses particles.pkl --ctf ctf.pkl  \
#     --wiener_constant 1.0 --sigma 1.0 --gaussian_threshold 0.9 --pixel_size PIXEL_SIZE
```
