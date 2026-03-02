# CryoARC - Cryo-EM Atomic-Resolution Conformations

CryoARC is a continuous heterogeneity analysis method for processing cryo-EM particle images using coevolution.

## Installation

We recommend to install CryoARC using Mamba:
```
mamba env create -f environment.yml
mamba activate cryoarc
```

After the installation, run the following command to install Openfold's dependencies:

```
./install_openfold_dependencies.sh
```


## Usage

### Create sequence embeddings

Start by placing your amino acid sequences in a single file using FASTA format in a fresh directory `your/sequence/dir`. Select the model weight you want to use. For multimer, we recommend to use `model_1_multimer_v3`. You can refer to the weights description in [Openfold](https://openfold.readthedocs.io/en/latest/Inference.html) documentation. Next use the following script to generate embeddings : 

```
python scripts/run_pretrained_openfold.py  your/sequence/dir     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path $CONDA_PREFIX/bin/jackhmmer   \
                    --hhblits_binary_path $CONDA_PREFIX/bin/hhblits          --hmmsearch_binary_path $CONDA_PREFIX/bin/hmmsearch    \
                          --hmmbuild_binary_path $CONDA_PREFIX/bin/hmmbuild          --kalign_binary_path $CONDA_PREFIX/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:0"  \
                                  --jax_param_path resources/params/params_model_1_multimer_v3.npz \
                                          --output_dir your/results/dir          --embeddings_output_path your/results/dir/embeddings.pt
```

This will produce both a PDB/MMCIF file and a `embeddings.pt` file in the `your/results/dir` directory.

### Parsing particle images and metadata

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

### Align embeddings and particles
Next step is to align the sequence embeddings with the particle images. 

```
python scripts/compute_initial_pose.py \
    initial_pose --backproject_path backproject/backproject.mrc \
    --embedding_pdb_path your/results/dir/embeddings.pdb
```

This should produce a `initial_pose.pdb` and `initial_pose.pt`. You can verify that the alignment went well by opening `initial_pose.pdb` and your backprojected volume `backproject/backproject.mrc` in a 3D viewer like ChimeraX and make sure both structures superpose.

### Training CryoARC

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
  --enc-dim 256 \
  --enc-layers 3 \
  --dec-dim 256 \
  --dec-layers 3 \
  --frozen_angle \
  --multimer \
  --lr 8e-5 \
  --chi_loss_weight 0.01\
  --viol_loss_weight 0.01\
  --domain_loss fourier 

```

