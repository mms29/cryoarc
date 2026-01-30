python scripts/run_pretrained_openfold.py  ../cryofold/cryobench_IgD/new_preds     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/jackhmmer   \
                    --hhblits_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/hhblits          --hmmsearch_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/hmmsearch    \
                          --hmmbuild_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/hmmbuild          --kalign_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:1"  \
                                  --jax_param_path /home/vuillemr/openfold/openfold/resources/params/params_model_1_multimer_v3.npz\
                                          --output_dir data/cryofold/cryobench_IgD/test          --embeddings_output_path data/cryofold/cryobench_IgD/test/test.pt  \
                                              --use_precomputed_alignments data/cryofold/cryobench_IgD/alignments/      --data_random_seed 43 


BASE_DIR="../cryofold/cryobench_IgD/IgG-1D/images/snr0.01"
RUN_DIR=$BASE_DIR/run_target_new

python -u ./flexfold/scripts/train_target.py \
    $BASE_DIR/sorted_particles.128.txt  \
    --poses  $BASE_DIR/particles.pkl\
    --ctf $BASE_DIR/ctf.pkl \
    -n 10000 \
    -o $RUN_DIR \
    --pixel_size 3.0 \
    --sigma 1.05\
    --quality_ratio 5.0 \
    --embedding_path ~/cryofold/cryobench_IgD/pred2/embeddings.pt  \
    --initial_pose_path $BASE_DIR/initial_pose_dummy.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_3_multimer_v3.npz \
    --batch-size 1  \
    --num-workers 0 \
    --zdim 4  \
    --domain real \
    --encode-mode conv \
    --enc-dim 32 \
    --enc-layers 5 \
    --dec-dim 256 \
    --dec-layers 3 \
    --target_file ~/cryofold/cryobench_IgD/1HZH.cif \
    --overwrite \
    --frozen_structure_module\
    --multimer \
    --wd 1e-5 \
    --lr 1e-4 \
    --chi_loss_weight 0.1\
    --viol_loss_weight 0.1\
    --warmup 100 \
    --domain_loss fourier \
    --multimer \
#     --use_lma \


python ./flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose_new \
 --from_aligned_pdb data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/target_fit_new_100.pdb \
 --alignment_reference data//cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_target_new_unfrozen/fit.100.pdb  --overwrite



#*******************CRYODRGN***********************************
BASE_DIR="../cryofold/cryobench_IgD/IgG-1D/images/snr0.01"
RUN_DIR=$BASE_DIR/run_cryodrgn
cryodrgn train_vae     $BASE_DIR/sorted_particles.128.txt  \
    --poses  $BASE_DIR/particles.pkl\
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --batch-size 1024  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \
cryodrgn analyze -o $RUN_DIR/analysis $RUN_DIR 99 --pc 2



BASE_DIR="../cryofold/cryobench_IgD/IgG-1D/images/snr0.01"
RUN_DIR=$BASE_DIR/run2

python ./flexfold/scripts/trajectory_from_model.py $RUN_DIR $RUN_DIR --epoch 20  --batch_size 1 --num_nodes 1 --devices 1
python ./flexfold/eval_vol.py $RUN_DIR/weights.20.pkl -c $RUN_DIR/config.yaml -o $RUN_DIR/analysis/test --zfile  $RUN_DIR/z.txt --no_volume

python ./flexfold/scripts/flexible_backprojection.py  \
    --reference $RUN_DIR/reference.pdb\
     -o $RUN_DIR/backproject_flex/   \
    --coordinates "$RUN_DIR/coordinates.dcd" \
    --coefs $RUN_DIR/coefs.pt    \
    --batch_size 4 --particles $BASE_DIR/sorted_particles.128.txt  --lazy   \
    --poses  $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl  \
    --wiener_constant 1.0 --sigma 1.0 --gaussian_threshold 0.99 --pixel_size 3.0
