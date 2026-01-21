#Embeddings
python ~/flexfold/flexfold/scripts/run_pretrained_openfold.py  data/cryofold/EMPIAR-10330/     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path /home/vuillemr/.conda/envs/flexfold/bin/jackhmmer   \
                    --hhblits_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hhblits          --hmmsearch_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmsearch    \
                          --hmmbuild_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmbuild          --kalign_binary_path /home/vuillemr/.conda/envs/flexfold/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:0"  \
                                  --jax_param_path /home/vuillemr/openfold/openfold/resources/params/params_model_1_multimer_v3.npz\
                                          --output_dir data/cryofold/EMPIAR-10330/          --embeddings_output_path data/cryofold/EMPIAR-10330/embeddings.pt  \
                                              --use_precomputed_alignments data/cryofold/EMPIAR-10330/alignments       --data_random_seed 45 \
                                                --target_file $BASE_DIR/6UKJ.cif \


BASE_DIR="data/cryofold/EMPIAR-10330/"

cryodrgn parse_ctf_star $BASE_DIR/particles.star -o $BASE_DIR/ctf.pkl --Apix 1.035 -D 300
cryodrgn parse_pose_star $BASE_DIR/particles.star -o $BASE_DIR/particles.pkl --Apix 1.035 -D 300
cryodrgn backproject_voxel $BASE_DIR/particles.star --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject --lazy






python ~/flexfold/flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose --backproject_path $BASE_DIR/backproject/backproject.mrc\
  --embedding_pdb_path $BASE_DIR/predictions/*_unrelaxed.pdb  --overwrite
python ~/flexfold/flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose_fitted --backproject_path $BASE_DIR/backproject/backproject.mrc\
  --embedding_pdb_path $BASE_DIR/run_target/fit.5000.pdb  --overwrite


BASE_DIR=data/cryofold/EMPIAR-10330/
RUN_DIR=$BASE_DIR/run_target3_8

python -u ./flexfold/scripts/train_target.py \
    $BASE_DIR/FinalRefinement-OriginalParticles-PfCRT.mrcs  \
    --poses  $BASE_DIR/particles.pkl\
    --ctf $BASE_DIR/ctf.pkl \
    --lazy \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.035 \
    --sigma 1.05\
    --quality_ratio 5.0 \
    --embedding_path $BASE_DIR/embeddings_masked.pt  \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_1_multimer_v3.npz \
    --batch-size 1  \
    --num-workers 0 \
    --zdim 8  \
    \
    --domain real \
    --encode-mode conv \
    --enc-dim 24 \
    --enc-layers 6 \
    --dec-dim 32 \
    --dec-layers 4 \
    --pair_stack\
    --no_blocks_sm 4\
    --target_file $BASE_DIR/target_masked.pdb \
    --overwrite \
    --frozen_angle\
    --multimer \
    --wd 0 \
    --lr 5e-5 \
    --chi_loss_weight 0.01\
    --viol_loss_weight 0.01\
    --warmup 100 \
    --domain_loss fourier \
    --multimer 


BASE_DIR=data/cryofold/EMPIAR-10330/
RUN_DIR=$BASE_DIR/run_target_mlp

python -u ./flexfold/scripts/train_target.py \
    $BASE_DIR/particles.star  \
    --poses  $BASE_DIR/particles.pkl\
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.035 \
    --sigma 1.05\
    --quality_ratio 5.0 \
    --embedding_path $BASE_DIR/embeddings.pt  \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_1_multimer_v3.npz \
    --batch-size 2  \
    --num-workers 0 \
    --zdim 4  \
    \
    --domain real \
    --encode-mode conv \
    --enc-dim 24 \
    --enc-layers 6 \
    --dec-dim 256 \
    --dec-layers 3 \
    --no_blocks_sm 4\
    --target_file $BASE_DIR/6UKJ.cif \
    --overwrite \
    --frozen_angle\
    --multimer \
    --wd 0 \
    --lr 1e-4 \
    --chi_loss_weight 0.01\
    --viol_loss_weight 0.01\
    --warmup 100 \
    --domain_loss fourier \
    --multimer 


python ~/flexfold/flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose_new --backproject_path $BASE_DIR/backproject/backproject.mrc\
  --embedding_pdb_path $BASE_DIR/run_target3/fit.500.pdb  --overwrite
python ~/flexfold/flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose_new \
 --from_aligned_pdb  $BASE_DIR/initial_pose_new_adjusted.pdb \
 --alignment_reference $BASE_DIR/run_target3/fit.300.pdb   \
  --overwrite

BASE_DIR=data/cryofold/EMPIAR-10330/
RUN_DIR=$BASE_DIR/run_new

python -u ./flexfold/scripts/train.py \
    $BASE_DIR/particles.star  \
    --poses  $BASE_DIR/particles.pkl\
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.035 \
    --sigma 1.05\
    --quality_ratio 5.0 \
    --embedding_path $BASE_DIR/embeddings_masked.pt  \
    --initial_pose_path $BASE_DIR/initial_pose_new.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_1_multimer_v3.npz \
    --batch-size 2  \
    --num-workers 0 \
    --zdim 4  \
    --domain real \
    --encode-mode conv \
    --enc-dim 24 \
    --enc-layers 6 \
    --dec-dim 32 \
    --dec-layers 4 \
    --pair_stack\
    --no_blocks_sm 4\
    --overwrite \
    --frozen_angle\
    --multimer \
    --wd 0 \
    --lr 8e-5 \
    --chi_loss_weight 0.01\
    --viol_loss_weight 0.01\
    --warmup 100 \
    --domain_loss real \
    --multimer \
      --train_val_ratio 0.995\
    --load  $BASE_DIR/run_target3/weights.300.pkl