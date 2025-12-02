


BASE_DIR="/home/vuillemr/flexfold/data/cryofold/EMPIAR-12093/"

python ~/flexfold/flexfold/scripts/rln2xmp.py $BASE_DIR/particles_1_100K.star $BASE_DIR/particles_1_100K.xmd
python ~/flexfold/flexfold/scripts/rln2xmp.py $BASE_DIR/particles_2_100K.star $BASE_DIR/particles_2_100K.xmd
python ~/flexfold/flexfold/scripts/rln2xmp.py $BASE_DIR/particles_1.star $BASE_DIR/particles_1.xmd
python ~/flexfold/flexfold/scripts/rln2xmp.py $BASE_DIR/particles_2.star $BASE_DIR/particles_2.xmd

xmipp_image_resize -i $BASE_DIR/particles_1.xmd --dim 200 -o $BASE_DIR/particles_1.mrcs --save_metadata_stack $BASE_DIR/particles_1_downsampled.xmd --keep_input_columns &

xmipp_image_resize -i $BASE_DIR/particles_2.xmd --dim 200 -o $BASE_DIR/particles_2.mrcs --save_metadata_stack $BASE_DIR/particles_2_downsampled.xmd --keep_input_columns &

xmipp_image_resize -i $BASE_DIR/particles_2_100K.xmd --dim 200 -o $BASE_DIR/particles_2_100K.mrcs --save_metadata_stack $BASE_DIR/particles_2_100K_downsampled.xmd --keep_input_columns &


cryodrgn parse_ctf_star $BASE_DIR/particles_1_100K_downsampled.star -o $BASE_DIR/ctf_1_100K.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_1_100K_downsampled.star -o $BASE_DIR/particles_1_100K.pkl
cryodrgn backproject_voxel $BASE_DIR/particles_1_100K_downsampled.star --poses $BASE_DIR/particles_1_100K.pkl --ctf $BASE_DIR/ctf_1_100K.pkl -o $BASE_DIR/backproject_1_100K --lazy

cryodrgn parse_ctf_star $BASE_DIR/particles_2_100K_downsampled.star -o $BASE_DIR/ctf_2_100K.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_2_100K_downsampled.star -o $BASE_DIR/particles_2_100K.pkl
cryodrgn backproject_voxel $BASE_DIR/particles_2_100K.mrcs --poses $BASE_DIR/particles_2_100K.pkl --ctf $BASE_DIR/ctf_2_100K.pkl -o $BASE_DIR/backproject_2_100K

cryodrgn parse_ctf_star $BASE_DIR/particles_1_downsampled.star -o $BASE_DIR/ctf_1.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_1_downsampled.star -o $BASE_DIR/particles_1.pkl
cryodrgn backproject_voxel $BASE_DIR/particles_1.mrcs --poses $BASE_DIR/particles_1.pkl --ctf $BASE_DIR/ctf_1.pkl -o $BASE_DIR/backproject_1 --first 10000 --lazy

cryodrgn parse_ctf_star $BASE_DIR/particles_2_downsampled.star -o $BASE_DIR/ctf_2.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_2_downsampled.star -o $BASE_DIR/particles_2.pkl
cryodrgn backproject_voxel $BASE_DIR/particles_2.mrcs --poses $BASE_DIR/particles_2.pkl --ctf $BASE_DIR/ctf_2.pkl -o $BASE_DIR/backproject_2 --first 10000 --lazy

python ~/flexfold/flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose --backproject_path $BASE_DIR/backproject_1_100K/backproject.mrc\
  --embedding_pdb_path $BASE_DIR/predictions/*_unrelaxed.pdb  --overwrite

python ~/flexfold/flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose \
 --from_aligned_pdb  $BASE_DIR/aligned_1.pdb \
 --alignment_reference  $BASE_DIR/predictions/*_unrelaxed.pdb  --overwrite

#Embeddings
python ~/flexfold/flexfold/scripts/run_pretrained_openfold.py  data/cryofold/EMPIAR-12093/     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path /home/vuillemr/.conda/envs/flexfold/bin/jackhmmer   \
                    --hhblits_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hhblits          --hmmsearch_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmsearch    \
                          --hmmbuild_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmbuild          --kalign_binary_path /home/vuillemr/.conda/envs/flexfold/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:0"  \
                                  --jax_param_path /home/vuillemr/openfold/openfold/resources/params/params_model_1_multimer_v3.npz\
                                          --output_dir data/cryofold/EMPIAR-12093/          --embeddings_output_path data/cryofold/EMPIAR-12093//embeddings.pt  \
                                            #   --use_precomputed_alignments data/cryofold/jillsData/pred2/alignments       --data_random_seed 43 

BASE_DIR="/home/vuillemr/flexfold/data/cryofold/EMPIAR-12093/"
RUN_DIR=$BASE_DIR/run

python -u ./flexfold/scripts/train.py \
    $BASE_DIR/particles_1_100K_downsampled.star  \
    --poses  $BASE_DIR/particles_1_100K.pkl --ctf $BASE_DIR/ctf_1_100K.pkl --lazy \
    -n 100 --pixel_size 1.656 --sigma 1.05 --all_atom --quality_ratio 5.0 \
    -o $RUN_DIR --embedding_path $BASE_DIR/embeddings.pt  \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_1_multimer_v3.npz \
    --batch-size 1  \
    --no_blocks_sm 4\
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \
    --overwrite \
    --frozen_angle\
    --multimer \
    --wd 0 \
    --lr 8e-5 \
    --chi_loss_weight 0.01\
    --viol_loss_weight 0.01\
    --warmup 100 \
    --domain_loss real \
        --train_val_ratio 0.999\
        --debug\
        --mpi_plugin\
        --num_nodes 3\
        --devices 4\




python ./flexfold/scripts/analyze.py -o $RUN_DIR/analysis $RUN_DIR 27 --pc 2

python ./flexfold/scripts/trajectory_from_model.py ../cryofold/HER2/data/run2/ ../cryofold/HER2/data/run2/ --epoch 2  --batch_size 4 --num_nodes 1 --devices 4
python ./flexfold/scripts/flexible_backprojection.py -i ../cryofold/HER2/data/run/ -o ../cryofold/HER2/data/backproject_flex/ \
 --batch_size 4 --particles ../cryofold/HER2/data/particles_100K.star \
  --poses  ../cryofold/HER2/data/particles_100K.pkl --ctf ../cryofold/HER2/data/ctf_100K.pkl \
  --wiener_constant 1.0 --sigma 2.0 --gaussian_threshold 0.90

