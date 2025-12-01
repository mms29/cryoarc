BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/test"

BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr1"
BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1"
BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01"
BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005"
BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"

RUN_DIR=$BASE_DIR/run


# Convert metadata
cryodrgn parse_ctf_star $BASE_DIR/particles_*.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_*.star -o $BASE_DIR/particles.pkl

# Backproject for verification
cryodrgn backproject_voxel $BASE_DIR/Particles/particles.mrcs --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject

# Cryodrgn
RUN_DIR=$BASE_DIR/run_cryodrgn
cryodrgn train_vae $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --uninvert-data \
    -n 100 \
    -o $RUN_DIR \
    --batch-size 512  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \
    --domain hartley
python ./scripts/assert.py -i $RUN_DIR -o $RUN_DIR --gt_pdbs "$BASE_DIR/../pdbs/*pdb" --gt_vols "$BASE_DIR/../vols128/*mrc" --epoch 99 --drgn --skip 50


#*DIRECT REAL *****************************************************
# python ./scripts/compute_initial_pose.py $BASE_DIR/../initial_pose \
#  --from_aligned_pdb $BASE_DIR/../aligned_embedding.pdb \
#  --alignment_reference ../cryofold/embeddings/4ake_A_embeddings.pdb  --overwrite


RUN_DIR=$BASE_DIR/run_fourier
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --uninvert-data \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.0 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/../initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \
    --domain fourier \
    --overwrite\
    --all_atom \
    --wd 1e-5\
    --lr 1e-4\
    --warmup 100\
    --train_val_ratio 0.995\
    --domain_loss fourier


python ./scripts/analyze.py  -o $RUN_DIR/analysis $RUN_DIR 99 --pc 2 
python ./scripts/assert.py -i $RUN_DIR -o $RUN_DIR --gt_pdbs "$BASE_DIR/../pdbs/*pdb" --gt_vols "$BASE_DIR/../vols128/*mrc" --epoch 99

python ./scripts/assert.py -i $RUN_DIR -o $RUN_DIR/debug --gt_pdbs "$BASE_DIR/../pdbs/*pdb" --gt_vols "$BASE_DIR/../vols/*mrc" --epoch 99 --debug

RUN_DIR=$BASE_DIR/run_conv
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --uninvert-data \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.0 \
    --sigma 1.0 \
    --quality_ratio 5.0 \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/../initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 32 \
    --enc-layers 5 \
    --dec-dim 256 \
    --dec-layers 3 \
    --domain real \
    --encode-mode conv \
    --overwrite\
    --all_atom \
    --wd 1e-5\
    --lr 1e-4\
    --warmup 100\
    --train_val_ratio 0.995



RUN_DIR=$BASE_DIR/run_conv_pair
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.0 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/../initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 32 \
    --enc-layers 5 \
    --dec-dim 32 \
    --dec-layers 4 \
    --domain real \
    --encode-mode conv \
    --overwrite\
    --all_atom \
    --pair_stack\
    --wd 1e-4\
    --lr 5e-4\
    --warmup 100\
    --train_val_ratio 0.99
    
    --lora_structure_module\



RUN_DIR=$BASE_DIR/run_test

python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs \
 --poses $BASE_DIR/particles.pkl  \
 --uninvert-data  \
 -n 100 \
 -o $RUN_DIR  \
 --pixel_size 1.0 \
 --sigma 1.0 \
 --quality_ratio 5.0 \
 --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
 --initial_pose_path $BASE_DIR/../initial_pose.pt \
  --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt\
  --batch-size 64\
  --num-workers 0 \
  --zdim 4\
  --enc-dim 32\
  --enc-layers 5\
  --dec-dim 256\
  --dec-layers 3\
  --domain real\
  --encode-mode conv\
  --overwrite\
  --all_atom\
  --wd 1e-5\
  --lr 1e-4\
  --warmup 100\
  --train_val_ratio 0.995\
  --domain_loss fourier\
  --do-pose-sgd\
  --pretrain 1\
  --pose-lr 5e-4\
  --pose-wd 1e-3\
  --frozen_angle\
  --viol_loss_weight 0.1\
  --chi_loss_weight 0.1\


BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr1"


conda activate drgnai
RUN_DIR=$BASE_DIR/run_drgnai
drgnai setup $RUN_DIR --particles $BASE_DIR/Particles/particles.mrcs  \
    --pose $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    --capture-setup spa\
    --reconstruction-type het \
    --pose-estimation refine \
    --conf-estimation autodecoder
drgnai train $RUN_DIR

conda activate flexfold
python ./flexfold/scripts/drgnai_assert.py -i $RUN_DIR/out -o $RUN_DIR --gt_pdbs "$BASE_DIR/../pdbs/*pdb" --gt_vols "$BASE_DIR/../vols128/*mrc" --epoch 99



##############

python ./flexfold/scripts/wrapped_backprojection.py \
 -i /home/vuillemr/flexfold/data/cryofold/AKMD/snr1/run/ -o /home/vuillemr/flexfold/data/cryofold/AKMD/snr1/backproject_test\
 --batch_size 16 --particles /home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005/Particles/particles.mrcs \
  --poses  /home/vuillemr/flexfold/data/cryofold/AKMD/snr1/particles.pkl --ctf /home/vuillemr/flexfold/data/cryofold/AKMD/snr1/ctf.pkl \
  --wiener_constant 1.0 --use_warp 
  
python ./flexfold/scripts/flexible_backprojection.py \
 -i /home/vuillemr/flexfold/data/cryofold/AKMD/snr1/run/ -o /home/vuillemr/flexfold/data/cryofold/AKMD/snr1/backproject_test\
 --batch_size 16 --particles /home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005/Particles/particles.mrcs \
  --poses  /home/vuillemr/flexfold/data/cryofold/AKMD/snr1/particles.pkl --ctf /home/vuillemr/flexfold/data/cryofold/AKMD/snr1/ctf.pkl \
  --wiener_constant 1.0  --sigma 4.0 --gaussian_threshold 0.80






BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr1"
RUN_DIR=$BASE_DIR/run_dynamight

dynamight optimize-deformations  --refinement-star-file $BASE_DIR/particles_006740.star  --output-directory $RUN_DIR --initial-model $BASE_DIR/backproject/backproject.mrc

dynamight optimize-inverse-deformations $RUN_DIR --checkpoint-file $RUN_DIR/forward_deformations/checkpoints/150.pth

dynamight deformable-backprojection $RUN_DIR  --vae-directory  $RUN_DIR/forward_deformations/checkpoints/150.pth