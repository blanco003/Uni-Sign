
@echo off
set output_dir=out\stage2_pretraining

set ckpt_path=out\stage1_pretraining\best_checkpoint.pth

python pre_training.py ^
   --batch-size 4 ^
   --gradient-accumulation-steps 8 ^
   --epochs 5 ^
   --opt AdamW ^
   --lr 3e-4 ^
   --quick_break 2048 ^
   --output_dir $output_dir ^
   --finetune $ckpt_path ^
   --dataset CSL_News ^
   --rgb_support
