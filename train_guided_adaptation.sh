CUDA_VISIBLE_DEVICES=0 python train_guided_adaptation.py --cls chair --save_interval 5 --data_folder /viscam/data/ShapeNet.v1_OccNetPC/ShapeNet --num_workers 8 |& tee logs/GA_chair.txt
