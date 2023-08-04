#cd /home/dell/leihaozhe/piou/src
# train batchsize range: 0-20
python main.py ctdet_angle --exp_id oban_1024_dla34_epoch240 --dataset oban --input_res 1024 --num_epochs 240 --lr_step 120,200 --gpus 0 --batch_size 5 --wh_weight 0.5 --num_workers 16 --aug_rot 0.5 --rotate 30.0 

# piou
python main.py ctdet_angle --exp_id oban_1024_dla34_epoch240 --dataset oban --input_res 1024 --num_epochs 120 --lr 5e-5 --lr_step 30,60 --gpus 0 --batch_size 5 --wh_weight 0.5 --num_workers 16 --aug_rot 0.5 --rotate 30.0 --piou_weight 1.0 --resume --load_model ../exp/ctdet_angle/oban_1024_dla34_epoch240/model_best.pth --val_intervals 20

# test can't use, need another option
#python test.py --exp_id oban_piou_512 :--not_prefetch_test ctdet --load_model /home/dell/leihaozhe/piou/exp/ctdet_angle/oban_512/model_best.pth
