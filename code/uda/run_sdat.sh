#python image_sdat2.py --gpu_id 0 --seed 2021 --da uda --output ckps/sdat/ --dset office-home --max_epoch 30 --s 0 --bottleneck 2048 --temperature 2.5 --log_results

#python image_sdat.py --gpu_id 0 --seed 2021 --da uda --output ckps/test/ --dset office-home --max_epoch 30 --s 0 --bottleneck 2048 --temperature 2.5 --log_results  --layer linear --classifier bn_relu 

#python image_sdat.py --gpu_id 0 --seed 2021 --da uda --output ckps/sdat2/2 --dset office-home --max_epoch 30 --s 0 --bottleneck 2048 --temperature 2.5 --log_results  --layer linear --classifier bn_relu --t 2

# python target_sdat.py --gpu_id 0 --seed 2021 --da uda --output ckps/sdat2_target2 --dset office-home --s 0 --cls_par 0.3 --ssl 0.6 --output_src ckps/sdat2/2 --bottleneck 2048 --layer linear --classifier bn_relu --t 2

# python image_sdat.py --gpu_id 0 --seed 2021 --da uda --output ckps/sdat2/3 --dset office-home --max_epoch 30 --s 0 --bottleneck 2048 --temperature 2.5 --log_results  --layer linear --classifier bn_relu --t 3

# python target_sdat.py --gpu_id 0 --seed 2021 --da uda --output ckps/sdat2_target3 --dset office-home --s 0 --cls_par 0.3 --ssl 0.6 --output_src ckps/sdat2/3 --bottleneck 2048 --layer linear --classifier bn_relu --t 3\


python image_sdat.py --gpu_id 0 --seed 2021 --da uda --output ckps/sdat/ --dset office-home --max_epoch 30 --s 0 --bottleneck 2048 --temperature 2.5 --log_results  --layer linear --classifier bn_relu 

python target_sdat.py --gpu_id 0 --seed 2021 --da uda --output ckps/sdat_target --dset office-home --s 0 --cls_par 0.3 --output_src ckps/sdat --bottleneck 2048 --layer linear --classifier bn_relu --ssl 0.6