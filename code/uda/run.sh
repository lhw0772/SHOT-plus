python image_source.py --gpu_id 0 --seed 2021 --da uda --output ckps/source/ --dset office-home --max_epoch 100 --s 0
python image_target.py --gpu_id 0 --seed 2021 --da uda --output ckps/target/ --dset office-home --s 0 --cls_par 0.3 --ssl 0.6 --output_src ckps/source/
python image_mixmatch.py --gpu_id 0 --seed 2021 --da uda --output ckps/mixmatch/ --dset office-home --max_epoch 100 --s 0 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --choice ent --ps 0.0

python image_source.py --gpu_id 0 --seed 2021 --da uda --output ckps/source/ --dset office-home --max_epoch 100 --s 1
python image_target.py --gpu_id 0 --seed 2021 --da uda --output ckps/target/ --dset office-home --s 1 --cls_par 0.3 --ssl 0.6 --output_src ckps/source/
python image_mixmatch.py --gpu_id 0 --seed 2021 --da uda --output ckps/mixmatch/ --dset office-home --max_epoch 100 --s 1 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --choice ent --ps 0.0

python image_source.py --gpu_id 0 --seed 2021 --da uda --output ckps/source/ --dset office-home --max_epoch 100 --s 2
python image_target.py --gpu_id 0 --seed 2021 --da uda --output ckps/target/ --dset office-home --s 2 --cls_par 0.3 --ssl 0.6 --output_src ckps/source/
python image_mixmatch.py --gpu_id 0 --seed 2021 --da uda --output ckps/mixmatch/ --dset office-home --max_epoch 100 --s 2 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --choice ent --ps 0.0

python image_source.py --gpu_id 0 --seed 2021 --da uda --output ckps/source/ --dset office-home --max_epoch 100 --s 3
python image_target.py --gpu_id 0 --seed 2021 --da uda --output ckps/target/ --dset office-home --s 3 --cls_par 0.3 --ssl 0.6 --output_src ckps/source/
python image_mixmatch.py --gpu_id 0 --seed 2021 --da uda --output ckps/mixmatch/ --dset office-home --max_epoch 100 --s 3 --output_tar ckps/target/ --cls_par 0.3 --ssl 0.6 --choice ent --ps 0.0

