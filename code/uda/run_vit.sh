# python image_source.py --gpu_id 0 --seed 2021 --da uda --output vit/source/ --dset office-home --max_epoch 100 --s 0 --net vit_base_patch16_224
# python image_target.py --gpu_id 0 --seed 2021 --da uda --output vit/target/ --dset office-home --s 0 --cls_par 0.3 --ssl 0.6 --output_src vit/source/ --net vit_base_patch16_224
python image_mixmatch.py --gpu_id 0 --seed 2021 --da uda --output vit/mixmatch/ --dset office-home --max_epoch 100 --s 0 --output_tar vit/target/ --cls_par 0.3 --ssl 0.6 --choice ent --ps 0.0 --net vit_base_patch16_224

python image_source.py --gpu_id 0 --seed 2021 --da uda --output vit/source/ --dset office-home --max_epoch 100 --s 1 --net vit_base_patch16_224
python image_target.py --gpu_id 0 --seed 2021 --da uda --output vit/target/ --dset office-home --s 1 --cls_par 0.3 --ssl 0.6 --output_src vit/source/ --net vit_base_patch16_224
python image_mixmatch.py --gpu_id 0 --seed 2021 --da uda --output vit/mixmatch/ --dset office-home --max_epoch 100 --s 1 --output_tar vit/target/ --cls_par 0.3 --ssl 0.6 --choice ent --ps 0.0 --net vit_base_patch16_224

python image_source.py --gpu_id 0 --seed 2021 --da uda --output vit/source/ --dset office-home --max_epoch 100 --s 2 --net vit_base_patch16_224
python image_target.py --gpu_id 0 --seed 2021 --da uda --output vit/target/ --dset office-home --s 2 --cls_par 0.3 --ssl 0.6 --output_src vit/source/ --net vit_base_patch16_224
python image_mixmatch.py --gpu_id 0 --seed 2021 --da uda --output vit/mixmatch/ --dset office-home --max_epoch 100 --s 2 --output_tar vit/target/ --cls_par 0.3 --ssl 0.6 --choice ent --ps 0.0 --net vit_base_patch16_224

python image_source.py --gpu_id 0 --seed 2021 --da uda --output vit/source/ --dset office-home --max_epoch 100 --s 3 --net vit_base_patch16_224
python image_target.py --gpu_id 0 --seed 2021 --da uda --output vit/target/ --dset office-home --s 3 --cls_par 0.3 --ssl 0.6 --output_src vit/source/ --net vit_base_patch16_224
python image_mixmatch.py --gpu_id 0 --seed 2021 --da uda --output vit/mixmatch/ --dset office-home --max_epoch 100 --s 3 --output_tar vit/target/ --cls_par 0.3 --ssl 0.6 --choice ent --ps 0.0 --net vit_base_patch16_224
