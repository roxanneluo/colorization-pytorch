set -x

SIZE="$1"
FLAGS="${@:2}"
FFHQ="dataset/ffhq"

FLAGS="${FLAGS} --loadSize ${SIZE} --fineSize ${SIZE} --data_path ${FFHQ} --display_freq 200 --update_html_freq 1000"

# Train classification network on small training set first
#python train.py --name siggraph_class_small --sample_p 1.0 --niter 100 --niter_decay 0 --classification --phase train_small ${FLAGS}

# Train classification network first
mkdir ./checkpoints/siggraph_class_${SIZE}
#cp ./checkpoints/siggraph_class_small/latest_net_G.pth ./checkpoints/siggraph_class/
#python train.py --name siggraph_class --sample_p 1.0 --niter 15 --niter_decay 0 --classification --load_model --phase train ${FLAGS}
python train.py --name siggraph_class_${SIZE} --sample_p 1.0 --niter 15 --niter_decay 0 --classification --phase train ${FLAGS}


# Train regression model (with color hints)
mkdir ./checkpoints/siggraph_reg_${SIZE}
cp ./checkpoints/siggraph_class_${SIZE}/latest_net_G.pth ./checkpoints/siggraph_reg_${SIZE}/
python train.py --name siggraph_reg_${SIZE} --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --load_model --phase train ${FLAGS}

# Turn down learning rate to 1e-6
mkdir ./checkpoints/siggraph_reg2_${SIZE}
cp ./checkpoints/siggraph_reg_${SIZE}/latest_net_G.pth ./checkpoints/siggraph_reg2_${SIZE}/
python train.py --name siggraph_reg2_${SIZE} --sample_p .125 --niter 5 --niter_decay 0 --lr 0.000001 --load_model --phase train ${FLAGS}
