set -x

SIZE="$1"
BLUR_RADIUS_STDDEV="${2:-0}"
FLAGS="${@:3}"
FFHQ="dataset/ffhq"


function make_suffix() {
    local suf="_${SIZE}"

    if [ ${BLUR_RADIUS_STDDEV} -ne 0 ]; then
        suf="${suf}-G${BLUR_RADIUS_STDDEV}"
    fi
    echo "${suf}"
}


FLAGS="${FLAGS} --loadSize ${SIZE} --fineSize ${SIZE} --data_path ${FFHQ} --display_freq 200 --update_html_freq 1000 --blur_radius_stddev ${BLUR_RADIUS_STDDEV}"
NAME_SUF="$(make_suffix)"


# Train classification network on small training set first
#python train.py --name siggraph_class_small --sample_p 1.0 --niter 100 --niter_decay 0 --classification --phase train_small ${FLAGS}

# Train classification network first
mkdir ./checkpoints/siggraph_class${NAME_SUF}
#cp ./checkpoints/siggraph_class_small/latest_net_G.pth ./checkpoints/siggraph_class/
#python train.py --name siggraph_class --sample_p 1.0 --niter 15 --niter_decay 0 --classification --load_model --phase train ${FLAGS}
python train.py --name siggraph_class${NAME_SUF} --sample_p 1.0 --niter 15 --niter_decay 0 --classification --phase train ${FLAGS}


# Train regression model (with color hints)
mkdir ./checkpoints/siggraph_reg${NAME_SUF}
cp ./checkpoints/siggraph_class${NAME_SUF}/latest_net_G.pth ./checkpoints/siggraph_reg${NAME_SUF}/
python train.py --name siggraph_reg${NAME_SUF} --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --load_model --phase train ${FLAGS}

# Turn down learning rate to 1e-6
mkdir ./checkpoints/siggraph_reg2${NAME_SUF}
cp ./checkpoints/siggraph_reg${NAME_SUF}/latest_net_G.pth ./checkpoints/siggraph_reg2${NAME_SUF}/
python train.py --name siggraph_reg2${NAME_SUF} --sample_p .125 --niter 5 --niter_decay 0 --lr 0.000001 --load_model --phase train ${FLAGS}
