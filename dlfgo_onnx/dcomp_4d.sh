# 배열 선언 (괄호와 공백 필요)
dataset=("truck")
decomp=("4d")
levels=("1")

# 반복문
for l in "${levels[@]}"; do
    for dc in "${decomp[@]}"; do
        for ds in "${dataset[@]}"; do
            echo "decomp: $dc, dataset: $ds, level: $l"
            python run.py --datadir "/data/igjeong/dataset/stanford_half/$ds" \
                            --dataset_name stanford \
                            --render_test \
                            --decomp "$dc" --levels "$l" --interp_mode 16 \
                            --gpuid 0 --epoch 100 \
                            --expname "$ds"_"$dc" \
                            --grid_dim 16 \
                            --mlp_depth 4 \
                            --mlp_width 128 \
                            --grid_num 8 8 128 64 \
                            --dump_images
            sleep 3
        done
    done
done
