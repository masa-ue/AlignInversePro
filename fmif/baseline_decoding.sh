# echo "Running TDS..." >> baseline_decoding.txt 2>&1
# for alpha in 0.01 0.1 0.5 1.5 5 15
# # for alpha in 0.1 0.3 0.5 0.7 0.9 1.5 5
# do
#     echo "Running TDS with alpha=$alpha..." >> baseline_decoding.txt 2>&1
#     CUDA_VISIBLE_DEVICES=7 python eval_finetune.py --decoding=tds --tds_alpha=$alpha >> baseline_decoding.txt 2>&1
# done

echo "Running DPS..." >> baseline_decoding.txt 2>&1
for scale in 10 100 1000 3000 10000 30000
# for scale in 100000 300000 500000 1000000
# for scale in 1 100 1000 10000 100000 1000000
do
    echo "Running DPS with scale=$scale..." >> baseline_decoding.txt 2>&1
    CUDA_VISIBLE_DEVICES=7 python eval_finetune.py --decoding=dps --dps_scale=$scale >> baseline_decoding.txt 2>&1
done
