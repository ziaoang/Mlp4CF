
for dataset in ml-10m ml-20m netflix; do
    for batch_size in 128; do
        for learn_rate in 0.1; do
            for re_lambda in 0.01; do
                if [ ! -f log/a_b_${dataset}_${batch_size}_${learn_rate}_${re_lambda}.log ]; then
                    python a_b.py ${dataset} ${batch_size} ${learn_rate} ${re_lambda} > log/a_b_${dataset}_${batch_size}_${learn_rate}_${re_lambda}.log
                fi
                if [ ! -f log/a_b_ab_${dataset}_${batch_size}_${learn_rate}_${re_lambda}.log ]; then
                    python a_b_ab.py ${dataset} ${batch_size} ${learn_rate} ${re_lambda} > log/a_b_ab_${dataset}_${batch_size}_${learn_rate}_${re_lambda}.log
                fi
            done
        done
    done
done


