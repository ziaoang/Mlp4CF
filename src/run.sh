for batch_size in 64 128 256 1024; do
    for learn_rate in 0.5 0.1 0.05 0.01; do
        for re_lambda in 0 0.1 0.01 0.001; do
            if [ ! -f log/mf_${batch_size}_${learn_rate}_${re_lambda}.log ]; then
                python mf.py ${batch_size} ${learn_rate} ${re_lambda} > log/mf_${batch_size}_${learn_rate}_${re_lambda}.log
            fi
            if [ ! -f log/bmf_${batch_size}_${learn_rate}_${re_lambda}.log ]; then
                python bmf.py ${batch_size} ${learn_rate} ${re_lambda} > log/bmf_${batch_size}_${learn_rate}_${re_lambda}.log
            fi
            if [ ! -f log/a_b_${batch_size}_${learn_rate}_${re_lambda}.log ]; then
                python a_b.py ${batch_size} ${learn_rate} ${re_lambda} > log/a_b_${batch_size}_${learn_rate}_${re_lambda}.log
            fi
            if [ ! -f log/a_b_ab_${batch_size}_${learn_rate}_${re_lambda}.log ]; then
                python a_b_ab.py ${batch_size} ${learn_rate} ${re_lambda} > log/a_b_ab_${batch_size}_${learn_rate}_${re_lambda}.log
            fi
        done
    done
done


