for batch_size in 64 256 1024; do
    for learn_rate in 0.5 0.1 0.05 0.01; do
        for re_lambda in 0 0.1 0.01 0.001; do
            python src/mf.py ${batch_size} ${batch_size} ${re_lambda} > log/mf_${batch_size}_${batch_size}_${re_lambda}.log
            python src/bmf.py ${batch_size} ${batch_size} ${re_lambda} > log/bmf_${batch_size}_${batch_size}_${re_lambda}.log
            python src/a_b.py ${batch_size} ${batch_size} ${re_lambda} > log/a_b_${batch_size}_${batch_size}_${re_lambda}.log
            python src/a_b_ab.py ${batch_size} ${batch_size} ${re_lambda} > log/a_b_ab_${batch_size}_${batch_size}_${re_lambda}.log
        done
    done
done

