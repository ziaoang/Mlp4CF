import os
import sys

df = open("result.txt", "w")
for method in ['mf', 'bmf', 'a_b', 'a_b_ab']:
    for batch_size in ['64', '256', '1024']:
        for learn_rate in ['0.5', '0.1', '0.05', '0.01']:
            for re_lambda in ['0', '0.1', '0.01', '0.001']:
                file_path = "log/%s_%s_%s_%s.log"%(method, batch_size, learn_rate, re_lambda)
                rmse_list, mae_list = [], []
                line_no = 0
                for line in open(file_path):
                    line_no += 1
                    if line_no <= 5:
                        continue
                    t = line.strip().split('\t')
                    rmse = float(t[0])
                    mae = float(t[1])
                    rmse_list.append(rmse)
                    mae_list.append(mae)
                min_rmse = min(rmse_list)
                min_mae = min(mae_list)
                df.write("%s\t%s\t%s\t%s\t%.4f\t%.4f\n"%(method, batch_size, learn_rate, re_lambda, min_rmse, min_mae))
df.close()



