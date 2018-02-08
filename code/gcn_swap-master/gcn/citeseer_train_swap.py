from __future__ import division

import os

order_num_list = [2, 3]
dropout_list = [i / 10 for i in range(10)]

for i in range(100):
    seed_num = i + 100
    for order_num in order_num_list:
        for dropout in dropout_list:
            with open('citeseer_train_simple_output.py', 'r') as f:
                f_s = f.read()
            f_s_l = f_s.split('\n')
            new_seed = "seed = %d"%(seed_num)
            new_order_num = "order_num= %d"%(order_num)
            new_dropout = "dropout = %.1f"%(dropout)
            f_s_l[10] = new_seed
            f_s_l[11] = new_order_num
            f_s_l[12] = new_dropout
            new_file = '\n'.join(f_s_l)
            
            with open('citeseer_train_tmp.py', 'w') as f:
                f.write(new_file)
            os.system('python citeseer_train_tmp.py')
