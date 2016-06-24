import tensorflow as tf
from load import load
import random
import sys

try:
    batch_size = int(sys.argv[1])
    learn_rate = float(sys.argv[2])
    re_lambda  = float(sys.argv[3])
except:
    print("batch_size learn_rate re_lambda")
    exit()

print("parameter list:")
print("batch size:\t%d"%batch_size)
print("learn rate:\t%.4f"%learn_rate)
print("regular lambda:\t%.4f"%re_lambda)
print("="*20)

k = 10
epoch_count = 100
dataset = "ml-1m"

# load data
train_set, test_set = load(dataset)
user_count, item_count  = 0, 0
for t in train_set:
    user_count = max(user_count, t[0])
    item_count = max(item_count, t[1])
user_count += 1
item_count += 1

# matrix factorization
u = tf.placeholder(tf.int32,   shape=[None])
v = tf.placeholder(tf.int32,   shape=[None])
r = tf.placeholder(tf.float32, shape=[None])

U = tf.Variable(tf.random_uniform([user_count, k], -0.05, 0.05))
V = tf.Variable(tf.random_uniform([item_count, k], -0.05, 0.05))
P = tf.Variable(tf.random_uniform([user_count], -0.05, 0.05))
Q = tf.Variable(tf.random_uniform([item_count], -0.05, 0.05))

u_factor = tf.gather(U, u)
v_factor = tf.gather(V, v)
u_bias = tf.gather(P, u)
v_bias = tf.gather(Q, v)
g_bias = tf.Variable(tf.random_uniform([1], -0.05, 0.05))

y = tf.reduce_sum(u_factor * v_factor, 1) + u_bias + v_bias + g_bias
rmse = tf.sqrt(tf.reduce_mean(tf.square(r - y)))
mae  = tf.reduce_mean(tf.abs(r - y))

# loss function
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

re_u_factor = tf.reduce_sum(tf.square(u_factor), 1)
re_v_factor = tf.reduce_sum(tf.square(v_factor), 1)
loss = tf.reduce_mean(tf.square(r - y) + re_lambda * (re_u_factor + re_v_factor))
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

# iterator
random.seed(123456789)
rmse_score_list = []
mae_score_list = []
for epoch in range(epoch_count):
    random.shuffle(train_set)

    # train
    for batch_id in range( len(train_set) / batch_size ):
        start = batch_id * batch_size
        end = start + batch_size
        
        batch_u, batch_v, batch_r = [], [], []
        for i in range(start, end):
            t = train_set[i]
            batch_u.append(t[0])
            batch_v.append(t[1])
            batch_r.append(t[2])
        
        train_step.run(feed_dict={u:batch_u, v:batch_v, r:batch_r})

    # predict
    test_u, test_v, test_r = [], [], []
    for t in test_set:
        test_u.append(t[0])
        test_v.append(t[1])
        test_r.append(t[2])

    rmse_score = rmse.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    mae_score = mae.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    rmse_score_list.append(rmse_score)
    mae_score_list.append(mae_score)
    print("%.4f"%rmse_score)

min_rmse = min(rmse_score_list)
min_mae = min(mae_score_list)
print("%d\t%.4f\t%.4f\t%.4f\t%.4f"%(batch_size, learn_rate, re_lambda, min_rmse, min_mae))



