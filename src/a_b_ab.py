import tensorflow as tf
from load import load
import random
import math
import sys

try:
    dataset    = sys.argv[1]
    batch_size = int(sys.argv[2])
    learn_rate = float(sys.argv[3])
    re_lambda  = float(sys.argv[4])
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

u_factor = tf.gather(U, u)
v_factor = tf.gather(V, v)

merge = tf.concat([u_factor, v_factor, u_factor * v_factor], 1)

size1 = 3 * k
size2 = 3 * k / 2
size3 = 1
scale12 = math.sqrt(6.0 / (size1 + size2)) 
scale23 = math.sqrt(6.0 / (size2 + size3)) 

W1 = tf.Variable(tf.random_uniform([size1, size2], -scale12, scale12))
b1 = tf.Variable(tf.random_uniform([size2],        -scale12, scale12))
W2 = tf.Variable(tf.random_uniform([size2, size3], -scale23, scale23))
b2 = tf.Variable(tf.random_uniform([size3],        -scale23, scale23))

y1 = tf.sigmoid(tf.matmul(merge, W1) + b1) 
y2  = tf.matmul(y1, W2) + b2

y = tf.reshape(y2, [-1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(r - y)))
mae  = tf.reduce_mean(tf.abs(r - y))

# loss function
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

re_u_factor = tf.reduce_sum(tf.square(u_factor), 1)
re_v_factor = tf.reduce_sum(tf.square(v_factor), 1)
loss = tf.reduce_mean(tf.square(r - y) + re_lambda * (re_u_factor + re_v_factor))
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

# iterator
random.seed(123456789)
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
    print("%.4f\t%.4f"%(rmse_score, mae_score))


