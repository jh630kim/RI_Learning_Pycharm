import tensorflow as tf


model_name = 'model_1'

'''
def weight(name, shape, w_type = 0):
    if w_type == 1:
        w = tf.get_variable(name=name, shape=shape,
                            initializer=tf.contrib.layers.xavier_initializer())
    elif w_type == 2:
        pass
    else:
        w = tf.Variable(tf.random_normal(shape, stddev=0.01))
    return w


def bias(name, shape, b_type = 0):
    if b_type == 1:
        b = tf.Variable(tf.random_normal(shape=shape), name=name)
    elif b_type == 2:
        pass
    else:
        b = tf.Variable(tf.constant(0.1, shape=shape), name=name)
    return b
'''

''' ============================= '''
''' 변수 설정 '''
''' ============================= '''
# 출력
# nb_classes = cfg.nb_classes  #0, R1, R2, S3, L4, L5
input_size = 16     # env.observation_space.n   # 16
output_size = 4     # env.action_space.n       # 4
# Learning
learning_rate = 0.1

# 입출력
# X = image수 x 가로 이미지 x 세로 이미지 x 색상
X = tf.placeholder(shape=[1, input_size],
                   dtype=tf.float32,
                   name='X')
Y = tf.placeholder(shape=[1, output_size],
                   dtype=tf.float32,
                   name='Y')    # Y Label

''' ============================= '''
''' 모델 설정 '''
''' ============================= '''
# (1) layer 1의 scope 지정
with tf.name_scope("Layer_1") as scope:
    W1 = tf.Variable(tf.random_uniform(shape=[input_size, 32],
                                       minval=0,
                                       maxval=0.01))
    b1 = tf.Variable(tf.random_uniform(shape=[32],
                                       minval=-1.0,
                                       maxval=1.0))
    L1 = tf.nn.relu(tf.matmul(X, W1)) + b1

# (1) layer 2의 scope 지정
with tf.name_scope("Layer_2") as scope:
    W2 = tf.Variable(tf.random_uniform(shape=[32, 32],
                                       minval=0,
                                       maxval=0.01))
    b2 = tf.Variable(tf.random_uniform(shape=[32],
                                       minval=-1.0,
                                       maxval=1.0))
    L2 = tf.nn.relu(tf.matmul(L1, W2)) + b2

# (1) layer 3의 scope 지정
with tf.name_scope("Layer_3") as scope:
    W3 = tf.Variable(tf.random_uniform(shape=[32, output_size],
                                       minval=0,
                                       maxval=0.01))
    b3 = tf.Variable(tf.random_uniform(shape=[output_size],
                                       minval=-1.0,
                                       maxval=1.0))
    L3 = tf.nn.sigmoid(tf.matmul(L2, W3)) + b3    # Q_Pred = Logits,


# 마지막 Layer...
Y_ = L3   # Logits

''' ============================= '''
''' cost=loss, train, accuracy 설정 '''
''' ============================= '''
cost = tf.reduce_sum(tf.square(Y - Y_))     # Y - Q_Pred
train = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(cost)

''' 정확도 측정 '''
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# create a summary to monitor cost tensor
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)
# tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()
logData = './logs'
summary_writer = tf.summary.FileWriter(logData,
                                       graph=tf.get_default_graph())

print('---------------------------------')
print('### Model Read Completed. ###')
print('---------------------------------')
