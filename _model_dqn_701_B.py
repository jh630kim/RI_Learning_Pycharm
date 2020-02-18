import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, session: tf.Session, input_size: int, output_size: int, name='main') -> None:
        """DQN Agent can

        1) Build network
        2) Predict Q_value given state
        3) Train parameters

        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=10, l_rate=0.001) -> None:
        """DQN Network architecture (simple MLP)

        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """
        with tf.variable_scope(self.net_name):
            # Input
            self._X = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.input_size],
                                     name='input_X')

            # Sung Kim 강의(Lab 7-2)
            # Layer 1
            self.W1 = tf.get_variable(name="W1",
                                 shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.relu(tf.matmul(self._X, self.W1))

            # Layer 2
            self.W2 = tf.get_variable(name="W2",
                                 shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer2 = tf.matmul(layer1, self.W2)

            # github의 Network 변형
            # tf.layers.dense(input size, output size, activation=xxx)
            new_layer1 = tf.layers.dense(self._X, h_size, activation=tf.nn.relu)
            new_layer2 = tf.layers.dense(new_layer1, self.output_size)

            # github의 Network 원본
            net = self._X
            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net_layer2 = tf.layers.dense(net, self.output_size)
            # Q prediction, Y_hat, hypothesis
            self._Qpred = net_layer2

            # Q-value(Target)
            self._Y = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.output_size],
                                     name='output_Y')

            # cost = loss function
            self._cost = tf.reduce_mean(tf.square(self._Y - self._Qpred))

            # learning
            self._train = tf.train.AdamOptimizer(learning_rate=l_rate)\
                .minimize(self._cost)


    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)

        Args:
            state (np.ndarray): State array, shape (n, input_dim)

        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """

        self.x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: self.x})


    def update(self, x_stack, y_stack) -> list:
        """Performs updates on given X and y and returns a result

        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)

        Returns:
            list: First element is loss, second element is a result from train step
        """
        return self.session.run([self._cost, self._train],
                                feed_dict={
                                    self._X: x_stack, self._Y: y_stack})
