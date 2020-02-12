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

    def _build_network(self, h_size=10, l_rate=0.01) -> None:
        """DQN Network architecture (simple MLP)

        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """
        # Input
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.input_size],
                                name='X')
        self.Y = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.output_size],
                                name='Y')

        # Layer 1
        w1 = tf.get_variable(name="W1",
                             shape=[self.input_size, h_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        l1 = tf.nn.tanh(tf.matmul(self.X, w1))

        # Layer 2
        w2 = tf.get_variable(name="W2",
                             shape=[h_size, self.output_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        l2 = tf.matmul(l1, w2)

        # Output
        self.Y_ = l2

        # cost
        self.cost = tf.reduce_mean(tf.square(self.Y - self.Y_))
        self.train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.cost)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)

        Args:
            state (np.ndarray): State array, shape (n, input_dim)

        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        self.state = np.reshape(state, [-1, self.input_size])
        return self.session.run(self.Y_, feed_dict={self.X: self.state})

    def update(self, x_stack, y_stack) -> list:
        """Performs updates on given X and y and returns a result

        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)

        Returns:
            list: First element is loss, second element is a result from train step
        """
        return self.session.run([self.cost, self.train],
                                feed_dict={
                                    self.X: x_stack, self.Y: y_stack})
