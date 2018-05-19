import tensorflow as tf

#tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory


class AdamOptimizer(tf.train.Optimizer):
	def __init__(self, alpha=0.001, 
						beta1=0.9, 
						beta2=0.999,
						epsilon=1e-8, loss=None, t_vars= None):
		
		self.alpha = alpha
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		
		self.m = {}
		self.u = {}
		self.t = tf.Variable(0.0, trainable=False)
		self.loss = loss
		self.t_vars = t_vars

		with tf.name_scope('optimizer/'):
			for v in t_vars:
				self.m[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)
				self.u[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)

	def get_variables(self):
		return list(self.m.values()) + list(self.u.values()) + [self.t]

	def update(self, ys, y_grads):
		#grads = gradients(self.loss, self.t_vars, checkpoints='speed')
		grads = tf.gradients(ys, self.t_vars, grad_ys=y_grads)

		t = self.t.assign_add(1.0)

		update_ops = []
		for (g,v) in zip(grads, self.t_vars):
			m = self.m[v].assign(self.beta1*self.m[v] + (1-self.beta1)*g)
			u = self.u[v].assign(self.beta2*self.u[v] + (1-self.beta2)*g*g)
			m_hat = m/(1-tf.pow(self.beta1,t))
			u_hat = u/(1-tf.pow(self.beta2,t))
			
			update = -self.alpha*m_hat/(tf.sqrt(u_hat) + self.epsilon)
			update_ops.append(v.assign_add(update))
			
		return tf.group(*update_ops)