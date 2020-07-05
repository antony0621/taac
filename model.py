import tensorflow as tf


class Model():
	def __init__(self, args):
		# initialize parameters
		self.window_size = args.window_size
		self.latent_dimension = args.latent_dimension
		self.hidden_units = args.hidden_units

		self.learn_rate = args.learn_rate
		self.keep_prob = args.keep_prob
		self.age_weight = args.age_weight

	def forward(self, data, batch_inputs, batch_labels, is_training):
		# embedding layer
		with tf.variable_scope("embedding_layer", reuse = tf.AUTO_REUSE):
			var, emb_list = self.__dict__, []
			for i, fea in enumerate(data.features):
				if fea in data.pretrain_fea:
					with tf.device('/cpu:0'):
						var[fea+'_emb_table'] = tf.get_variable(name=fea+"_emb_table", shape=data.__dict__[fea+'_embedMatrix'].shape,
										initializer=tf.constant_initializer(data.__dict__[fea+'_embedMatrix']), trainable=False)
						var[fea+'_embs'] = tf.nn.embedding_lookup(var[fea+'_emb_table'], batch_inputs[:,:,i])
				else:
					var[fea+'_emb_table'] = tf.get_variable(name=fea+"_emb_table", shape=[data.__dict__[fea+'_size'], self.latent_dimension],
													initializer=tf.truncated_normal_initializer(stddev=0.01))
					var[fea+'_embs'] = tf.nn.embedding_lookup(var[fea+'_emb_table'], batch_inputs[:,:,i])
				emb_list.append(var[fea+'_embs'])

		# lstm layer
		sequences = tf.concat(emb_list, -1)
		if not self.hidden_units: self.hidden_units = sequences.get_shape()[-1]
# 		sequence_length = tf.reduce_sum(tf.to_int32(tf.not_equal(batch_inputs[:,:,0], 0)), 1)

		cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_units)  # forward
		cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_units)  # backward
		with tf.variable_scope("dynamic_rnn", reuse = tf.AUTO_REUSE):
			outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sequences, dtype=tf.float32)

		outputs_fw, outputs_bw = outputs
		outputs = outputs_fw + outputs_bw
        
		att = attention(outputs)
		outputs = tf.multiply(outputs, att)
        # outputs = conv1d(outputs)
		lstm_out = lstm(outputs, self.hidden_units)
		lstm_out = tf.reduce_max(lstm_out, 1)

		# prediction layer
		with tf.variable_scope("prediction_layer", reuse = tf.AUTO_REUSE):
			aggregation = lstm_out
			if is_training: aggregation = tf.nn.dropout(aggregation, self.keep_prob)
			age_weights = tf.get_variable("age_weights", shape=[aggregation.get_shape()[-1], 10], initializer=tf.truncated_normal_initializer(stddev=0.01))
			gender_weights = tf.get_variable("gender_weights", shape=[aggregation.get_shape()[-1], 2], initializer=tf.truncated_normal_initializer(stddev=0.01))

		if is_training:
			age_logits = tf.matmul(aggregation, age_weights)
			age_labels = batch_labels % 10
			age_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=age_labels, logits=age_logits))

			gender_logits = tf.matmul(aggregation, gender_weights)
			gender_labels = batch_labels // 10
			gender_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gender_labels, logits=gender_logits))

			total_loss = age_loss*self.age_weight + gender_loss*(1-self.age_weight)
			train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(total_loss)
			return age_loss, gender_loss, total_loss, train_op
		else:
			age_predictions = tf.matmul(aggregation, age_weights)
			age_predictions = tf.nn.softmax(age_predictions)

			gender_predictions = tf.matmul(aggregation, gender_weights)
			gender_predictions = tf.nn.softmax(gender_predictions)
			return age_predictions, gender_predictions


def lstm(sequences, hidden_units=None):
	batch_size = tf.shape(sequences)[0]
	time_steps = sequences.get_shape()[1]
	input_dimension = sequences.get_shape()[2]
	if not hidden_units: hidden_units = input_dimension

	cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
	initial_state = cell.zero_state(batch_size, tf.float32)
	state = initial_state
	outputs = []

	with tf.variable_scope("lstm", reuse = tf.AUTO_REUSE):
		for t in range(time_steps):
			output, state = cell(sequences[:,t,:], state)
			outputs.append(output)

	outputs = tf.stack(outputs, 1)
	return outputs
    
    
def conv1d(input_, output_channels, dilation=1, kernel_size=1, causal=False, name="dilated_conv"):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0))

        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias

        return tf.squeeze(out, [1])
        
def weight_variable(shape, validate_shape=False):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, validate_shape=validate_shape)

def bias_variable(shape, validate_shape=False):
   # initial = tf.constant(0.1, shape=shape)
    initial = tf.truncated_normal(shape, mean =0.1, stddev=0.0001)
    return tf.Variable(initial, validate_shape=validate_shape)

def attention(x, name='attention'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        batch = tf.shape(x)[0]
        c = int(x.get_shape()[1])
        dim = int(x.get_shape()[2])

        gap_vec = tf.reduce_mean(x, axis=1, keep_dims=True)
        print(gap_vec.get_shape())
        w1 = weight_variable([batch, dim, dim], validate_shape=False)
        b1 = bias_variable([batch, dim], validate_shape=False)

        ker = tf.nn.relu(tf.matmul(w1, tf.transpose(gap_vec,[0,2,1])) + tf.expand_dims(b1, -1))
        print(ker.get_shape())
        w2 = weight_variable([batch, dim, dim], validate_shape=False)
        b2 = bias_variable([batch, dim], validate_shape=False)

        ker2 = tf.matmul(w2,ker) + tf.expand_dims(b2, -1)
        print(ker2.get_shape())
        att = tf.transpose(tf.transpose(x,[0,2,1])*ker2,[0,2,1])
        att = tf.nn.softmax(att/10, axis=1)
        return tf.nn.sigmoid(att)
