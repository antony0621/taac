from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import gensim
import pickle
import os


class Data:
	def __init__(self, args):
		# initialize variables
		self.input_path = args.input_path
		self.w2v_path = args.w2v_path

		self.window_size = args.window_size
		self.num_folds = args.num_folds
		self.validation_fold = args.validation_fold

		self.features = ['week', 'creative', 'clicktime', 'product', 
                   'category', 'advertiser', 'industry', 'userprofile']
		self.pretrain_fea = ['creative', 'advertiser', 'product', 
                       'category', 'userprofile']

		# read train/test file
		train_all_data = pd.read_csv(os.path.join(self.input_path, 'train_all_data.csv'))
		test_all_data = pd.read_csv(os.path.join(self.input_path, 'test_all_data.csv'))

		# add week feature
		train_all_data['week'] = train_all_data['time'] % 7 + 1
		test_all_data['week'] = test_all_data['time'] % 7 + 1
        
		# add user profile feature
		train_all_data['userprofile'] = 0
		test_all_data['userprofile'] = 0
		
		user_id = 1
        
        
		# get the pretrained embedding matrix
		var = self.__dict__
		for fea in self.features:
			var[fea+'_size'] = pd.concat([train_all_data[fea], test_all_data[fea]]).nunique() + 1
			print('size of feature {}: {}'.format(fea, var[fea+'_size']))

			if fea in self.pretrain_fea:
				w2v_path = os.path.join(self.w2v_path, '{}_w2v.bin'.format(fea))
				w2vModel = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
				fea_word2idx, var[fea+'_embedMatrix'] = build_word2idx_embedMatrix(w2vModel)

				train_all_data[fea] = train_all_data[fea].astype('str').map(lambda x: fea_word2idx[x])
				test_all_data[fea] = test_all_data[fea].astype('str').map(lambda x: fea_word2idx[x])

		# use k-flod cross validation to generate validation users
		users = train_all_data['user'].unique()
		user_size = len(users)
		np.random.seed(11)
		shuffled_indices = np.random.permutation(user_size)
		shuffled_users = users[shuffled_indices]

		fold_size = user_size // self.num_folds
		valid_start_index, valid_end_index = fold_size*(self.validation_fold-1), fold_size*self.validation_fold
		valid_users = shuffled_users[valid_start_index: valid_end_index]

		# generate inputs of train/validation/test set
		self.train_data = train_all_data[~train_all_data['user'].isin(valid_users)]
		self.valid_data = train_all_data[train_all_data['user'].isin(valid_users)]
		self.test_data = test_all_data

		for mode in ['train', 'valid', 'test']:
			tail_data = var[mode+'_data'].groupby('user').tail(self.window_size)
			length_list = tail_data['user'].value_counts().sort_index().tolist()
			fea_values = []

			for fea in self.features:
				array = generate_array(tail_data[fea], length_list, self.window_size)
				fea_values.append(array)

			var[mode+'_inputs'] = np.stack(fea_values, -1)
			print('shape of {}_inputs: {}'.format(mode, var[mode+'_inputs'].shape))

		# generate labels of train/valid set
		self.train_labels = np.array(self.train_data.groupby('user')['agender'].tail(1))
		print('shape of train_labels: {}'.format(self.train_labels.shape))
		self.valid_labels = np.array(self.valid_data.groupby('user')['agender'].tail(1))
		print('shape of valid_labels: {}'.format(self.valid_labels.shape))


def generate_array(data, length_list, max_length):
	indices, indptr, tmp = [], [0], 0

	for n in length_list:
		indices.extend(list(range(max_length-n, max_length)))
		tmp += n
		indptr.append(tmp)

	csr = csr_matrix((data, indices, indptr), shape=(len(length_list), max_length))
	return csr.toarray()


def build_word2idx_embedMatrix(w2vModel):
	vocab_size, embedding_size = len(w2vModel.wv.vocab), w2vModel.vector_size
	embedMatrix = np.zeros((vocab_size+1, embedding_size))
	word2idx, index = {'0': 0}, 0

	for w in w2vModel.wv.vocab.keys():
		index += 1
		embedMatrix[index] = w2vModel.wv[w]
		word2idx[w] = index

	return word2idx, embedMatrix
