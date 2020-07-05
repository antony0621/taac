import pandas as pd
import logging
import gensim
import argparse
import os


def main():
	# define arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, default='../data')
	parser.add_argument('--output_path', type=str, default='.')

	parser.add_argument('--feature', type=str, required=True)
	parser.add_argument('--size', type=int, default=150)
	parser.add_argument('--window', type=int, default=10)
	parser.add_argument('--sg', type=int, default=1)
	parser.add_argument('--min_count', type=int, default=1)
	parser.add_argument('--iter', type=int, default=10)
	args = parser.parse_args()
	print(str(args))

	# merge train data and test data
	train_all_data = pd.read_csv(os.path.join(args.input_path, 'train_all_data.csv'))
	test_all_data = pd.read_csv(os.path.join(args.input_path, 'test_all_data.csv'))
	all_data = pd.concat([train_all_data.drop(columns=['agender']), test_all_data])

	# generate word list for specified feature
	all_data[args.feature] = all_data[args.feature].astype('str')
	word_list = list(all_data.groupby('user').apply(lambda x: list(x[args.feature])))

	# begin to train word2vector
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = gensim.models.Word2Vec(word_list, size=args.size, window=args.window, sg=args.sg, min_count=args.min_count, iter=args.iter)

	# save train result
	dir_name = 'size{}_win{}_iter{}'.format(args.size, args.window, args.iter)
	output_dir = os.path.join(args.output_path, dir_name)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	file_name = '{}_w2v.bin'.format(args.feature)
	model.wv.save_word2vec_format(os.path.join(output_dir, file_name), binary=True)


if __name__ == '__main__':
	main()
