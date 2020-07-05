from data import Data
from model import Model

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import time
import os


def main():
	# define agruments
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, default='files/data')
	parser.add_argument('--w2v_path', type=str, default='files/w2v/size150_win12_iter10')
	parser.add_argument('--output_path', type=str, default='files/results')

	parser.add_argument('--window_size', type=int, default=128)
	parser.add_argument('--latent_dimension', type=int, default=64)
	parser.add_argument('--hidden_units', type=int, default=600)

	parser.add_argument('--num_epochs', type=int, default=400)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--learn_rate', type=float, default=1e-3)
	parser.add_argument('--keep_prob', type=float, default=0.6)
	parser.add_argument('--age_weight', type=float, default=0.4)

	parser.add_argument('--logid', type=int, default=0)
	parser.add_argument('--num_folds', type=int, default=10)
	parser.add_argument('--validation_fold', type=int, default=10)
	args = parser.parse_args()
	print(str(args))

	# define data
	print('data process begin...')
	t1 = time.time()
	data = Data(args)
	used_time = (time.time() - t1) / 60
	print('data process end, used time: {:.2f} min'.format(used_time))

	# define and train model
	model = Model(args)
	train(args, data, model)


def train(args, data, model):
	# define loss and predictions
	batch_inputs = tf.placeholder(tf.int32, [args.batch_size, args.window_size, len(data.features)])
	batch_labels = tf.placeholder(tf.int32, [args.batch_size, ])
	age_loss, gender_loss, total_loss, train_op = model.forward(data, batch_inputs, batch_labels, True)
	age_predictions, gender_predictions = model.forward(data, batch_inputs, None, False)

	# calculate batch_size and iterations for train/validation/test set
	train_iterations = data.train_inputs.shape[0] // args.batch_size + 1
	valid_batch_size = args.batch_size * 10
	valid_iterations = data.valid_inputs.shape[0] // valid_batch_size + 1
	test_batch_size = args.batch_size * 10
	test_iterations = data.test_inputs.shape[0] // test_batch_size + 1

	# define evaluation metric for early stopping, b means best
	btotal_acc, bage_acc, bgender_acc, bepoch, btest_age_preds, btest_gender_preds, times = 0, 0, 0, 0, 0, 0, []

	# begin to train model
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(1, args.num_epochs+1):
			# shuffle train data at each epoch
			shuffled_indices = np.random.permutation(data.train_inputs.shape[0])
			data.train_inputs = data.train_inputs[shuffled_indices]
			data.train_labels = data.train_labels[shuffled_indices]

			# compute and display train loss
			start_time = time.time()
			for j in range(1, train_iterations+1):
				start, end = args.batch_size * (j-1), args.batch_size * j
				train_feed_dict = {batch_inputs: data.train_inputs[start:end], 
							batch_labels: data.train_labels[start:end]}
				age_loss_, gender_loss_, total_loss_, train_op_ = sess.run([age_loss, gender_loss, total_loss, train_op], 
											   feed_dict = train_feed_dict)
				if j % (train_iterations//5) == 0: 
					print('epoch: {}, iteration: {}, age_loss: {:.6f}, gender_loss: {:.6f}, total_loss: {:.6f}'.format
					     (i, j, age_loss_, gender_loss_, total_loss_))
			times.append(time.time() - start_time)

			if i % 1 == 0:
				# generate validation predictions
				valid_age_list, valid_gender_list = [], []
				for j in range(1, valid_iterations+1):
					start, end = valid_batch_size * (j-1), valid_batch_size * j
					valid_feed_dict = {batch_inputs: data.valid_inputs[start:end]}
					valid_age_predictions, valid_gender_predictions = sess.run([age_predictions, gender_predictions], 
												   feed_dict=valid_feed_dict)
					valid_age_list.append(valid_age_predictions)
					valid_gender_list.append(valid_gender_predictions)

				# compute validation performance
				valid_pred_age = np.argmax(np.concatenate(valid_age_list, 0), 1) + 1
				valid_pred_gender = np.argmax(np.concatenate(valid_gender_list, 0), 1) + 1
				valid_true_age, valid_true_gender = data.valid_labels%10+1, data.valid_labels//10+1

				age_acc = np.mean(np.equal(valid_pred_age, valid_true_age))
				gender_acc = np.mean(np.equal(valid_pred_gender, valid_true_gender))
				total_acc = age_acc + gender_acc	
				print('epoch: {}, (valid) age acc: {:.8f}, gender acc: {:.8f}, total acc: {:.8f}\n'.format(i, age_acc, gender_acc, total_acc))

				# generate test predictions
				test_age_list, test_gender_list = [], []
				for j in range(1, test_iterations+1):
					start, end = test_batch_size * (j-1), test_batch_size * j
					test_feed_dict = {batch_inputs: data.test_inputs[start:end]}
					test_age_predictions, test_gender_predictions = sess.run([age_predictions, gender_predictions], 
												 feed_dict=test_feed_dict)
					test_age_list.append(test_age_predictions)
					test_gender_list.append(test_gender_predictions)
				test_age_preds = np.concatenate(test_age_list, 0)
				test_gender_preds = np.concatenate(test_gender_list, 0)

				# record the best validation performance 
				if total_acc >= btotal_acc:
					btotal_acc, bage_acc, bgender_acc, bepoch, btest_age_preds, btest_gender_preds = \
					total_acc, age_acc, gender_acc, i, test_age_preds, test_gender_preds

				# early stopping
				if i - bepoch >= 2:
					# display consumed time and model arguments
					total_time = sum(times[:bepoch]) / 60
					avg_time = np.mean(times[:bepoch]) / 60
					print(args.logid, ':', args.window_size, args.latent_dimension, args.hidden_units, '\t', 
							       args.num_epochs, args.batch_size, args.learn_rate, args.keep_prob, args.age_weight, '\t', 
							       args.num_folds, args.validation_fold)

					# display the best validation performance
					print('best epoch: {}, time: {:.2f} min, {:.2f} min, (valid) age acc: {:.8f}, gender acc: {:.8f}, total acc: {:.8f}'.format
						(bepoch, total_time, avg_time, bage_acc, bgender_acc, btotal_acc))

					# compute and save the corresponding test result, prediction_*.npy is used for merging
					output_dir = os.path.join(args.output_path, str(args.logid))
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)

					prediction_age_path = os.path.join(output_dir, 'prediction_age_{}_{}_{}.npy'.format(args.logid, args.num_folds, args.validation_fold))
					np.save(prediction_age_path, btest_age_preds)
					prediction_gender_path = os.path.join(output_dir, 'prediction_gender_{}_{}_{}.npy'.format(args.logid, args.num_folds, args.validation_fold))
					np.save(prediction_gender_path, btest_gender_preds)
					
					test_pred_age = np.argmax(btest_age_preds, 1) + 1
					test_pred_gender = np.argmax(btest_gender_preds, 1) + 1
					submissions = pd.DataFrame({'user_id':np.arange(3000001, 4000001), 'predicted_age':test_pred_age, 'predicted_gender':test_pred_gender})
					submission_path = os.path.join(output_dir, 'submission_{}_{}_{}.csv'.format(args.logid, args.num_folds, args.validation_fold))
					submissions.to_csv(submission_path, index=False)

					break


if __name__ == '__main__':
	main()
