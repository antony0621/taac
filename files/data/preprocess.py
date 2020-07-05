import os
import argparse
import pandas as pd


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, default='.')
	parser.add_argument('--output_path', type=str, default='.')
	args = parser.parse_args()

	# 读取训练集的用户信息文件，融合年龄与性别
	train_path = os.path.join(args.input_path, 'train_preliminary')
	train_user = pd.read_csv(os.path.join(train_path, 'user.csv'))
	train_user['agender'] = train_user['age'] + (train_user['gender']-1)*10 - 1

	# 读取训练集的点击日志和广告信息，判断点击日志与广告信息、用户信息的拼接主键是否一致
	train_click_log = pd.read_csv(os.path.join(train_path, 'click_log.csv'))
	train_ad = pd.read_csv(os.path.join(train_path, 'ad.csv'))
	print(set(train_click_log['creative_id']) == set(train_ad['creative_id']))
	print(set(train_click_log['user_id']) == set(train_user['user_id']))

	# 将点击日志分别和广告信息、用户信息拼接
	train_all_data = pd.merge(train_click_log, train_ad, on=['creative_id'], how='left')
	train_all_data = pd.merge(train_all_data, train_user, on=['user_id'], how='left')

	# 调整拼接好的训练集数据格式
	del train_all_data['age']
	del train_all_data['gender']

	train_all_data = train_all_data.replace('\\N', 0)
	train_all_data = train_all_data.astype('int64')
	train_all_data = train_all_data.sort_values(['user_id', 'time'])

	train_all_data.rename(columns={'user_id':'user', 'creative_id':'creative', 'click_times':'clicktime', 
				       'ad_id':'ad', 'product_id':'product', 'product_category':'category', 
				       'advertiser_id':'advertiser'}, inplace=True)

	# 读取测试集的点击日志和广告信息
	test_path = os.path.join(args.input_path, 'test')
	test_click_log = pd.read_csv(os.path.join(test_path, 'click_log.csv'))
	test_ad = pd.read_csv(os.path.join(test_path, 'ad.csv'))

	# 验证点击日志与广告信息的拼接主键是否一致
	print(set(test_click_log['creative_id']) == set(test_ad['creative_id']))

	# 将点击日志和广告信息拼接
	test_all_data = pd.merge(test_click_log, test_ad, on=['creative_id'], how='left')

	# 调整拼接好的测试集数据格式
	test_all_data = test_all_data.replace('\\N', 0)
	test_all_data = test_all_data.astype('int64')
	test_all_data = test_all_data.sort_values(['user_id', 'time'])

	test_all_data.rename(columns={'user_id':'user', 'creative_id':'creative', 'click_times':'clicktime', 
				      'ad_id':'ad', 'product_id':'product', 'product_category':'category', 
				      'advertiser_id':'advertiser'}, inplace=True)

	# 对训练集和测试集的特征列重新进行映射，使其成为从1开始的连续整数
	features = ['creative', 'clicktime', 'ad', 'product', 'category', 'advertiser', 'industry']

	for fea in features:
		# 先检测该特征是否为从1开始的连续整数
		fea_set = set(train_all_data[fea]) | set(test_all_data[fea])
		fea_size, fea_min, fea_max = len(fea_set), min(fea_set), max(fea_set)
		print('before mapping, for feature %s, size: %d, min: %d, max: %d' % (fea, fea_size, fea_min, fea_max))

		# 如果该特征不是从1开始的连续整数，则进行映射
		if fea_max != fea_size:
			# 生成映射后的特征值
			fea_df = pd.DataFrame(data=fea_set, columns=[fea])
			fea_df = fea_df.sort_values(by=[fea]).reset_index(drop=True)
			fea_df['mapped_'+fea] = fea_df.index + 1

			# 对该特征进行映射
			train_all_data = pd.merge(train_all_data, fea_df, on=[fea], how='left')
			test_all_data = pd.merge(test_all_data, fea_df, on=[fea], how='left')

			# 检测映射后的该特征是否为从1开始的连续整数
			fea_set = set(train_all_data['mapped_'+fea]) | set(test_all_data['mapped_'+fea])
			fea_size, fea_min, fea_max = len(fea_set), min(fea_set), max(fea_set)
			print('after mapping, for feature %s, size: %d, min: %d, max: %d' % (fea, fea_size, fea_min, fea_max))

	# 对映射完的训练集数据进行格式调整
	train_all_data = train_all_data[['time','user','mapped_creative','mapped_clicktime','mapped_ad',
					 'mapped_product','category','mapped_advertiser','mapped_industry','agender']]

	train_all_data.rename(columns={'mapped_creative':'creative', 'mapped_clicktime':'clicktime', 
				       'mapped_ad':'ad', 'mapped_product':'product', 'mapped_advertiser':'advertiser', 
				       'mapped_industry':'industry'}, inplace=True)

	# 对映射完的测试集数据进行格式调整
	test_all_data = test_all_data[['time','user','mapped_creative','mapped_clicktime','mapped_ad',
				       'mapped_product','category','mapped_advertiser','mapped_industry']]

	test_all_data.rename(columns={'mapped_creative':'creative', 'mapped_clicktime':'clicktime', 
				      'mapped_ad':'ad', 'mapped_product':'product', 'mapped_advertiser':'advertiser', 
				      'mapped_industry':'industry'}, inplace=True)

	# 保存训练集和测试集文件
	train_all_data.to_csv(os.path.join(args.output_path, 'train_all_data.csv'), index=False)
	test_all_data.to_csv(os.path.join(args.output_path, 'test_all_data.csv'), index=False)


if __name__ == '__main__':
	main()
