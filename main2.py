'''
This file provides
train, test, visualization operations on market and duke dataset
'''
import argparse
import os
import torch
import ast
from core import ReIDLoaders, Base, train_an_epoch, test, visualize ,test1
from tools import make_dirs, Logger, os_walk, time_now

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(config):

	# init loaders and base
	loaders = ReIDLoaders(config)
	base = Base(config)

	# make directions
	make_dirs(base.output_path)
	make_dirs(base.save_model_path)
	make_dirs(base.save_logs_path)
	base.vnet.load_state_dict(torch.load('/home/w/LR/mode/sdcnet_150.pkl'))
	# init logger
	logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
	# logger = Logger(os.path.join(config.output_path, 'log.txt'))
	logger(config)


	assert config.mode in ['train', 'test', 'visualize']
	if config.mode == 'train':  # train mode


		# automatically resume model from the latest one
		if config.auto_resume_training_from_lastest_steps:
			start_train_epoch = base.resume_last_model()

		# main loop
		for current_epoch in range(start_train_epoch, config.total_train_epochs):
			# save model

			base.save_model(current_epoch)
			base.sdcnet_optimizer.step()
			# train
			_, results = train_an_epoch(config, base, loaders, current_epoch)
			logger('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))

			if current_epoch + 1 >= 30 and (current_epoch + 1) % 3 == 0:
				pr_mAP, pr_CMC, pi_mAP, pi_CMC, or_mAP, or_CMC = test1(config, base, loaders)
				logger('Time: {}; Test On Partial_reid Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset, pr_mAP, pr_CMC))
				logger('Time: {}; Test On Partial_ilids Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(),
																								 config.test1_dataset,
																								 pi_mAP, pi_CMC))
				logger('Time: {}; Test On Occulided_reid Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(),
																								 config.test2_dataset,
																								 or_mAP, or_CMC))
			if current_epoch + 1 >= 100 and (current_epoch + 1) % 2 == 0:
				pr_mAP, pr_CMC, pi_mAP, pi_CMC, or_mAP, or_CMC = test1(config, base, loaders)
				logger('Time: {}; Test On Partial_reid Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset, pr_mAP, pr_CMC))
				logger('Time: {}; Test On Partial_ilids Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(),
																								 config.test1_dataset,
																								 pi_mAP, pi_CMC))
				logger('Time: {}; Test On Occulided_reid Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(),
																								 config.test2_dataset,
																								 or_mAP, or_CMC))

		# test
		base.save_model(config.total_train_epochs)
		pr_mAP, pr_CMC, pi_mAP, pi_CMC, or_mAP, or_CMC = test(config, base, loaders)
		logger(
			'Time: {}; Test On Partial_reid Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset,
																					  pr_mAP, pr_CMC))
		logger('Time: {}; Test On Partial_ilids Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(),
																						  config.test1_dataset,
																						  pi_mAP, pi_CMC))
		logger('Time: {}; Test On Occulided_reid Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(),
																						   config.test2_dataset,
																						   or_mAP, or_CMC))



	elif config.mode == 'test':	# test mode
		base.resume_from_model(config.resume_test_model)
		base.resume_from_model2(config.resume_test_model2)
		pr_mAP, pr_CMC, pi_mAP, pi_CMC, or_mAP, or_CMC = test(config, base, loaders)
		logger(
			'Time: {}; Test On Partial_reid Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset,
																					  pr_mAP, pr_CMC))
		logger('Time: {}; Test On Partial_ilids Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(),
																						  config.test1_dataset,
																						  pi_mAP, pi_CMC))
		logger('Time: {}; Test On Occulided_reid Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(),
																						   config.test2_dataset,
																						   or_mAP, or_CMC))



	elif config.mode == 'visualize': # visualization mode
		base.resume_from_model(config.resume_visualize_model)
		base.resume_from_model2(config.resume_test_model2)
		visualize(config, base, loaders)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	#
	parser.add_argument('--cuda', type=str, default='cuda')
	parser.add_argument('--mode', type=str, default='train', help='train, test or visualize')
	parser.add_argument('--sdcnet', type=str, default='train', help='train, test or visualize')
	parser.add_argument('--vnet', type=str, default='train', help='train, test or visualize')
	parser.add_argument('--output_path', type=str, default='results/',
						help='path to save related informations')

	# dataset configuration
	parser.add_argument('--market_path', type=str, default='/home/w/LR/Market-1501-v15.09.15/Market-1501-v15.09.15/')
	parser.add_argument('--duke_path', type=str, default='/home/w/LR/grea-q market/')
	parser.add_argument('--msmt_path', type=str, default='/data/datasets/MSMT17_V1/')
	parser.add_argument('--pduke_path', type=str, default='/home/w/LR/P-DukeMTMC-reid-1/')
	parser.add_argument('--occluded_reid_path', type=str, default='/home/w/LR/o1/')
	parser.add_argument('--partial_reid_path', type=str, default='/home/w/LR/Partial-REID_Dataset/Partial-REID_Dataset/')
	parser.add_argument('--partial_ilids_path', type=str, default='/home/w/LR/Partial_iLIDS/')
	parser.add_argument('--combine_all', type=ast.literal_eval, default=False, help='train+query+gallery as train')
	parser.add_argument('--test_dataset', type=str, default='partial_reid', help='market, duke, pduke ,occluded_reid, wildtrack')
	parser.add_argument('--test1_dataset', type=str, default='partial_ilids',
						help='market, duke, pduke ,occluded_reid, wildtrack')
	parser.add_argument('--test2_dataset', type=str, default='occluded_reid',
						help='market, duke, pduke ,occluded_reid, wildtrack')
	parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
	parser.add_argument('--p', type=int, default=8, help='person count in a batch')
	parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')

	# data augmentation
	parser.add_argument('--use_rea', type=ast.literal_eval, default=False)
	parser.add_argument('--use_colorjitor', type=ast.literal_eval, default=True)

	# model configuration
	parser.add_argument('--cnnbackbone', type=str, default='res50', help='res50, res50ibna')
	parser.add_argument('--pid_num', type=int, default=702, help='market:751(combineall-1503), duke:702(1812), pduke:665, msmt:1041(3060), njust:spr3869(5086),win,both(7729)')
	parser.add_argument('--occluded_num', type=int, default=2,help='occluded or not occluded')
	parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')

	# train configuration
	parser.add_argument('--steps', type=int, default=404, help='1 epoch include many steps')
	parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
	parser.add_argument('--base_learning_rate', type=float, default=0.0002)
	# parser.add_argument('__oc_learning_rate', type=float, default=0.0003)
	parser.add_argument('--weight_decay', type=float, default=0.0005)
	parser.add_argument('--total_train_epochs', type=int, default=150)
	parser.add_argument('--auto_resume_training_from_lastest_steps', type=ast.literal_eval, default=True)
	parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')

	# test configuration
	parser.add_argument('--resume_test_model', type=str, default='/home/w/LR/code36+discriminator/2/sdcnet_78.pkl', help='')
	parser.add_argument('--test_mode', type=str, default='inter-camera', help='inter-camera, intra-camera, all')

	# visualization configuration
	parser.add_argument('--resume_visualize_model', type=str, default='/path/to/pretrained/model.pkl',
						help='only availiable under visualize model')
	parser.add_argument('--visualize_dataset', type=str, default='',
						help='market, duke, only  only availiable under visualize model')
	parser.add_argument('--visualize_mode', type=str, default='inter-camera',
						help='inter-camera, intra-camera, all, only availiable under visualize model')
	parser.add_argument('--visualize_mode_onlyshow', type=str, default='pos', help='pos, neg, none')
	parser.add_argument('--visualize_output_path', type=str, default='results/visualization/',
						help='path to save visualization results, only availiable under visualize model')


	# main
	config = parser.parse_args()
	main(config)



