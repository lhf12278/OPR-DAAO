import torch
from tools import time_now, CatMeter, ReIDEvaluator, PrecisionRecall
import numpy as np
import matplotlib.pyplot as plt
import os

def test(config, base, loaders):

	base.set_eval()

	# meters
	pr_query_features_meter, pr_query_pids_meter, pr_query_cids_meter = CatMeter(), CatMeter(), CatMeter()
	# query_features_meter, query_pids_meter = CatMeter(), CatMeter()
	pr_gallery_features_meter, pr_gallery_pids_meter, pr_gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
	# gallery_features_meter, gallery_pids_meter = CatMeter(), CatMeter()
	pi_query_features_meter, pi_query_pids_meter, pi_query_cids_meter = CatMeter(), CatMeter(), CatMeter()
	pi_gallery_features_meter, pi_gallery_pids_meter, pi_gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

	or_query_features_meter, or_query_pids_meter, or_query_cids_meter = CatMeter(), CatMeter(), CatMeter()
	or_gallery_features_meter, or_gallery_pids_meter, or_gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
	
	pr_loaders = [loaders.partial_reid_query_loader, loaders.partial_reid_gallery_loader]
	pi_loaders = [loaders.partial_ilids_query_loader, loaders.partial_ilids_gallery_loader]
	or_loaders = [loaders.occluded_reid_query_loader, loaders.occluded_reid_gallery_loader]

	# init dataset
	# if config.test_dataset == 'market':
	# 	loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
	# elif config.test_dataset == 'pduke':
	# 	loaders = [loaders.pduke_query_loader, loaders.pduke_gallery_loader]
	# elif config.test_dataset == 'partial_ilids':
	# 	loaders = [loaders.partial_ilids_query_loader, loaders.partial_ilids_gallery_loader]
	# elif config.test_dataset == 'partial_reid':
	# 	loaders = [loaders.partial_reid_query_loader, loaders.partial_reid_gallery_loader]
	# elif config.test_dataset == 'occluded_reid':
	# 	loaders = [loaders.occluded_reid_query_loader, loaders.occluded_reid_gallery_loader]
	# elif config.test_dataset == 'duke':
	# 	loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]




	print(time_now(), 'partial_reid_feature start')

	# compute query and gallery features
	with torch.no_grad():
		for pr_loader_id, pr_loader in enumerate(pr_loaders):
			for pr_data in pr_loader:
				# compute feautres

				pr_images, pr_pids, pr_cids = pr_data

				# images, pids = data
				pr_images = pr_images.to(base.device)
				pr_sumfeatures = base.sdcnet(pr_images)

				# save as query features
				if pr_loader_id == 0:
					pr_query_features_meter.update(pr_sumfeatures.data)
					pr_query_pids_meter.update(pr_pids)
					pr_query_cids_meter.update(pr_cids)

				# save as gallery features
				elif pr_loader_id == 1:
					pr_gallery_features_meter.update(pr_sumfeatures.data)
					pr_gallery_pids_meter.update(pr_pids)
					pr_gallery_cids_meter.update(pr_cids)

	print(time_now(), 'partial_reid_feature done')

	print(time_now(), 'partial_ilids_feature start')

	# compute query and gallery features
	with torch.no_grad():
		for pi_loader_id, pi_loader in enumerate(pi_loaders):
			for pi_data in pi_loader:
				# compute feautres

				pi_images, pi_pids, pi_cids =pi_data

				# images, pids = data
				pi_images = pi_images.to(base.device)
				pi_sumfeatures = base.sdcnet(pi_images)

				# save as query features
				if pi_loader_id == 0:
					pi_query_features_meter.update(pi_sumfeatures.data)
					pi_query_pids_meter.update(pi_pids)
					pi_query_cids_meter.update(pi_cids)

				# save as gallery features
				elif pi_loader_id == 1:
					pi_gallery_features_meter.update(pi_sumfeatures.data)
					pi_gallery_pids_meter.update(pi_pids)
					pi_gallery_cids_meter.update(pi_cids)

	print(time_now(), 'partial_ilids_feature done')
	
	print(time_now(), 'occulded_reid_feature start')

	# compute query and gallery features
	with torch.no_grad():
		for or_loader_id, or_loader in enumerate(or_loaders):
			for or_data in or_loader:
				# compute feautres

				or_images, or_pids, or_cids =or_data


				or_images = or_images.to(base.device)
				or_sumfeatures = base.sdcnet(or_images)

				# save as query features
				if or_loader_id == 0:
					or_query_features_meter.update(or_sumfeatures.data)
					or_query_pids_meter.update(or_pids)
					or_query_cids_meter.update(or_cids)

				# save as gallery features
				elif or_loader_id == 1:
					or_gallery_features_meter.update(or_sumfeatures.data)
					or_gallery_pids_meter.update(or_pids)
					or_gallery_cids_meter.update(or_cids)

	print(time_now(), 'occulded_reid_feature done')



	#
	pr_query_features = pr_query_features_meter.get_val_numpy()
	pr_gallery_features = pr_gallery_features_meter.get_val_numpy()
	
	pi_query_features = pi_query_features_meter.get_val_numpy()
	pi_gallery_features = pi_gallery_features_meter.get_val_numpy()
	
	or_query_features = or_query_features_meter.get_val_numpy()
	or_gallery_features = or_gallery_features_meter.get_val_numpy()

	# compute mAP and rank@k
	pr_mAP, pr_CMC = ReIDEvaluator(dist='cosine', mode=config.test_mode).evaluate(
		pr_query_features, pr_query_cids_meter.get_val_numpy(), pr_query_pids_meter.get_val_numpy(),
		pr_gallery_features, pr_gallery_cids_meter.get_val_numpy(), pr_gallery_pids_meter.get_val_numpy())
	pi_mAP, pi_CMC = ReIDEvaluator(dist='cosine', mode=config.test_mode).evaluate(
		pi_query_features, pi_query_cids_meter.get_val_numpy(), pi_query_pids_meter.get_val_numpy(),
		pi_gallery_features, pi_gallery_cids_meter.get_val_numpy(), pi_gallery_pids_meter.get_val_numpy())
	or_mAP, or_CMC = ReIDEvaluator(dist='cosine', mode=config.test_mode).evaluate(
		or_query_features, or_query_cids_meter.get_val_numpy(), or_query_pids_meter.get_val_numpy(),
		or_gallery_features, or_gallery_cids_meter.get_val_numpy(), or_gallery_pids_meter.get_val_numpy())

	return pr_mAP, pr_CMC[0: 30], pi_mAP, pi_CMC[0: 30], or_mAP, or_CMC[0: 30]








# 	# comp ute precision-recall curve
# 	thresholds = np.linspace(1.0, 0.0, num=101)
# 	pres, recalls, thresholds = PrecisionRecall(dist='cosine', mode=config.test_mode).evaluate(
# 		thresholds, query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
# 		gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy())
# 
# 	return mAP, CMC[0: 150]
# 
# 
# def plot_prerecall_curve(config, pres, recalls, thresholds, mAP, CMC, label):
# 
# 	plt.plot(recalls, pres, label='{model},map:{map},cmc135:{cmc}'.format(
# 		model=label, map=round(mAP, 2), cmc=[round(CMC[0], 2), round(CMC[2], 2), round(CMC[4], 2)]))
# 	plt.xlabel('recall')
# 	plt.ylabel('precision')
# 	plt.title('precision-recall curve')
# 	plt.legend()
# 	plt.grid()
# 	plt.savefig(os.path.join(config.output_path, 'precisio-recall-curve.png'))

