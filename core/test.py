import torch
from tools import time_now, CatMeter, ReIDEvaluator, PrecisionRecall
import numpy as np
import matplotlib.pyplot as plt
import os

def test(config, base, loaders):

	base.set_eval()

	# meters
	query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
	# query_features_meter, query_pids_meter = CatMeter(), CatMeter()
	gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
	# gallery_features_meter, gallery_pids_meter = CatMeter(), CatMeter()

	# init dataset
	if config.test_dataset == 'market':
		loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
	elif config.test_dataset == 'pduke':
		loaders = [loaders.pduke_query_loader, loaders.pduke_gallery_loader]
	elif config.test_dataset == 'partial_ilids':
		loaders = [loaders.partial_ilids_query_loader, loaders.partial_ilids_gallery_loader]
	elif config.test_dataset == 'partial_reid':
		loaders = [loaders.partial_reid_query_loader, loaders.partial_reid_gallery_loader]
	elif config.test_dataset == 'occluded_reid':
		loaders = [loaders.occluded_reid_query_loader, loaders.occluded_reid_gallery_loader]
	elif config.test_dataset == 'duke':
		loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]
	print(time_now(), 'feature start')

	# compute query and gallery features
	with torch.no_grad():
		for loader_id, loader in enumerate(loaders):
			for data in loader:
				# compute feautres

				images, pids, cids = data
				images = images.to(base.device)
				sumfeatures = base.sdcnet(images,base.nonnlocal)

				if loader_id == 0:
					query_features_meter.update(sumfeatures.data)
					query_pids_meter.update(pids)
					query_cids_meter.update(cids)

				# save as gallery features
				elif loader_id == 1:
					gallery_features_meter.update(sumfeatures.data)
					gallery_pids_meter.update(pids)
					gallery_cids_meter.update(cids)

	print(time_now(), 'feature done')

	#
	query_features = query_features_meter.get_val_numpy()
	gallery_features = gallery_features_meter.get_val_numpy()

	# compute mAP and rank@k
	mAP, CMC = ReIDEvaluator(dist='cosine', mode=config.test_mode).evaluate(
		query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
		gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy())

	# comp ute precision-recall curve
	thresholds = np.linspace(1.0, 0.0, num=101)
	pres, recalls, thresholds = PrecisionRecall(dist='cosine', mode=config.test_mode).evaluate(
		thresholds, query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
		gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy())

	return mAP, CMC[0: 40]


def plot_prerecall_curve(config, pres, recalls, thresholds, mAP, CMC, label):

	plt.plot(recalls, pres, label='{model},map:{map},cmc135:{cmc}'.format(
		model=label, map=round(mAP, 2), cmc=[round(CMC[0], 2), round(CMC[2], 2), round(CMC[4], 2)]))
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.title('precision-recall curve')
	plt.legend()
	plt.grid()
	plt.savefig(os.path.join(config.output_path, 'precisio-recall-curve.png'))

