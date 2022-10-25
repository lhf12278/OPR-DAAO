import torch
from tools import MultiItemAverageMeter, accuracy
from numpy import *


def train_an_epoch(config, base, loaders, epoch=None):


	base.set_train()
	meter = MultiItemAverageMeter()

	base.sdcnet_optimizer.step()
	base.sdcnet_lr_scheduler.step(epoch)
	for _ in range(config.steps):

		### load a batch data
		quanshen_imgs, quanshen_pids, oimgs, ntrans_quanshen_imgs, ntrans_oimgs = loaders.train_quanshen_iter.next_one()
		quanshen_imgs, quanshen_pids, oimgs, ntrans_quanshen_imgs, ntrans_oimgs = quanshen_imgs.to(base.device), quanshen_pids.to(base.device), \
																				  oimgs.to(base.device), ntrans_quanshen_imgs.to(base.device), ntrans_oimgs.to(base.device)

		if 'res' in config.cnnbackbone:

			mixgapfeatures, mixgmpfeatures, mixsumfeatures, mixgap_cls_score, mixgmp_cls_score, mixsum_cls_score, out2, out3, out4, out3sum, out2sum = base.sdcnet(ntrans_oimgs,base.nonnlocal)
			q_mixgapfeatures, q_mixgmpfeatures, q_mixsumfeatures, q_mixgap_cls_score, q_mixgmp_cls_score, q_mixsum_cls_score, q_out2, q_out3, q_out4, q_out3sum, q_out2sum = base.sdcnet(ntrans_quanshen_imgs,base.nonnlocal)

			with torch.no_grad():
 					gapfeatures_t, gap_cls_score_t, gmpfeatures_t, gmp_cls_score_t, sumfeatures_t, sum_cls_score_t, out2_t, out2gap, out3_t, out4_t = base.vnet(ntrans_quanshen_imgs)

			z, W_y = base.nonnlocal(out4)

			mixgapfeather = base.sdcnet.gap(W_y+out4_t).squeeze(dim=2).squeeze(dim=2)
			mixgap_bned_features, mixgap_cls_score = base.vnet.classifier(mixgapfeather)

			q_z,q_W_y =base.nonnlocal(q_out4)
			q_mixgapfeather = base.sdcnet.gap(q_W_y +out4_t).squeeze(dim=2).squeeze(dim=2)
			q_mixgapfeather  , q_mixquanshen_cls_score = base.vnet.classifier(q_mixgapfeather)

			mixgap_ide_loss = base.ide_creiteron(mixgap_cls_score,quanshen_pids)
			q_mixgap_ide_loss = base.ide_creiteron(q_mixquanshen_cls_score,quanshen_pids)

			q_mixquanshen_gap_ide_loss = base.ide_creiteron(q_mixgap_cls_score, quanshen_pids)
			q_mixquanshen_gmp_ide_loss = base.ide_creiteron(q_mixgmp_cls_score, quanshen_pids)
			q_mixquanshen_sum_ide_loss = base.ide_creiteron(q_mixsum_cls_score, quanshen_pids)

			mixquanshen_gap_ide_loss = base.ide_creiteron(mixgap_cls_score, quanshen_pids)
			mixquanshen_gmp_ide_loss = base.ide_creiteron(mixgmp_cls_score, quanshen_pids)
			mixquanshen_sum_ide_loss = base.ide_creiteron(mixsum_cls_score, quanshen_pids)

			q_mixquanshen_gap_triplet_loss = base.triplet_creiteron(q_mixgapfeatures, q_mixgapfeatures,
																	q_mixgapfeatures, quanshen_pids, quanshen_pids,
																	quanshen_pids)
			q_mixquanshen_gmp_triplet_loss = base.triplet_creiteron(q_mixgmpfeatures, q_mixgmpfeatures,
																	q_mixgmpfeatures, quanshen_pids, quanshen_pids,
																	quanshen_pids)
			q_mixquanshen_sum_triplet_loss = base.triplet_creiteron(q_mixsumfeatures, q_mixsumfeatures,
																	q_mixsumfeatures, quanshen_pids, quanshen_pids,
																	quanshen_pids)

			l2loss_map4 = base.l2_loss(out4, out4_t)
			l2loss_map3 = base.l2_loss(out3, out3_t)

			loss = 0.1 * (mixquanshen_gap_ide_loss + mixquanshen_gmp_ide_loss + mixquanshen_sum_ide_loss) \
				   + 1*(q_mixquanshen_gap_ide_loss + q_mixquanshen_gmp_ide_loss + q_mixquanshen_sum_ide_loss) + \
				   q_mixquanshen_gap_triplet_loss + q_mixquanshen_gmp_triplet_loss + q_mixquanshen_sum_triplet_loss \
				   + (l2loss_map4 + l2loss_map3) * 0.01+(mixgap_ide_loss+q_mixgap_ide_loss)*0.001




			base.sdcnet_optimizer.zero_grad()
			base.nonnlocal_optimizer.zero_grad()
			loss.backward()
			base.sdcnet_optimizer.step()
			base.nonnlocal_optimizer.step()

			acc1 = accuracy(mixsum_cls_score, quanshen_pids, [1])[0]

			### recored
			meter.update({'mixquanshen_sum_ide_loss': mixquanshen_sum_ide_loss,
						  'q_mixquanshen_sum_triplet_loss': q_mixquanshen_sum_triplet_loss.data, \
						  'mixgap_ide_loss': mixgap_ide_loss.data, 'q_mixgap_ide_loss': q_mixgap_ide_loss, 'l2loss_map4': l2loss_map4.data, \
						  'q_mixquanshen_sum_ide_loss': q_mixquanshen_sum_ide_loss.data, 'acc1': acc1})
	return meter.get_val(), meter.get_str()

