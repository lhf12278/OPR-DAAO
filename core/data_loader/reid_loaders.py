import sys
sys.path.append('../')

from .dataset import *
from .loader import *

import torchvision.transforms as transforms
from tools import *


class ReIDLoaders:

    def __init__(self, config):

        # resize --> flip --> pad+crop --> colorjitor(optional) --> totensor+norm --> rea (optional)
        transform_train = [
            transforms.Resize(config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5)]
        # transforms.Pad(10),
        # transforms.RandomCrop(config.image_size)
        if config.use_colorjitor: # use colorjitor
            transform_train.append(transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_train.extend([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if config.use_rea: # use rea
            transform_train.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
        self.transform_train = transforms.Compose(transform_train)

        # resize --> totensor --> norm
        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            # transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.datasets = ['market', 'duke', 'msmt', 'occluded_reid', 'pduke', 'pduke_zhedang', 'pduke_quanshen', 'njust_win', 'njust_spr', 'njust_both', 'wildtrack']

        # dataset
        self.market_path = config.market_path
        self.duke_path = config.duke_path
        self.pduke_path = config.pduke_path
        self.msmt_path = config.msmt_path
        self.partial_reid_path = config.partial_reid_path
        self.occluded_reid_path = config.occluded_reid_path
        self.partial_ilids_path = config.partial_ilids_path
        self.combine_all = config.combine_all
        self.train_quanshen_dataset = config.train_quanshen_dataset
        self.train_zhedang_dataset = config.train_zhedang_dataset
        self.test_dataset = config.test_dataset


        # batch size
        self.p = config.p
        self.k = config.k

        # load
        self._load()


    def _load(self):

        '''init train dataset'''
        train_quanshen_samples = self._get_train_samples(self.train_quanshen_dataset)
        self.train_quanshen_iter = self._get_uniform_iter_quanshen(train_quanshen_samples, self.transform_train, self.p, self.k)

        '''init test dataset'''
        if self.test_dataset == 'market':
            self.market_query_samples, self.market_gallery_samples = self._get_test_samples('market')
            self.market_query_loader = self._get_loader(self.market_query_samples, self.transform_test, 128)
            self.market_gallery_loader = self._get_loader(self.market_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'occluded_reid':
            self.occluded_reid_query_samples, self.occluded_reid_gallery_samples =self._get_test_samples('occluded_reid')
            self.occluded_reid_query_loader = self._get_loader(self.occluded_reid_query_samples, self.transform_test, 128)
            self.occluded_reid_gallery_loader = self._get_loader(self.occluded_reid_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'pduke':
            self.pduke_query_samples, self.pduke_gallery_samples = self._get_test_samples('pduke')
            self.pduke_query_loader = self._get_loader(self.pduke_query_samples, self.transform_test, 128)
            self.pduke_gallery_loader = self._get_loader(self.pduke_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'duke':
            self.duke_query_samples, self.duke_gallery_samples = self._get_test_samples('duke')
            self.duke_query_loader = self._get_loader(self.duke_query_samples, self.transform_test, 128)
            self.duke_gallery_loader = self._get_loader(self.duke_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'partial_reid':
            self.partial_reid_query_samples, self.partial_reid_gallery_samples = self._get_test_samples('partial_reid')
            self.partial_reid_query_loader = self._get_loader(self.partial_reid_query_samples, self.transform_test,128)
            self.partial_reid_gallery_loader = self._get_loader(self.partial_reid_gallery_samples,self.transform_test, 128)
        elif self.test_dataset == 'partial_ilids':
            self.partial_ilids_query_samples, self.partial_ilids_gallery_samples = self._get_test_samples('partial_ilids')
            self.partial_ilids_query_loader = self._get_loader(self.partial_ilids_query_samples, self.transform_test, 128)
            self.partial_ilids_gallery_loader = self._get_loader(self.partial_ilids_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'msmt':
            self.msmt_query_samples, self.msmt_gallery_samples = self._get_test_samples('msmt')
            self.msmt_query_loader = self._get_loader(self.msmt_query_samples, self.transform_test, 128)
            self.msmt_gallery_loader = self._get_loader(self.msmt_gallery_samples, self.transform_test, 128)
        elif 'njust' in self.test_dataset:
            self.njust_query_samples, self.njust_gallery_samples = self._get_test_samples(self.test_dataset)
            self.njust_query_loader = self._get_loader(self.njust_query_samples, self.transform_test, 128)
            self.njust_gallery_loader = self._get_loader(self.njust_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'wildtrack':
            self.wildtrack_query_samples, self.wildtrack_gallery_samples = self._get_test_samples(self.test_dataset)
            self.wildtrack_query_loader = self._get_loader(self.wildtrack_query_samples, self.transform_test, 128)
            self.wildtrack_gallery_loader = self._get_loader(self.wildtrack_gallery_samples, self.transform_test, 128)


    def _get_train_samples(self, train_dataset):
        '''get train samples, support multi-dataset'''
        samples_list = []
        for a_train_dataset in train_dataset:
            if a_train_dataset == 'market':
                samples = Samples4Market(self.market_path, relabel=True, combineall=self.combine_all).train
            elif a_train_dataset == 'pduke_quanshen':
                samples = Samples4PDuke(self.pduke_path, relabel=True).traing
            elif a_train_dataset == 'pduke_zhedang':
                samples = Samples4PDuke(self.pduke_path,relabel=True).trainp
            elif a_train_dataset == 'duke':
                samples = Samples4Duke(self.duke_path, relabel=True, combineall=self.combine_all).train
            elif a_train_dataset == 'msmt':
                samples = Samples4MSMT17(self.msmt_path, relabel=True, combineall=self.combine_all).train
            elif 'njust' in a_train_dataset:
                assert a_train_dataset in ['njust_win', 'njust_spr', 'njust_both']
                season = a_train_dataset.split('_')[1]
                samples = Samples4NJUST365(self.njust_path, relabel=True, combineall=self.combine_all, season=season).train
            samples_list.append(samples)
        if len(train_dataset) > 1:
            samples = combine_samples(samples_list)
            samples = PersonReIDSamples._relabels(None, samples, 1)
            PersonReIDSamples._show_info(None, samples, samples, samples, name=str(train_dataset))
        return samples

    def _get_test_samples(self, test_dataset):
        if test_dataset == 'market':
            market = Samples4Market(self.market_path, relabel=True, combineall=self.combine_all)
            query_samples = market.query
            gallery_samples = market.gallery
        if test_dataset == 'pduke':
            pduke = Samples4PDuke(self.pduke_path, relabel=True)
            query_samples = pduke.query
            gallery_samples = pduke.gallery
        if test_dataset == 'partial_reid':
            partial_reid = Samples4Partial(self.partial_reid_path, relabel=True)
            query_samples = partial_reid.query
            gallery_samples = partial_reid.gallery
        if test_dataset == 'partial_ilids':
            partial_ilids = Samples4Ilids(self.partial_ilids_path)
            query_samples = partial_ilids.query
            gallery_samples = partial_ilids.gallery
        if test_dataset == 'occluded_reid':
            occluded_reid = Samples4Oreid(self.occluded_reid_path)
            query_samples = occluded_reid.query
            gallery_samples = occluded_reid.gallery
        elif test_dataset == 'duke':
            duke = Samples4Duke(self.duke_path, relabel=True, combineall=self.combine_all)
            query_samples = duke.query
            gallery_samples = duke.gallery
        elif 'msmt' in test_dataset:
            msmt = Samples4MSMT17(self.msmt_path, combineall=self.combine_all)
            query_samples = msmt.query
            gallery_samples = msmt.gallery
        elif 'njust' in test_dataset:
            assert test_dataset in ['njust_win', 'njust_spr', 'njust_both']
            season = test_dataset.split('_')[1]
            njust = Samples4NJUST365(self.njust_path, combineall=self.combine_all, season=season)
            query_samples = njust.query
            gallery_samples = njust.gallery
        elif test_dataset == 'wildtrack':
            wildtrack = Samples4WildTrack(self.wildtrack_path)
            query_samples = wildtrack.query_samples
            gallery_samples = wildtrack.gallery_samples
        return query_samples, gallery_samples

    def _get_uniform_iter_quanshen(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        '''
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False, sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)
        return iters

    def _get_uniform_iter_zhedang(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        '''
        dataset = PersonReIDDataSet_zhedang(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False, sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)
        return iters

    def _get_random_iter(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        iters = IterLoader(loader)
        return iters

    def _get_random_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        return loader

    def _get_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet_test(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader
