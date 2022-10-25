import numpy as np
import copy
from tools import *
import os
import copy
import math
from PIL import Image
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from easydict import EasyDict
import random
import torchvision.transforms as transforms

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class PersonReIDSamples:

    def _relabels(self, samples, label_index):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        ids = []
        for sample in samples:
            ids.append(sample[label_index])
        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()
        # reorder
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples

    def _load_images_path(self, folder_dir):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_jpg_path1(self,folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_jpg_name1(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_jpg_path0(self,folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_jpg_name0(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_tif_path1(self,folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.tif' in file_name:
                identi_id, camera_id = self._analysis_tif_name1(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_tif_path0(self,folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.tif' in file_name:
                identi_id, camera_id = self._analysis_tif_name0(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_pduke_gimages_path(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, Olable = self._analysis_pdukeg_name(file_name)
                samples.append([root_path + file_name, identi_id, Olable])
        return samples

    def _load_pduke_pimages_path(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, Olable = self._analysis_pdukep_name(file_name)
                samples.append([root_path + file_name, identi_id, Olable])
        return samples

    def _load_pduke_query_images_path(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_query_pduke_name(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_pduke_gallery_images_path(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_gallery_pduke_name(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_tif_path1(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.tif' in file_name:
                identi_id, camera_id = self._analysis_tif_name1(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_tif_path0(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.tif' in file_name:
                identi_id, camera_id = self._analysis_tif_name0(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _analysis_tif_name1(self, file_name):
        split_list = file_name.replace('.tif', '').split('_')
        identi_id, camera_id = int(split_list[0]), 1
        return identi_id, camera_id

    def _analysis_tif_name0(self, file_name):
        split_list = file_name.replace('.tif', '').split('_')
        identi_id, camera_id = int(split_list[0]), 0
        return identi_id, camera_id

    def _load_jpg1_path(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_jpg_name1(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _load_jpg0_path(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_jpg_name0(file_name)
                samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return: 0844, 3
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

    def _analysis_tif_name(self, file_name):
        split_list = file_name.replace('.tif', '').split('_')
        identi_id = int(split_list[0])
        return identi_id

    def _analysis_pdukeg_name(self,file_name):
        split_list = file_name.replace('jpg', '').split('_')
        identi_id, Olable = int(split_list[0]), 1
        return identi_id, Olable

    def _analysis_pdukep_name(self,file_name):
        split_list = file_name.replace('jpg', '').split('_')
        identi_id, Olable = int(split_list[0]), 1
        return identi_id, Olable

    def _analysis_query_pduke_name(self,file_name):
        split_list = file_name.replace('jpg', '').split('_')
        identi_id, camera_id = int(split_list[0]), 1
        return identi_id, camera_id

    def _analysis_gallery_pduke_name(self,file_name):
        split_list = file_name.replace('jpg', '').split('_')
        identi_id, camera_id = int(split_list[0]), 0
        return identi_id, camera_id

    def _analysis_jpg_name1(self, file_name):
        split_list = file_name.replace('.jpg', '').split('_')
        identi_id, camera_id = int(split_list[0]), 1
        return identi_id, camera_id

    def _analysis_jpg_name0(self, file_name):
        split_list = file_name.replace('.jpg', '').split('_')
        identi_id, camera_id = int(split_list[0]), 0
        return identi_id, camera_id

    # def _show_info(self, train, query, gallery, name=None):
    #     def analyze(samples):
    #         pid_num = len(set([sample[1] for sample in samples]))
    #         cid_num = len(set([sample[2] for sample in samples]))
    #         sample_num = len(samples)
    #         return sample_num, pid_num, cid_num
    #
    #     train_info = analyze(train)
    #     query_info = analyze(query)
    #     gallery_info = analyze(gallery)
    #
    #     # please kindly install prettytable: ```pip install prettyrable```
    #     table = PrettyTable(['set', 'images', 'identities', 'cameras'])
    #     table.add_row([self.__class__.__name__ if name is None else name, '', '', ''])
    #     table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])
    #     table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2])])
    #     table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])
    #     print(table)

class Samples4PDuke(PersonReIDSamples):
    '''
    PDuke Dataset
    '''
    def __init__(self, pduke_path, relabel=True, combineall=False):

        # parameters
        self.pduke_path = pduke_path
        self.relabel = relabel
        self.combineall = combineall

        # paths of traing,trainp,query，gallery
        traing_path = os.path.join(self.pduke_path, 'train/whole_body_images/')
        trainp_path = os.path.join(self.pduke_path, 'train/occluded_body_images/')
        query_path = os.path.join(self.pduke_path, 'test/occluded_body_images/')
        gallery_path = os.path.join(self.pduke_path, 'test/whole_body_images/')

        # load
        traing = self._load_pduke_gimages_path(traing_path)
        trainp = self._load_pduke_pimages_path(trainp_path)
        query = self._load_pduke_query_images_path(query_path)
        gallery = self._load_pduke_gallery_images_path(gallery_path)
        if self.combineall:
            traing += copy.deepcopy(query) + copy.deepcopy(gallery)
            trainp += traing

         # reorder person identities
        if self.relabel:
            traing = self._relabels(traing, 1)
            trainp = self._relabels(trainp, 1)
        self.traing, self.trainp, self.query, self.gallery = traing, trainp, query, gallery
        # self._show_info(traing, trainp, query, gallery)

class Samples4Oreid(PersonReIDSamples):
    '''
    occluded Dataset
    '''
    def __init__(self, occluded_reid_path):

        # parameters
        self.occluded_reid_path = occluded_reid_path

        #path of query and gallery
        query_path = os.path.join(self.occluded_reid_path, 'occluded_body_images/')
        gallery_path = os.path.join(self.occluded_reid_path, 'whole_body_images/')

        # load
        query = self._load_tif_path1(query_path)
        gallery = self._load_tif_path0(gallery_path)
        self.query, self.gallery = query, gallery

class Samples4Partial(PersonReIDSamples):

    def __init__(self, partial_reid_path, relabel=True, combineall=False):

        # parameters
        self.partial_reid_path = partial_reid_path
        self.relabel = relabel
        self.combineall = combineall

        # path of query and gallery
        query_path = os.path.join(self.partial_reid_path, 'occluded_body_images/')
        gallery_path = os.path.join(self.partial_reid_path, 'whole_body_images/')

        # load
        query = self._load_jpg_path1(query_path)
        gallery = self._load_jpg_path0(gallery_path)
        self.query, self.gallery = query, gallery

class Samples4Ilids(PersonReIDSamples):

    def __init__(self, partial_ilids_path, relabel=True, combineall=False):

        # parameters
        self.partial_ilids_path = partial_ilids_path
        self.relabel = relabel
        self.combineall = combineall

        # path of query and gallery
        query_path = os.path.join(self.partial_ilids_path, 'Probe/')
        gallery_path = os.path.join(self.partial_ilids_path, 'Gallery/')

        # load
        query = self._load_jpg_path1(query_path)
        gallery = self._load_jpg_path0(gallery_path)
        self.query, self.gallery = query, gallery


class Samples4Market(PersonReIDSamples):
    '''
    Market Dataset
    '''
    def __init__(self, market_path, relabel=True, combineall=False):

        # parameters
        self.market_path = market_path
        self.relabel = relabel
        self.combineall = combineall

        # paths of train, query and gallery
        train_path = os.path.join(self.market_path, 'bounding_box_train/')
        query_path = os.path.join(self.market_path, 'query/')
        gallery_path = os.path.join(self.market_path, 'bounding_box_test/')

        # load
        train = self._load_images_path(train_path)
        query = self._load_images_path(query_path)
        gallery = self._load_images_path(gallery_path)
        if self.combineall:
            train += copy.deepcopy(query) + copy.deepcopy(gallery)

        # reorder person identities
        if self.relabel:
            train = self._relabels(train, 1)
        self.train, self.query, self.gallery = train, query, gallery
        # self._show_info(train, query, gallery)

class Samples4Duke(PersonReIDSamples):
    '''
    Duke dataset
    '''
    def __init__(self, market_path, relabel=True, combineall=False):

        # parameters
        self.market_path = market_path
        self.relabel = relabel
        self.combineall = combineall

        # paths of train, query and gallery
        train_path = os.path.join(self.market_path, 'bounding_box_train/')
        query_path = os.path.join(self.market_path, 'query/')
        gallery_path = os.path.join(self.market_path, 'bounding_box_test/')

        # load
        train = self._load_images_path(train_path)
        query = self._load_images_path(query_path)
        gallery = self._load_images_path(gallery_path)
        if self.combineall:
            train += copy.deepcopy(query) + copy.deepcopy(gallery)
        # reorder person identities
        if self.relabel:
            train = self._relabels(train, 1)
        self.train, self.query, self.gallery = train, query, gallery
        # self._show_info(train, query, gallery)

    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0002_c1_f0044158.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id


class Samples4MSMT17(PersonReIDSamples):
    '''
    load MSMT17 dataset
    '''
    def __init__(self, path, relabel=True, combineall=False):
        list_train_path = os.path.join(path, 'list_train.txt')
        list_val_path = os.path.join(path, 'list_val.txt')
        list_query_path = os.path.join(path, 'list_query.txt')
        list_gallery_path = os.path.join(path, 'list_gallery.txt')

        train = self._load_list(os.path.join(path, 'train/'), list_train_path)
        val = self._load_list(os.path.join(path, 'train/'), list_val_path)
        query = self._load_list(os.path.join(path, 'test/'), list_query_path)
        gallery = self._load_list(os.path.join(path, 'test/'), list_gallery_path)
        train = copy.deepcopy(train) + copy.deepcopy(val)
        if combineall:
            train = combine_samples([copy.deepcopy(train), copy.deepcopy(query)+copy.deepcopy(gallery)])
        if relabel:
            train = self._relabels(train, 1)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def _load_list(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        data = []
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            img_path = os.path.join(dir_path, img_path)
            data.append([img_path, pid, camid])
        return data


class Samples4NJUST365(PersonReIDSamples):
    '''
    load NJUST dataset
    '''
    def __init__(self, path, relabel=True, combineall=False, season=''):
        assert season in ['win', 'spr', 'both']
        # winter setting
        if season == 'win' or season == 'both':
            train_path = os.path.join(path, 'copy_dataset_win_train_backup/')
            query_path = os.path.join(path, 'copy_dataset_win_test_query/')
            gallery_path = os.path.join(path, 'copy_dataset_win_test_gallery/')
            train = self._load_images_path(train_path)
            query = self._load_querygallery_images_path(query_path)
            gallery = self._load_querygallery_images_path(gallery_path)
            if combineall:
                train += copy.deepcopy(query) + copy.deepcopy(gallery)
            self.win = EasyDict()
            self.win.train, self.win.query, self.win.gallery = \
                copy.deepcopy(train), copy.deepcopy(query), copy.deepcopy(gallery)
            self.train, self.query, self.gallery = \
                copy.deepcopy(train), copy.deepcopy(query), copy.deepcopy(gallery)
        # spring setting
        if season == 'spr' or season == 'both':
            train_path = os.path.join(path, 'copy_dataset_spr_train/')
            query_path = os.path.join(path, 'copy_dataset_spr_test_query/')
            gallery_path = os.path.join(path, 'copy_dataset_spr_test_gallery/')
            train = self._load_images_path(train_path)
            query = self._load_querygallery_images_path(query_path)
            gallery = self._load_querygallery_images_path(gallery_path)
            if combineall:
                train += copy.deepcopy(query) + copy.deepcopy(gallery)
            if relabel:
                train = self._relabels(train, 1)
            self.spr = EasyDict()
            self.spr.train, self.spr.query, self.spr.gallery = \
                copy.deepcopy(train), copy.deepcopy(query), copy.deepcopy(gallery)
            self.train, self.query, self.gallery = \
                copy.deepcopy(train), copy.deepcopy(query), copy.deepcopy(gallery)
        #
        if season == 'both':
            print('combine njust_spr and njust_win ')
            # self.train = copy.deepcopy(self.win.train) + copy.deepcopy(self.spr.train)
            # self.query = copy.deepcopy(self.win.query) + copy.deepcopy(self.spr.query)
            # self.gallery = copy.deepcopy(self.win.gallery) + copy.deepcopy(self.spr.gallery)
            self.train = combine_samples([copy.deepcopy(self.win.train), copy.deepcopy(self.spr.train)])
            self.query = combine_samples([copy.deepcopy(self.win.query), copy.deepcopy(self.spr.query)])
            self.gallery = combine_samples([copy.deepcopy(self.win.gallery), copy.deepcopy(self.spr.gallery)])
        if relabel:
            self.train = self._relabels(self.train, 1)
        self._show_info(self.train, self.query, self.gallery, name='NJUST365-{}'.format(season))

    def _load_images_path(self, folder_dir):
        _, sub_dirs, _ = os_walk(folder_dir)
        samples = []
        for sub_dir in sub_dirs:
            _, _, images = os_walk(os.path.join(folder_dir, sub_dir))
            for image in images:
                if '.jpg' in image or '.png' in image:
                    pid = int(sub_dir)
                    try:
                        cid = int(image[5:7])
                    except:
                        continue
                    samples.append([os.path.join(os.path.join(folder_dir, sub_dir), image), pid, cid])
        return samples

    def _load_querygallery_images_path(self, folder_dir):
        _, _, images = os_walk(folder_dir)
        samples = []
        for image in images:
            if '.jpg' in image:
                pid = int(image.split('_')[0])
                cid = int(image.split('_')[1][5:7])
                samples.append([os.path.join(folder_dir, image), pid, cid])
        return samples


class Samples4WildTrack(PersonReIDSamples):

    def __init__(self, folder_dir):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, id_paths, _ = os_walk(folder_dir)
        for id_path in id_paths:
            # print(root_path, id_path)
            _, _, files_name = os_walk(os.path.join(root_path, id_path))
            for file_name in files_name:
                if '.png' or '.jpg' in file_name:
                    camera_id = self._analysis_file_name(file_name)
                    img_path = os.path.join(root_path, id_path, file_name)
                    # size = Image.open(img_path).size
                    samples.append([img_path, int(id_path), int(camera_id)])

        random.seed('wildtrack')
        self.query_samples, self.gallery_samples = random.choices(samples, k=1000), samples
        self._show_info(samples, self.query_samples, self.gallery_samples)

    def _analysis_file_name(self, file_name):
        camera_id = file_name.replace('.png', '').split('_')[0]
        return camera_id

    # def _vis_size_distri(self, samples):
    #     pixels = [sample[3][1]*sample[3][0] for sample in samples]
    #     plt.figure()
    #     plt.hist(pixels, 500, range=[0,200000])
    #     plt.savefig('./pixel_distri.png')


def combine_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''
    all_samples = []
    max_pid, max_cid = 0, 0
    for samples in samples_list:
        for a_sample in samples:
            img_path = a_sample[0]
            pid = max_pid + a_sample[1]
            cid = max_cid + a_sample[2]
            all_samples.append([img_path, pid, cid])
        max_pid = max([sample[1] for sample in all_samples])
        max_cid = max([sample[2] for sample in all_samples])
    return all_samples



class PersonReIDDataSet:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.transform2 = RandomErasing(probability=1, mean=[0.485, 0.456, 0.406])
        self.transform3 = LRRandomErasing()
        self.TT =transforms.ToTensor()
        self.dd = transforms.ToPILImage()
        self.transform_test = transforms.Compose([
            transforms.Resize([256, 128], interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])
        a = random.randint(0, 12935)
        while index == a:
            a = random.randint(0, 12935)
        that_sample = copy.deepcopy(self.samples[a])


        this_sample[0] = self._loader(this_sample[0])

        this_sample[0].save('./aaa.jpg')
        fff = copy.deepcopy(this_sample[0])
        www = copy.deepcopy(this_sample[0])
        ddd = copy.deepcopy(this_sample[0])
        that_sample[0] = self._loader(that_sample[0])
        eee = copy.deepcopy(that_sample[0])
        that_sample[0].save('./bbb.jpg')

        img_paste = self.get_zuoibiao(eee, ddd)
        img_paste.save('./abn.jpg')
        this_sample[0].save('./sss.jpg')



        this_sample[0] = self.transform(this_sample[0])
        qqq = self.transform(www)
        ppp = self.dd(qqq)
        ppp.save('./ppp.jpg')
        this_sample[1] = np.array(this_sample[1])
        this_sample[2] = self.transform(img_paste)
        this_sample.append(1)
        this_sample[3] = self.transform_test(fff)
        this_sample.append(1)
        this_sample[4] = self.transform_test(img_paste)

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')
    def get_zuoibiao(self,img,img2):
        img_tensor = self.TT(img)
        zuobiao = self.transform3(img_tensor)
        img = img.crop(zuobiao)
        img2.paste(img, zuobiao)
        return img2

# class PersonReIDDataSet:
#
#     def __init__(self, samples, transform):
#         self.samples = samples
#         self.transform = transform
#         self.transform2 = RandomErasing(probability=1, mean=[0.485, 0.456, 0.406])
#         self.transform3 = LRRandomErasing()
#         self.TT =transforms.ToTensor()
#
#     def __getitem__(self, index):
#
#         this_sample = copy.deepcopy(self.samples[index])
#         a = random.randint(0, 10000)
#         while index == a:
#             a = random.randint(0, 10000)
#         that_sample = copy.deepcopy(self.samples[a])
#
#
#         this_sample[0] = self._loader(this_sample[0])
#
#         this_sample[0].save('./aaa.jpg')
#         ddd = copy.deepcopy(this_sample[0])
#         that_sample[0] = self._loader(that_sample[0])
#         eee = copy.deepcopy(that_sample[0])
#         that_sample[0].save('./bbb.jpg')
#
#         img_paste = self.get_zuoibiao(eee, ddd)
#         img_paste.save('./abn.jpg')
#         this_sample[0].save('./sss.jpg')
#
#
#
#         this_sample[0] = self.transform(this_sample[0])
#         this_sample[1] = np.array(this_sample[1])
#         this_sample[2] = self.transform(img_paste)
#
#         return this_sample
#
#     def __len__(self):
#         return len(self.samples)
#
#     def _loader(self, img_path):
#         return Image.open(img_path).convert('RGB')
#     def get_zuoibiao(self,img,img2):
#
#
#         img_tensor = self.TT(img)
#         zuobiao = self.transform3(img_tensor)
#         img = img.crop(zuobiao)
#         img2.paste(img, zuobiao)
#         return img2

class PersonReIDDataSet_zhedang:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.transform2 =transforms.Compose(RandomErasing(probability=0, mean=[0.485, 0.456, 0.406]))


    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])

        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')

class PersonReIDDataSet_test:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.transform2 =transforms.Compose(RandomErasing(probability=0, mean=[0.485, 0.456, 0.406]))


    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])

        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])
        this_sample[2] = np.array(this_sample[2])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')