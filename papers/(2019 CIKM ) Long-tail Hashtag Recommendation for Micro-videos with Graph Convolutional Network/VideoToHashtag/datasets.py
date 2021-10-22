import torch
import torch.utils.data as Data
import h5py
import json
import numpy as np
import os


class INSVIDEO(Data.Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.dataset = []
        self.get_dataset()

    def get_dataset(self):
        path_dict = {'train': 'train_data_u_shuffle.json', 'eval': 'video_id-hashtag_id_eval.json',
                     'test': 'video_id-hashtag_id_test.json', 'longtail_test': 'longtail_test.json'}
        dataset_path = os.path.join(self.root, path_dict[self.phase])
        self.dataset = json.loads(json.load(open(dataset_path, 'r', encoding="utf-8")))

        # extracted features
        image_path = os.path.join(self.root, 'images_feature.hdf5')
        audio_path = os.path.join(self.root, 'audios_feature.hdf5')
        text_path = os.path.join(self.root, 'texts_feature.hdf5')

        self.image_data = h5py.File(image_path, 'r')
        self.audio_data = h5py.File(audio_path, 'r')
        self.text_data = h5py.File(text_path, 'r')

        # video-hashtag dict, video-user dict
        v2h_path = os.path.join(self.root, 'video_id-hashtag_id.json')
        v2u_path = os.path.join(self.root, 'video_id-user_id.json')
        self.v2h = json.loads(json.load(open(v2h_path, 'r', encoding="utf-8")))
        self.v2u = json.loads(json.load(open(v2u_path, 'r', encoding="utf-8")))


        self.hashtag_num = 15751

    def __getitem__(self, index):
        # 获取一个数据
        # 再次可以写预处理
        # 返回一个数据对（x,y）
        if self.phase=='train':
            item = self.dataset[index]
            video_id = item[0]
            positive_id = int(item[1])
            negative_id = np.random.randint(self.hashtag_num)
            # 如果重复了就重新获取
            while negative_id in self.v2h[video_id]:
                negative_id = np.random.randint(self.hashtag_num)
            user_id = int(item[2])

            image_feature = self.image_data[video_id][()]
            audio_feature = self.audio_data[video_id][()]
            text_feature = self.text_data[video_id][()]

            return image_feature, audio_feature, text_feature, positive_id, negative_id, user_id
        else:
            item = self.dataset[index]
            key,value=item
            video_id = key
            true_label=value
            user_id=self.v2u[video_id]
            all_hashtag = [i for i in range(self.hashtag_num)]

            image_feature = self.image_data[video_id][()]
            audio_feature = self.audio_data[video_id][()]
            text_feature = self.text_data[video_id][()]

            return image_feature, audio_feature, text_feature, all_hashtag, true_label, user_id

    def __len__(self):
        # 返回数据集的大小
        return len(self.dataset)

