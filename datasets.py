import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from utils import makedir_exist_ok
import random
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


class Reddit_Img_Pair(data.Dataset):
    """
    All the data sets should be the subclass of torch.utils.data.Dataset

    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        re_select (bool, optional): If true, select the images randomly from the original data to combine the pairs
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        pair_mode: 1, 2, 3, or 4, corresponding to four kinds of image pairs. For details, see the  self.re_select().
            only meaningful when the `train=True`.
    """
    training_file = 'training.pt'
    test_file = 'test.pt'
    imgs_file = 'images'

    def __init__(self, root, train=True, transform=None, target_transform=None, re_select=False, pair_mode=1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.pair_mode = pair_mode
        self.training_file = "training_" + str(pair_mode) + ".pt"

        self.df_ori, self.len_ori = self.read_origin_csv()  # the info of the original data

        if re_select:
            self.re_select()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use re_select=True to randomly generate the images pairs from the original image folders')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        # print(data_file)
        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file))  # according to the flag to load train/test file.

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def read_origin_csv(self):
        ori_path = os.path.join(self.root, 'datasets', 'data_set_iv.csv')
        df = pd.read_csv(ori_path)
        return df, df.shape[0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (imgs, target) after transformation and the label
                    indicate whether the first image is more viral than the second one
        """
        [img_id_1, img_id_2], label = self.data[index], int(
            self.targets[index])
        img_name_1 = str(img_id_1) + '.jpg'
        img_name_2 = str(img_id_2) + '.jpg'
        img_1 = Image.open(os.path.join(
            self.root, self.imgs_file, img_name_1))
        img_2 = Image.open(os.path.join(self.root, self.imgs_file, img_name_2))
        img_pair = [img_1, img_2]
        if self.transform is not None:
            img_pair = [self.transform(img) for img in img_pair]
        else:
            # if use this, be careful abou the image size (need to be consistent with the conv layers)
            img_pair = [transforms.functional.to_tensor(
                img) for img in img_pair]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img_pair, label

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def re_select(self):
        """
        Re-select the image to build the pairs for Siamese networks.
        There are four data set which are going to be built (training):
            - 250 images with highest virality score, 250 images with the lowest virality score.
            - random image pairs:
                - 250 highest images with 250 images randomly selected from the lower part
                - 250 images randomly selected from the highest part and 250 least viral images
                - randomly select 250 images from higher part and lower part respectively.
        This is the main function of the data set.
        """
        if self._check_exists():
            return
        makedir_exist_ok(self.processed_folder)

        # select the images pairs randomly
        if self.pair_mode == 1:
            img_pairs, labels = self._build_img_pair_1()
        elif self.pair_mode == 2:
            img_pairs, labels = self._build_img_pair_2()
        elif self.pair_mode == 3:
            img_pairs, labels = self._build_img_pair_3()
        elif self.pair_mode == 4:
            img_pairs, labels = self._build_img_pair_4()

        img_pairs_test, labels_test = self._build_test_pairs()
        # use the tuple to store the image pairs and labels, then save into the processed folder.
        print('Processing... building image pairs')
        data_pair_labels = (img_pairs, labels)
        data_pair_labels_test = (img_pairs_test, labels_test)

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(data_pair_labels, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(data_pair_labels_test, f)

        print('Done!')

    def _build_img_pair_1(self):
        lower_half_id = self.df_ori.iloc[:250]['img_id']
        higher_half_id = self.df_ori.iloc[-250:]['img_id']
        img_pair = []
        label = np.zeros(250)
        for index, (i, j) in enumerate(zip(random.sample(list(lower_half_id), 250), random.sample(list(higher_half_id), 250))):
            if random.random() <= 0.5:
                img_pair.append([j, i])
                label[index] = 1  # opera in_place
            else:
                img_pair.append([i, j])
        return img_pair, label

    def _build_img_pair_2(self):
        half_point = int(self.len_ori / 2)
        lower_half_id = random.sample(
            list(self.df_ori.iloc[:half_point]['img_id']), 250)
        higher_half_id = self.df_ori.iloc[-250:]['img_id']
        img_pair = []
        label = np.zeros(250)
        for index, (i, j) in enumerate(zip(lower_half_id, random.sample(list(higher_half_id)))):
            if random.random() <= 0.5:
                img_pair.append([j, i])
                label[index] = 1  # opera in_place
            else:
                img_pair.append([i, j])
        return img_pair, label

    def _build_img_pair_3(self):
        half_point = int(self.len_ori / 2)
        lower_half_id = self.df_ori.iloc[:250]['img_id']
        higher_half_id = random.sample(
            list(self.df_ori.iloc[half_point:]['img_id']), 250)
        img_pair = []
        label = np.zeros(250)
        for index, (i, j) in enumerate(zip(random.sample(list(lower_half_id)), higher_half_id)):
            if random.random() <= 0.5:
                img_pair.append([j, i])
                label[index] = 1  # opera in_place
            else:
                img_pair.append([i, j])
        return img_pair, label

    def _build_img_pair_4(self):
        half_point = int(self.len_ori / 2)
        lower_half_id = random.sample(
            list(self.df_ori.iloc[:half_point]['img_id']), 250)
        higher_half_id = random.sample(
            list(self.df_ori.iloc[half_point:]['img_id']), 250)

        img_pair = []
        label = np.zeros(250)
        for index, (i, j) in enumerate(zip(lower_half_id, higher_half_id)):
            if random.random() <= 0.5:
                img_pair.append([j, i])
                label[index] = 1  # opera in_place
            else:
                img_pair.append([i, j])
        return img_pair, label

    def _build_test_pairs(self):
        '''
        Recall the function of building the img_pair_4
        '''
        return self._build_img_pair_4()

    def __repr__(self):
        '''
        Show the info of the data set.
        '''
        # pairing_mode_info_list = ['']
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of image pairs: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        fmt_str += 'pairing mode: {}'.format(self.pair_mode)

        return fmt_str


class Reddit_Images_for_Classification(data.Dataset):
    '''
    args:
        train: True, indicate the training data set, False means the test data
        transform: the transforms implemented on the images
        target_transform: for the label
        re_split: split the data set into training and test file
        balance: default False, if True, would use downsample to balance the data set.
    return:
        the data set (training and testing. 3000+, 1000+. If balance = True, it will only contain 500 images in total)
    '''
    imgs_file = 'images'
    training_file = "classification_training_data.pt"
    test_file = "classification_test_data.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, re_split=False, balance=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.balance = balance  # indicate whether do the downsampling

        if self.balance:
            self.training_file = 'balance_' + self.training_file
            self.test_file = 'balance_' + self.test_file

        # two columns : img_id , abel
        self.df_subreddit = self.read_classification_data()

        if re_split:
            self.split_train_test(self.df_subreddit)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use re_select=True to randomly generate the images pairs from the original image folders')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        # print(data_file)
        self.data = torch.load(
            os.path.join(self.processed_folder, data_file))  # according to the flag to load train/test file.

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def __len__(self):
        return self.data.shape[0]

    def read_classification_data(self):
        data_path = os.path.join(self.root, 'datasets',
                                 'image4classification.csv')
        df = pd.read_csv(data_path)
        return df

    def __getitem__(self, index):
        # the label is str (pay attention)
        img_id, img_label = self.data.iloc[index][['img_id', 'label']]

        img_name = str(img_id) + '.jpg'
        img = Image.open(os.path.join(
            self.root, self.imgs_file, img_name))
        if self.transform is not None:
            img = self.transform(img)
        else:
            # if use this, be careful abou the image size (need to be consistent with the conv layers)
            img = transforms.functional.to_tensor(img)

        if self.target_transform is not None:
            img_label = self.target_transform(label)

        return img, img_label

    def split_train_test(self, dataframe):

        if self._check_exists():
            return
        if self.balance:
            dataframe = self.balance_dataframe(dataframe)
        makedir_exist_ok(self.processed_folder)

        train_df, test_df = train_test_split(dataframe, test_size=0.2)
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(train_df, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_df, f)
        print("Split the training and test file successfully! {}".format(
            "It's the balanced data" if self.balance else ""))

    def balance_dataframe(self, dataframe):
        '''
        There are five subreddits in the data set with huge inbalance.
        To balance the data set, downsample.
        100 images for each subreddit.
        '''

        subreddit_dis = [[sub, num] for sub, num in dataframe.groupby(
            ['label']).count()['subreddit'].items()]  # number of each subreddit
        dataframe_list = []
        for sub, num in subreddit_dis:
            index_sample_i = random.sample(range(num), 100)
            dataframe_target = dataframe[dataframe['label']
                                         == sub].iloc[index_sample_i]
            dataframe_list.append(dataframe_target)
        datarame_concanate = pd.concat(
            [i for i in dataframe_list], axis=0, ignore_index=True)
        return datarame_concanate

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def __repr__(self):
        '''
        Show the info of the dataset for the classification task.
        '''
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Subreddits: funny, WTF, aww, atheism, gaming \n'
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Training or Test: {}\n'.format(tmp)
        fmt_str += '    Balance or not: {}\n'.format(self.balance)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
