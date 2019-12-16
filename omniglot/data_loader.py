import argparse
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from omniglot import OmniglotImage, OmniglotInfo

def drawer_separation():
    drawer = [i for i in range(1, 21)]
    random.shuffle(drawer)

    drawer_train = drawer[:12]
    drawer_valid = drawer[12:16]
    drawer_test = drawer[16:]

    return drawer_train, drawer_valid, drawer_test


def eval_separation():
    info = OmniglotInfo()
    eval_dirs = info.evaluation
    random.shuffle(eval_dirs)

    valid_dirs = eval_dirs[:10]
    test_dirs = eval_dirs[10:]

    return valid_dirs, test_dirs


def preprocess_train(drawer_train, dir_path='/data/omniglot', num=30000):
    info = OmniglotInfo()

    # default 30,000 pairs - 15,000 true, 15,000 false
    pairs = []

    for i in range(int(num/2)):
        language = random.choice(info.background)
        char = random.choice(info.char[language])
        drawer1, drawer2 = np.random.choice(drawer_train, 2, replace=False)

        pairs.append([language, language, char, char, drawer1, drawer2])

    # same language, different char pairs
    for i in range(int(num/6)) :
        language = random.choice(info.background)
        char1 = random.choice(info.char[language])
        char2 = random.choice(info.char[language])
        while char2 == char1 :
            char2 = random.choice(info.char[language])
        assert char1 != char2
        drawer1, drawer2 = np.random.choice(drawer_train, 2)

        pairs.append([language, language, char1, char2, drawer1, drawer2])

    # different language, diffrent char pairs
    for i in range(int(num/3)) :
        language1 = random.choice(info.background)
        language2 = random.choice(info.background)
        while language2 == language1 :
            language2 = random.choice(info.background)
        char1 = random.choice(info.char[language1])
        char2 = random.choice(info.char[language2])
        drawer1, drawer2 = np.random.choice(drawer_train, 2)

        pairs.append([language1, language2, char1, char2, drawer1, drawer2])
    
    random.shuffle(pairs)
    
    images = torch.zeros(num, 2, 1, info.image_size, info.image_size)
    labels = torch.zeros(num, 1)
    count = 0
    for pair in pairs:
        # pair = [language1, language2, char1, char2, drawer1, drawer2]
        if pair[0] == pair[1] and pair[2] == pair[3] :
            labels[count] = torch.FloatTensor([1.])
        else :
            labels[count] = torch.FloatTensor([0.])

        img1 = OmniglotImage('background')
        img2 = OmniglotImage('background')
        img1.get_info(pair[0], pair[2], pair[4])
        img2.get_info(pair[1], pair[3], pair[5])

        # You can define your own transformations on this part
        img1 = img1.get_image()
        img2 = img2.get_image()

        assert img1.shape == img2.shape

        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        images[count, 0] = img1
        images[count, 1] = img2
        
        count += 1

    return images, labels, pairs


def preprocess_verification(drawer_evaltype, languages_range, dir_name='./data/omniglot', num=400) :
    info = OmniglotInfo()

     # 400 pairs
    pairs = []

    for i in range(int(num/2)) :
        language = random.choice(languages_range)
        char = random.choice(info.char[language])
        drawer1, drawer2 = np.random.choice(drawer_evaltype, 2, replace=False)

        pairs.append([language, language, char, char, drawer1, drawer2])

    for i in range(int(num/2)) :
        language1 = random.choice(info.evaluation)
        language2 = random.choice(info.evaluation)
        char1 = random.choice(info.char[language1])
        char2 = random.choice(info.char[language2])
        while language1 == language2 and char1 == char2 :
            language2 = random.choice(info.evaluation)
            char2 = random.choice(info.char[language2])
        
        drawer1, drawer2 = np.random.choice(drawer_evaltype, 2)

        pairs.append([language1, language2, char1, char2, drawer1, drawer2])

    random.shuffle(pairs)

    images = torch.zeros(num, 2, 1, info.image_size, info.image_size)
    labels = torch.zeros(num, 1)
    count = 0
    for pair in pairs :
        # pair = [language1, language2, char1, char2, drawer1, drawer2]
        if pair[0] == pair[1] and pair[2] == pair[3] :
            labels[count] = torch.FloatTensor([1.])
        else :
            labels[count] = torch.FloatTensor([0.])

        img1 = OmniglotImage('evaluation')
        img2 = OmniglotImage('evaluation')
        img1.get_info(pair[0], pair[2], pair[4])
        img2.get_info(pair[1], pair[3], pair[5])
        
        img1 = img1.get_image()
        img2 = img2.get_image()
        
        assert img1.shape == img2.shape

        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        images[count, 0] = img1
        images[count, 1] = img2
        
        count += 1
    
    return images, labels, pairs


def check_preprocess(dir_path, train_num=30000, eval_num=400):
    drawer_train, drawer_valid, drawer_test = drawer_separation()
    valid_dirs, test_dirs = eval_separation()
    # print(valid_dirs)
    # print(test_dirs)
    train_images, train_labels, train_pairs = preprocess_train(drawer_train, dir_path=dir_path)
    # print(train_images.size())
    # print(train_labels.size())

    verification_images, verification_labels, verification_pairs = preprocess_verification(drawer_valid, valid_dirs, dir_path, eval_num)
    # print(verification_images.size())
    # print(verification_labels.size())

    return train_images, train_labels, verification_images, verification_labels


def preprocess(dir_path, train_num=30000, eval_num=400):
    train_images, train_labels, verification_images, verification_labels = check_preprocess(dir_path, train_num, eval_num)

    train_dataset = TensorDataset(train_images, train_labels)
    verification_dataset = TensorDataset(verification_images, verification_labels)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    verification_loader = DataLoader(verification_dataset, batch_size=1, shuffle=False, drop_last=False)

    return train_loader, verification_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/omniglot',
                        help='omniglot dataset directory path')
    args = parser.parse_args()

    check_preprocess(args.data_dir)
