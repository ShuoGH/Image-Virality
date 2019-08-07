import pandas as pd
import numpy as np
import os
from fake_useragent import UserAgent
import requests
import argparse
import sys
import random
from time import sleep


def get_random_header():
    '''
    Using the fake_useragent to randomly choose your useragent to avoid the anti-spider tech.
    You should always update the referer, even when you are doing test.
    '''
    fake_ua = UserAgent()
    user_agent = fake_ua.random

    headers = {
        'User-Agent': user_agent,
    }
    return headers


def random_sleep_time():
    '''
    randomly sleep during the scraping work
    '''
    possibility_like = random.random()
    if possibility_like < 0.2:
        sleep_time = random.randint(5, 10)
    elif possibility_like < 0.8:
        sleep_time = random.randint(1, 5)
    else:
        sleep_time = 1
    return sleep_time


def get_url_list(data_path, num_limit):
    '''
    args:
        data_path: the path of the data set
        num_limit: the number of pictures downloaded (for testing)
    return:
        url_list: pandas.Series
    '''
    if num_limit:
        url_list = pd.read_csv(data_path)['img_link'][:num_limit]
    else:
        url_list = pd.read_csv(data_path)['img_link']
    return url_list


def download_images(url_list, sleep_flag):
    '''
    args:
        the url list which contains the list of urls
    return:
    '''
    current_path = os.getcwd()

    for i, url in enumerate(url_list):
        if sleep_flag:
            sleep(random_sleep_time())
        pic_name = os.path.split(url)[-1]
        pic_path = os.path.join(current_path, 'images', pic_name)
        with open(pic_path, 'wb') as f:
            f.write(requests.get(url, headers=get_random_header()).content)
        print("Downloading {} image, name {}".format(i, pic_name))
    return


def main(args):

    url_list = get_url_list(data_path=args.data_path,
                            num_limit=args.num_images)
    download_images(url_list, args.sleep_flag)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download the images according to the .csv file.')

    parser.add_argument('--num_images', type=int, default=None, metavar='N',
                        help='number of images for downloading (default: no limit)')
    parser.add_argument('--data_path', type=str, default='./data_set_iv.csv',
                        help='path to .csv data set')
    parser.add_argument('--sleep_flag', type=int, default=1,
                        help='whether to sleep during scraping ( 1: True, 0: False, default 1)')

    args = parser.parse_args()

    main(args)
