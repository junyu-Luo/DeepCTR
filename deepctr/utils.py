# -*- coding: utf-8 -*-
'''
@Time  : 2022/11/11
@Author: junyuluo
'''

import json
import time
import pickle
import random
import datetime
import os


def show_model_structure(model, path='model.png'):
    import tensorflow as tf
    tf.keras.utils.plot_model(
        model,
        to_file=path,
        show_shapes=True,
    )
    print('save image done in {}'.format(path))


def get_date(ndays=-1, base_date=None, fmt='%Y%m%d', base_date_fmt=None):
    ''' 获取日期
        get_date() == get_date(-1) == "20220829"  # 返回当前时间(20220830)的昨天
        get_date(-2, "20220103") == "20220101"  # 返回base_date的2天前
    '''
    if base_date is None:
        try:
            import pytz
            base_datetime = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
        except Exception:
            base_datetime = datetime.datetime.now()
    else:
        base_datetime = datetime.datetime.strptime(base_date, base_date_fmt or fmt)
    return datetime.datetime.strftime(base_datetime + datetime.timedelta(ndays), fmt)


def date_range(start, end, end_include=False, step=1, fmt="%Y%m%d"):
    ''' 日期版的range
        date_range("20220101", "20220102") == ["20220101"]
        date_range("20220101", "20220102", True) == ["20220101", "20220102"]
    '''
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, fmt) - strptime(start, fmt)).days
    days = days + int(step / abs(step)) if end_include else days  # +1 OR -1
    return [strftime(strptime(start, fmt) + datetime.timedelta(i), fmt) for i in range(0, days, step)]


def read_json(file, encoding='utf-8', path=''):
    data = open(os.path.join(path, file), encoding=encoding).read()
    return json.loads(data)


def save_json(file, content, path='', mode='w'):
    f = open(os.path.join(path, file), mode, encoding='utf-8')
    json_str = json.dumps(content, indent=2, ensure_ascii=False)
    f.write(json_str)
    f.close()


def read_in_block(file, path=''):
    with open(os.path.join(path, file), "r", encoding="utf-8") as f:
        while True:
            block = f.readline()  # 每次读取固定长度到内存缓冲区
            if block:
                yield block
            else:
                return  # 如果读取到文件末尾，则退出


def read_txt(file, path=''):
    """
        读取txt文件，默认utf8格式, 不能有空行
    :param file_path: str, 文件路径
    :param encode_type: str, 编码格式
    :return: list
    """
    res = []
    for row in read_in_block(file, path=path):
        res.append(row.strip())
    return res


def save_txt(file, data_list, path='', mode='a+', encoding='utf-8'):
    with open(os.path.join(path, file), mode, encoding=encoding) as f:
        for line in data_list:
            f.write(line + '\n')
    f.close()


def save_str2txt(file, data, path='', mode='a+', encoding='utf-8'):
    writefile = open(os.path.join(path, file), mode, encoding=encoding)
    writefile.write(data + '\n')
    writefile.close()


def read_pkl(file, path=''):
    with open(os.path.join(path, file), 'rb') as f:
        return pickle.load(f)


def save_pkl(file, content, path=''):
    f = open(os.path.join(path, file), 'wb')
    pickle.dump(content, f)
    f.close()


def flatList(l):
    for el in l:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            for sub in flatList(el):
                yield sub
        else:
            yield el


def flatDict(x):
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in flatDict(value):
                k = f'{key}_{k}'
                yield (k, v)
        else:
            yield (key, value)


def dir_file_name(path, add_filter_char=[], sample_num=None):
    import os
    def filter_char(file):
        filter_char_list = ['crc', '_SUCCESS', 'ipynb']
        filter_char_list = filter_char_list + add_filter_char
        for char in filter_char_list:
            if file.find(char) != -1:
                return False
        return True

    pathnames = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            if filter_char(file):
                pathnames.append(file)

    if sample_num and sample_num < len(pathnames):
        pathnames = random.sample(pathnames, sample_num)

    return pathnames


def get_dir_files(path):
    pathnames = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            pathnames += [os.path.join(dirpath, filename)]
    return pathnames


def show_cosed_time(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print('{} took {} seconds'.format(f.__name__, round(end - start, 3)))
        return result

    return wrapper


def yaml2js(file, path=''):
    import yaml
    with open(os.path.join(path, file), encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data


def js2yaml(file, content, path=''):
    import yaml
    with open(os.path.join(path, file), 'w', encoding='utf-8') as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)


def count_list(list_, sort=True, reverse=True):
    from collections import Counter
    count_dict = dict(Counter(list_))
    if sort:
        count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=reverse))
    return count_dict


def list_split_2(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     比例
    :param shuffle:   是否打乱
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        import random
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


def list_split(full_list, split_nunb):
    pass


def GetClassFuncName(ClassName):
    return [a for a in dir(ClassName) if not a.startswith('__')]
