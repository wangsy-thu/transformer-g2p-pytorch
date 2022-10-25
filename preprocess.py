import json
import cmudict
import random
import os
from config import HP


invalid_chars = ['.', '\'', '-']


def invalid(in_str: str) -> bool:
    for c in invalid_chars:
        if c in in_str:
            return True
    return False


if __name__ == '__main__':
    """
    数据预处理：将带有非法字符的key的条目删除
    """
    if not os.path.isdir(HP.data_dir):
        os.mkdir(HP.data_dir)
    cmu_data_dict = cmudict.dict()
    train_dict = {}
    test_dict = {}
    val_dict = {}
    print('raw data length:{}\n'.format(len(cmu_data_dict)))
    for key in cmu_data_dict.keys():
        if invalid(key) or len(cmu_data_dict[key]) > 1:
            continue
        if random.random() < 0.8:
            train_dict[key] = ' '.join([s for s in list(cmu_data_dict[key])[0]])
        elif 0.8 <= random.random() < 0.9:
            test_dict[key] = ' '.join([s for s in list(cmu_data_dict[key])[0]])
        elif 0.9 <= random.random() <= 1:
            val_dict[key] = ' '.join([s for s in list(cmu_data_dict[key])[0]])
    print('Train size:{}'.format(len(train_dict)))
    print('Test size:{}'.format(len(test_dict)))
    print('Train size:{}'.format(len(val_dict)))

    with open('data/data_train.json', 'w', encoding='utf-8') as fp:
        json.dump(train_dict, fp)

    with open('data/data_test.json', 'w', encoding='utf-8') as fp:
        json.dump(test_dict, fp)

    with open('data/data_val.json', 'w', encoding='utf-8') as fp:
        json.dump(val_dict, fp)

    print("""
==========================================
            Dump Successfully!
==========================================
    """)
