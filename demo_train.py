# -*- coding: utf-8 -*-

import argparse, os, platform, sys
from sklearn.metrics import roc_auc_score
from deepctr.metrics import parallel_cal_group_auc
from deepctr.data_pipeline import TFrecordBuilder, get_train_features, FeatureColumn
from deepctr.trainer import Trainer, Tester
from deepctr.models.multitask.esmm import ESMM
from deepctr.utils import get_date, date_range, dir_file_name, show_model_structure
import datetime
import tensorflow as tf

# params
parser = argparse.ArgumentParser(description="xxx")

if platform.system().lower() == 'windows':
    # 本地环境
    parser.add_argument('--dt', type=str, default='20230719', required=False, help='today')
    parser.add_argument('--model_id', type=str, default='xx', required=False)
    parser.add_argument('--base_path', type=str, default='./v1/', required=False)
    parser.add_argument('--idp_id', type=int, default=1, required=False)
    parser.add_argument('--train_dt_duration', type=int, default=2, required=False)
    parser.add_argument('--epoch', type=int, default=1, required=False)
    parser.add_argument('--push_env', type=str, default='', required=False, help='上传模型管理平台的环境')

elif platform.system().lower() == 'linux' and 'ipykernel' in sys.argv[0]:
    # jupyter 环境
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    yesterday = yesterday.strftime('%Y%m%d')
    parser.add_argument('--dt', type=str, default='{}'.format(yesterday), required=False, help='today')
    parser.add_argument('--model_id', type=str, default='chatroom_esmm_v1', required=False)
    parser.add_argument('--base_path', type=str, default='/data/rcmd_data/user/ljy/mt/home_page/chatroom/v1/',
                        required=False)
    parser.add_argument('--idp_id', type=int, default=13317, required=False)
    parser.add_argument('--train_dt_duration', type=int, default=14, required=False)
    parser.add_argument('--epoch', type=int, default=15, required=False)
    parser.add_argument('--push_env', type=str, default='stage', required=False, help='上传模型管理平台的环境')

else:
    # idp 环境
    parser.add_argument('--dt', type=str, required=True, help='today')
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--base_path', type=str, default='/data/rcmd_data/user/ljy/chatroom/v1',
                        required=False)
    parser.add_argument('--idp_id', type=int, required=True)
    parser.add_argument('--train_dt_duration', type=int, default=14, required=False)
    parser.add_argument('--epoch', type=int, default=15, required=False)
    parser.add_argument('--push_env', type=str, default='product,stage', required=False, help='上传模型管理平台的环境')

parser.add_argument('--model_metric', type=set, default={}, required=False, help='存储模型评估指标的字典')
parser.add_argument('--decay_epoch', type=int, default=5, required=False, help='超过多少个epoch后，学习率开始衰减')
parser.add_argument('--decay_factor', type=float, default=0.8, required=False, help='超过多少个epoch后，学习率衰减的因子大小')
parser.add_argument('--batch_size', type=int, default=1024, required=False)
parser.add_argument('--learning_rate', type=float, default=0.002, required=False)

parser.add_argument('--if_train', type=int, default=1, required=False)
parser.add_argument('--if_save', type=int, default=0, required=False)
parser.add_argument('--if_test', type=int, default=1, required=False)

parser.add_argument('--verbose', type=int, default=1, required=False, help='0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录')
args, _ = parser.parse_known_args()

args.choose_label = ['label_enter', 'label_gangup']
args.final_label_name = args.choose_label[-1]
args.choose_feat = ['u_algo_mt_app_id', 'e_algo_mt_tag_id_exp', 'e_algo_mt_room_para_tag_list',
                    'e_algo_mt_male_user_cnt', 'e_algo_mt_female_user_cnt', 'e_algo_mt_male_mic_user_cnt',
                    'e_algo_mt_female_mic_user_cnt', 'e_algo_mt_publish_time', 'u_algo_mt_enter_tag_ls_6time',
                    'u_algo_mt_enter_room_para_tag_ls_6time', 'u_algo_mt_gangup_tag_ls_6time',
                    'u_algo_mt_gangup_room_para_tag_ls_6time', 'u_algo_mt_reg_time', 'u_algo_mt_gender',
                    'u_algo_mt_age', 'o_algo_mt_reg_time', 'o_algo_mt_gender', 'o_algo_mt_age',
                    'i_algo_mt_publish_tag_cnt_3w', 'i_algo_mt_exposure_tag_cnt_3w', 'i_algo_mt_click_tag_cnt_3w',
                    'i_algo_mt_stay_tag_dur_3w', 'i_algo_mt_gang_up_tag_cnt_3w', 'i_algo_mt_on_mic_tag_cnt_3w',
                    'i_algo_mt_publish_tag_cnt_1w', 'i_algo_mt_exposure_tag_cnt_1w', 'i_algo_mt_click_tag_cnt_1w',
                    'i_algo_mt_stay_tag_dur_1w', 'i_algo_mt_gang_up_tag_cnt_1w', 'i_algo_mt_on_mic_tag_cnt_1w',
                    'i_algo_mt_publish_tag_cnt_3d', 'i_algo_mt_exposure_tag_cnt_3d', 'i_algo_mt_click_tag_cnt_3d',
                    'i_algo_mt_stay_tag_dur_3d', 'i_algo_mt_gang_up_tag_cnt_3d', 'i_algo_mt_on_mic_tag_cnt_3d',
                    'u_algo_mt_list_tag_exp_cnt_1w', 'u_algo_mt_list_tag_enter_cnt_1w',
                    'u_algo_mt_list_tag_gangup_cnt_1w', 'u_algo_mt_list_tag_duration_1w',
                    'u_algo_mt_list_tag_mic_cnt_1w', 'u_algo_mt_list_tag_exp_room_cnt_1w',
                    'u_algo_mt_list_tag_enter_room_cnt_1w', 'u_algo_mt_list_tag_gangup_room_cnt_1w',
                    'u_algo_mt_list_tag_mic_room_cnt_1w', 'u_algo_mt_tag_enter_cnt_1w', 'u_algo_mt_tag_gangup_cnt_1w',
                    'u_algo_mt_tag_duration_1w', 'u_algo_mt_tag_mic_cnt_1w', 'u_algo_mt_tag_enter_room_cnt_1w',
                    'u_algo_mt_tag_gangup_room_cnt_1w', 'u_algo_mt_tag_mic_room_cnt_1w',
                    'u_algo_mt_list_tag_exp_cnt_3w', 'u_algo_mt_list_tag_enter_cnt_3w',
                    'u_algo_mt_list_tag_gangup_cnt_3w', 'u_algo_mt_list_tag_duration_3w',
                    'u_algo_mt_list_tag_mic_cnt_3w', 'u_algo_mt_list_tag_exp_room_cnt_3w',
                    'u_algo_mt_list_tag_enter_room_cnt_3w', 'u_algo_mt_list_tag_gangup_room_cnt_3w',
                    'u_algo_mt_list_tag_mic_room_cnt_3w']
                    # , 'u_algo_mt_province', 'o_algo_mt_province', 'u_algo_mt_city', 'o_algo_mt_city'


updated_js = os.path.join(args.base_path, 'update_featjs',
                          'feature_{}_{}.json'.format(get_date(-14, args.dt), get_date(-1, args.dt)))
args.conf_js = get_train_features(updated_js, args.choose_label, args.choose_feat)
fc = FeatureColumn(args.conf_js)

builder = TFrecordBuilder(args.conf_js)
args.dataset_path = os.path.join(args.base_path, 'tfrecord')
print('dataset_path', args.dataset_path)

args.model_metric = {}

# 加载数据集和模型save路径
args.save_model_path = os.path.join(args.base_path, 'saved_model', args.dt)

# 模型初始化
model = ESMM(fc.features, task_names=args.choose_label, task_types=['binary'] * len(args.choose_label))

# 打印模型结构 输出图片
# show_model_structure(model, 'model.png')

# 用当天数据，加载昨天的模型进行测试，测试昨天的模型效果咋样
if args.if_test:
    import tensorflow as tf
    from deepctr.layers import custom_objects

    test_path = os.path.join(args.base_path, 'tfrecord', args.dt)

    try:
        yesterday_model_path = args.save_model_path.replace(args.dt, get_date(-1, args.dt))
        yesterday_model = tf.keras.models.load_model(yesterday_model_path, custom_objects=custom_objects)

        test_dataset = builder.generate(test_path, batch_size=args.batch_size)
        y_true, uid_list = builder.get_pred_y_uid(test_dataset, return_dict=False)
        y_pred = yesterday_model.predict(test_dataset, batch_size=1024, verbose=args.verbose)
        auc = roc_auc_score(y_true['label_gangup'], y_score=y_pred)
        gauc = parallel_cal_group_auc(labels=y_true['label_gangup'], preds=y_pred, user_id_list=uid_list)

        args.model_metric['yesterday model {} auc'.format(args.final_label_name)] = round(auc, 5)
        args.model_metric['yesterday model {} gauc'.format(args.final_label_name)] = round(gauc, 5)
        for key, val in args.model_metric.items():
            print(key, val)

    except:
        print('未能加载到昨天模型')

print(args.model_metric)

# trian
if args.if_train:
    args.model_metric = Trainer(args, model, fc)

print(args.model_metric)
