# -*- coding: utf-8 -*-
import shutil
import time
import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from utils import get_date, date_range

from deepctr.utils import read_json, save_json, dir_file_name, list_split_2
from deepctr.data_pipeline import TFrecordBuilder
from deepctr.metrics import parallel_cal_group_auc
from deepctr.warm_up import tfserving_warmup


def push_mvp(args):
    print('---------------------------mvp begin---------------------------')
    try:
        args.push_env
    except:
        args.push_env = ''
    # 默认不上传mvp
    if args.push_env:
        model_info = {
            "model_score": args.model_metric,
            "feature_list": [],
            "file": {
                "name": args.model_id,
                "address": '{}'.format(args.save_model_path),  # obs自行备份文件地址，可乱写但不能不填
                "local_address": args.save_model_path,  # 模型文件绝对路径
                "md5": ' '
            }
        }
        print('model_info: ', model_info)
        # start_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # model_info_to_mvp(model_id=args.model_id, model_info=model_info, task_id=args.idp_id,
        #                   task_name=args.model_id,
        #                   start_date=start_date, env=args.push_env)
        print(model_info)


def Tester(args, model):
    print('---------------------------testing begin---------------------------')
    print('test_path', args.test_path)
    builder = TFrecordBuilder(args.conf_js)
    test_dataset = builder.generate(args.test_path, batch_size=args.batch_size)
    if len(args.choose_label) == 1:
        y_true, uid_list = builder.get_pred_y_uid(test_dataset, return_dict=False)
        y_pred = model.predict(test_dataset, batch_size=1024, verbose=args.verbose)
        auc = roc_auc_score(y_true, y_score=y_pred)
        gauc = parallel_cal_group_auc(labels=y_true, preds=y_pred, user_id_list=uid_list)
        args.model_metric['auc'] = round(auc, 5)
        args.model_metric['gauc'] = round(gauc, 5)
        for key, val in args.model_metric.items():
            print(key, val)
    else:
        y_true, uid_list = builder.get_pred_y_uid(test_dataset, return_dict=False)
        y_pred = model.predict(test_dataset, batch_size=1024, verbose=args.verbose)

        for i, name in enumerate(args.choose_label):
            auc = roc_auc_score(y_true=y_true[name], y_score=y_pred[i])
            gauc = parallel_cal_group_auc(labels=y_true[name], preds=y_pred[i], user_id_list=uid_list)
            args.model_metric['{} auc'.format(name)] = round(auc, 5)
            args.model_metric['{} gauc'.format(name)] = round(gauc, 5)
            for key, val in args.model_metric.items():
                print(key, val)
    print('---------------------------testing done---------------------------')
    return args.model_metric


def Trainer(args, model, fc):
    if len(args.choose_label) == 1:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, name='adam'),
            loss=['binary_crossentropy'],
            metrics=['AUC'])
    else:
        loss = {i: "binary_crossentropy" for i in args.choose_label}
        metrics = {i: "AUC" for i in args.choose_label}
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, name='adam'),
            loss=loss,
            # loss_weights=loss_weights,
            metrics=metrics, )

    print('---------------------------generate dataset begin---------------------------')
    try:
        args.if_timeline
    except:
        args.if_timeline = True

    if args.if_timeline:
        # dt为当前日期，当train_dt_duration=14天时，t-14 ~ t-2 天为训练集，
        # t-1当天分出一部分给训练集，一部分给验证集，t分出一部分给训练集，一部分给测试集
        t_1_train, t_1_dev = list_split_2(dir_file_name(os.path.join(args.dataset_path, get_date(-1, args.dt))),
                                          ratio=0.7)
        t_train, t_test = list_split_2(dir_file_name(os.path.join(args.dataset_path, args.dt)), ratio=0.7)
        args.train_path = [os.path.join(args.dataset_path, x) for x in
                           date_range(get_date(1 - args.train_dt_duration, get_date(-1, args.dt)),
                                      args.dt)] + t_1_train + t_train
        args.dev_path = t_1_dev
        args.test_path = t_test
    else:
        # dt为当前日期，当train_dt_duration=14天
        args.train_path = [os.path.join(args.base_path, 'tfrecord', x)
                           for x in date_range(start=get_date(-args.train_dt_duration, args.dt), end=args.dt)]
        args.dev_path = os.path.join(args.base_path, 'tfrecord', args.dt)

        args.test_path = os.path.join(args.base_path, 'tfrecord', args.dt)

    print('train_path', args.train_path)
    print('dev_path', args.dev_path)
    print('test_path', args.test_path)

    builder = TFrecordBuilder(args.conf_js)

    train_dataset = builder.generate(args.train_path, batch_size=args.batch_size)
    dev_dataset = builder.generate(args.dev_path, batch_size=args.batch_size)
    print('---------------------------generate dataset done---------------------------')

    if args.if_train:
        print('---------------------------train begin---------------------------')
        try:
            args.decay_epoch
        except:
            args.decay_epoch = 8
        try:
            args.decay_factor
        except:
            args.decay_factor = 0.8

        def scheduler(epoch, lr):
            if epoch <= args.decay_epoch:
                return lr
            else:
                return lr * args.decay_factor

        if len(args.choose_label) == 1:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3, mode='max', restore_best_weights=True),
                tf.keras.callbacks.LearningRateScheduler(scheduler)
            ]
        else:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_{}_auc_{}'.format(args.final_label_name, len(args.choose_label) - 1),
                    patience=3,
                    mode='max', restore_best_weights=True),
                tf.keras.callbacks.LearningRateScheduler(scheduler)
            ]

        start = time.time()

        history = model.fit(train_dataset, epochs=args.epoch, validation_data=dev_dataset, callbacks=callbacks,
                            verbose=args.verbose)

        train_best = max(history.history['{}_auc_{}'.format(args.final_label_name, len(args.choose_label) - 1)])
        val_best = max(history.history['val_{}_auc_{}'.format(args.final_label_name, len(args.choose_label) - 1)])

        print('train_best', train_best)
        print('val_best', val_best)

        end = time.time()
        print('train cost time {}'.format(round(end - start, 5)))
        print('---------------------------train done---------------------------')

    if args.if_save:
        print('---------------------------save model---------------------------')
        # 适配多目标需要保存原始模型需求
        if len(args.choose_label) > 1:
            try:
                args.save_model_origin_path
            except:
                args.save_model_origin_path = os.path.join(args.base_path, 'saved_model_origin', args.dt)
            # 本地不需要判断是否存在会覆盖，但idp上由于挂载盘的原因不删容易报错
            if os.path.exists(args.save_model_origin_path):
                shutil.rmtree(args.save_model_origin_path)  # 递归删除文件夹，即：删除非空文件夹
            model.save(args.save_model_origin_path)

        if os.path.exists(args.save_model_path):
            shutil.rmtree(args.save_model_path)  # 递归删除文件夹，即：删除非空文件夹
        # 若为多目标，抽出想要的那个label层保存给到工程组
        if len(args.choose_label) > 1:
            assert args.final_label_name, '{} final_label_name'.format(args.final_label_name)
            if args.final_label_name in args.choose_label:
                model_single_label = tf.keras.models.Model(inputs=model.input,
                                                           outputs=model.get_layer(args.final_label_name).output)
                model_single_label.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, name='adam'),
                    loss=['binary_crossentropy'],
                    metrics=['AUC'])
                model_single_label.save(args.save_model_path)
            else:
                print('final_label_name is {}'.format(args.final_label_name))
        else:
            model.save(args.save_model_path)

        if isinstance(args.conf_js, str):
            conf_js = read_json(args.conf_js)
        else:
            conf_js = args.conf_js
        save_json('{}/feature.json'.format(args.save_model_path), conf_js)
        save_json('{}/token_map.json'.format(args.save_model_path), {})
        print('---------------------------warm begin---------------------------')
        # warm up
        for data in dev_dataset:
            x, y = data
            tfserving_warmup({f.name: tf.expand_dims(x[f.name], 1) for f in fc.features}, args.save_model_path)
            break
        print('---------------------------warm done---------------------------')

        print('---------------------------save done---------------------------')

    try:
        args.model_metric
    except:
        args.model_metric = {}

    if args.if_test:
        print('---------------------------test begin---------------------------')
        if args.if_test:
            args.model_metric = Tester(args, model)
        print('---------------------------test done---------------------------')

    print('---------------------------update mvp begin---------------------------')
    push_mvp(args)
    print('---------------------------update mvp done---------------------------')

    print('---------------------------delete datasets---------------------------')
    # 删除24天前的数据集（不删问题也不大）
    delete_path_list = [os.path.join(args.dataset_path, get_date(-24, args.dt)),
                        os.path.join(args.dataset_path, get_date(-24, args.dt)),
                        os.path.join(args.save_model_path, get_date(-24, args.dt))]
    for path in delete_path_list:
        try:
            os.system(f'hdfs dfs -rm -r {path}')
        except:
            pass
    print('---------------------------delete datasets done---------------------------')

    return args.model_metric
