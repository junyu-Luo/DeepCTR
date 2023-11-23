# %% import
import os, sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, FloatType, IntegerType, StringType, ArrayType

from rcmd_utils.common_util import get_date, date_range
from rcmd_utils.experiment.zhoutao.common_util import get_logger
from rcmd_utils.experiment.zhoutao.spark_util import get_hdfs_or_obs_file, delete_dfs_path, copy_local_to_dfs
from rcmd_utils.experiment.zhoutao.feature_util_v2 import DenseFeature, SparseFeature, VarLenFeature, PersonaFeature

# %% init
logger = get_logger('logger');
print = logger.info
DEV_MODE = 'ipy' in sys.argv[0]  # jupyter或ipython下开发
if not DEV_MODE:
    print('DEBUG: IDP mode')
    spark = SparkSession.builder.appName('better_interact_music_rooms').enableHiveSupport().getOrCreate()
else:
    print('DEBUG: DEV mode')
    import sys

    sys.path.append('./zhoutao/utils/')
    from pyspark.sql import functions as F


    def spark_jupyter_8887(app_name='ljy_tes', executor_instances=4, executor_core=1, executor_memory='2g', driver_memory='2g',
                           queue="spark", addconf={}, adddist=[]):
        '''
        用于 8887 服务器
        '''
        distfiles = 'file:/usr/hdp/3.1.4.0-315/spark2/python/lib/pyspark.zip,file:/usr/hdp/3.1.4.0-315/spark2/python/lib/py4j-0.10.7-src.zip'
        if len(adddist) > 0:
            add = ','.join([str(x) for x in adddist])
            distfiles = add + ',' + distfiles
            print('spark.yarn.dist.files:' + distfiles)

        from pyspark.sql import SparkSession
        from pyspark import SparkConf
        conf = SparkConf().setAppName(app_name).setMaster("yarn") \
            .set('spark.yarn.queue', queue) \
            .set('spark.yarn.dist.archives', 'hdfs:///user/root/jupyter-env.tar.gz') \
            .set('spark.yarn.dist.files', distfiles) \
            .setExecutorEnv('PYTHONPATH', 'pyspark.zip:py4j-0.10.7-src.zip')

        conf.set("spark.jars",
                 "hdfs:///user/root/tfrecord_lib/spark-connector_2.11-1.10.0.jar,hdfs:///user/root/xgboost_lib/xgboost4j-spark-0.90.jar,hdfs:///user/root/xgboost_lib/xgboost4j-0.90.jar,hdfs:///user/root/hadoop-lzo-0.6.0.3.1.4.0-315.jar,hdfs:///user/root/hbase-spark-1.0.0.jar")
        conf.set("spark.executor.memory", executor_memory)
        conf.set("spark.driver.memory", driver_memory)
        conf.set("spark.executor.cores", executor_core)
        conf.set("spark.executor.instances", executor_instances)
        conf.set("spark.driver.maxResultSize", "0")
        #     conf.set("hive.exec.orc.split.strategy", "ETL")
        #     conf.set("spark.sql.files.ignoreCorruptFiles", "true")
        #     conf.set("spark.dynamicAllocation.enabled","true")
        conf.set("spark.shuffle.service.enabled", "true")
        #     conf.set("spark.sql.broadcastTimeout", 10000)

        for kk, vv in addconf.items(): conf.set(kk, vv)

        spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
        return spark

    # jar = 'hdfs:///user/root/tfrecord_lib/spark-tfrecord_2.12-0.4.0.jar'  # for spark3
    jar = 'hdfs:///user/root/tfrecord_lib/spark-tfrecord_2.11-0.2.4.jar'  # for spark2
    spark = spark_jupyter_8887(addconf={'spark.jars': jar})


# %% functions
def force_cast(df, name2datatype):
    ''' 类型转换 '''
    for name, dtype in name2datatype.items():
        datatype = dtype.lower()
        if name == 'i_expo_titleTags':  # 特殊
            continue

        if datatype in ('string', 'str'):
            df = df.withColumn(name, F.col(name).cast('string'))
        elif datatype in ('int32', 'int'):
            df = df.withColumn(name, F.col(name).cast('int'))
        elif datatype in ('float32', 'float'):
            df = df.withColumn(name, F.col(name).cast('float'))
        elif datatype.endswith('vec'):
            if datatype.startswith('string'):
                df = df.withColumn(name, F.col(name).cast('array<string>'))
            elif datatype.startswith('int'):
                df = df.withColumn(name, F.col(name).cast('array<int>'))
            elif datatype.startswith('float'):
                df = df.withColumn(name, F.col(name).cast('array<float>'))
            else:
                raise ValueError(f'Unknown datatype({datatype})')
        elif datatype.startswith('map'):
            ktype, vtype = datatype[3:], datatype[3:].rstrip('0123456789')
            key_type = FloatType if ktype.startswith('float') else IntegerType if ktype.startswith(
                'int') else StringType if ktype.startswith('string') else None
            value_type = FloatType if vtype.endswith('float') else IntegerType if vtype.endswith(
                'int') else StringType if vtype.endswith('string') else None
            assert key_type is not None and value_type is not None, f'Unknown datatype({datatype})'

            @F.udf(MapType(key_type(), value_type()))
            def donothing(x):
                return x

            df = df.withColumn(name, donothing(F.col(name)))
        else:
            raise ValueError(f'Unknown datatype({datatype})')
    return df


def fill_na(df, name2default):
    ''' 填充缺失值 '''
    default_dict = {}
    for name, default in name2default.items():
        if name == 'i_expo_titleTags':  # 特殊
            default_dict[name] = ''
            continue

        if isinstance(default, (list, tuple)):
            df = df.withColumn(name, F.coalesce(F.col(name), F.array([F.lit(x) for x in default])))
        elif isinstance(default, dict):
            pass  # 暂不填充缺失值。后续对map处理的算子需要考虑这点
        else:
            default_dict[name] = default
    df = df.fillna(default_dict)
    return df


def persona_to_feature(df, persona_features):
    ''' 画像处理成特征 '''
    return df


def fill_vocab_or_op(df, persona_features, verbose=True):
    # 统计vocab和op参数
    for pf in persona_features:
        fc = pf.tf_input
        fn = fc.name
        if isinstance(fc, VarLenFeature):
            if not fc.vocab:
                sep, max_len = fc.sep, fc.max_len
                vocab_count = df.select(fn).rdd.flatMap(
                    lambda x: [(value, 1) for i, value in enumerate(x[fn].split(sep, max_len)) if i < max_len]
                ).reduceByKey(lambda x, y: x + y).collectAsMap()
                if verbose:
                    print(f'[vocab_or_op] {fn}: {str(vocab_count)[:1000]}')

                if fc.min_cnt <= 1:
                    fc.vocab = [value for value, count in vocab_count.items() if
                                count >= fc.min_cnt and value != pf.default]  # 不含默认值
                else:
                    fc.vocab = [value for value, count in vocab_count.items() if count >= fc.min_cnt]  # 可以包含默认值
                fc.vocab_size = len(fc.vocab)
                fc.kwargs['vocab_count'] = vocab_count
        elif isinstance(fc, SparseFeature):
            if not fc.vocab:
                # if fn in ('u_cpid', 'i_owner_cpid'):
                if 'before_vocab' in fc.kwargs:
                    # 特殊情况
                    @F.udf(StringType())
                    def split_dot_get_last(col):
                        return col.split('.')[-1]

                    if fc.kwargs['before_vocab'] == 'split_dot_get_last':
                        vocab_count = df.withColumn('tmp_' + fn, split_dot_get_last(fn)).groupby(
                            'tmp_' + fn).count().toPandas().sort_values('count', ascending=False).set_index(
                            'tmp_' + fn)['count'].to_dict()
                    else:
                        raise Exception('before_vocab: %s is not exists' % fc.kwargs['before_vocab'])
                else:
                    vocab_count = df.groupby(fn).count().toPandas().sort_values('count', ascending=False).set_index(fn)[
                        'count'].to_dict()
                if verbose:
                    print(f'[vocab_or_op] {fn}: {str(vocab_count)[:1000]}')

                if fc.min_cnt <= 1:
                    fc.vocab = [value for value, count in vocab_count.items() if
                                count >= fc.min_cnt and value != pf.default]  # 不含默认值
                else:
                    if fc.kwargs.get('oov_equals_default', True):
                        fc.vocab = [value for value, count in vocab_count.items() if
                                    count >= fc.min_cnt and value != pf.default]  # 不可以包含默认值
                    else:
                        fc.vocab = [value for value, count in vocab_count.items() if count >= fc.min_cnt]  # 可以包含默认值
                fc.vocab_size = len(fc.vocab)
                fc.kwargs['vocab_count'] = vocab_count
        elif isinstance(fc, DenseFeature):
            if fc.op in ('normal', 'log_normal', 'min_max') and fc.op_num_x is None:
                if True:
                    stats = df.agg(
                        F.min(F.when(F.col(pf.name) == pf.default, F.lit(None)).otherwise(F.col(pf.name))).alias('min'),
                        F.max(F.when(F.col(pf.name) == pf.default, F.lit(None)).otherwise(F.col(pf.name))).alias('max'),
                        F.mean(F.when(F.col(pf.name) == pf.default, F.lit(None)).otherwise(F.col(pf.name))).alias(
                            'mean'),
                        F.mean(
                            F.log(F.when(F.col(pf.name) == pf.default, F.lit(None)).otherwise(F.col(pf.name)))).alias(
                            'log_mean'),
                        F.stddev(F.when(F.col(pf.name) == pf.default, F.lit(None)).otherwise(F.col(pf.name))).alias(
                            'stddev'),
                        F.stddev(
                            F.log(F.when(F.col(pf.name) == pf.default, F.lit(None)).otherwise(F.col(pf.name)))).alias(
                            'log_stddev'),
                    )
                else:
                    stats_ori = df.agg(
                        F.min(F.col(pf.name)).alias('min'),
                        F.max(F.col(pf.name)).alias('max'),
                        F.mean(F.col(pf.name)).alias('mean'),
                        F.mean(F.log(F.col(pf.name))).alias('log_mean'),
                        F.stddev(F.col(pf.name)).alias('stddev'),
                        F.stddev(F.log(F.col(pf.name))).alias('log_stddev'),
                    )

                stats_dict = stats.toPandas().T[0].apply(lambda x: round(x, 6)).to_dict()
                if verbose:
                    print(f'[vocab_or_op] {fn}: {stats_dict}')

                if fc.op == 'normal':
                    fc.op_num_x = stats_dict['mean']
                    fc.op_num_y = stats_dict['stddev']
                elif fc.op == 'min_max':
                    fc.op_num_x = stats_dict['min']
                    fc.op_num_y = stats_dict['max']
                elif fc.op == 'log_normal':
                    fc.op_num_x = stats_dict['log_mean']
                    fc.op_num_y = stats_dict['log_stddev']
                else:
                    raise ValueError(f'{fc.name}:{fc.op} 没有处理op_num_x和op_num_y')

                fc.kwargs['op_stats'] = stats_dict
            elif fc.op in ('ratio',):
                fc.op_num_x = 1.0 if fc.shape[0] > 5 else 2.0
            else:
                pass
        else:
            raise ValueError('Unknown xxxFeature')


def save_features_info(persona_features, output_path, dir_name='features_info', version=1, sc=None, spark=None):
    import os, json, pickle
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temp_dir:
        dir_name = os.path.join(temp_dir, dir_name);
        os.mkdir(dir_name)

        with open(f'{dir_name}/persona_features.pkl', 'wb') as fw:
            pickle.dump(persona_features, fw)
        with open(f'{dir_name}/feature.json', 'w', encoding='utf-8') as fw:
            feature_json = PersonaFeature.gen_feature_json(persona_features, version=version)
            json.dump(feature_json, fw, indent=2, ensure_ascii=False)
        with open(f'{dir_name}/token_map.json', 'w', encoding='utf-8') as fw:
            fw.write('{}')

        if os.system(f'hdfs dfs -put -f {dir_name} {output_path}') != 0:
            if sc is None:
                sc = spark.sparkContext
            delete_dfs_path(sc, os.path.join(output_path, dir_name.strip('/').split('/')[-1]))
            if not copy_local_to_dfs(sc, dir_name, output_path):
                raise Exception('put到hdfs失败：%s' % dir_name)


# %% persona features
persona_features = [
    # 环境上下文
    PersonaFeature('e_hour', 0, 'int32', None, is_feature=False, expr='get_now_hour_batch()',
                   tf_input=SparseFeature('e_hour', dtype='int32', vocab=tuple(range(24)), emb_dim=8)),
    PersonaFeature('e_enter_source', 0, 'int32', None, is_feature=False, expr='batch(quickmatch)',
                   expr_params={'quickmatch': 60},
                   tf_input=SparseFeature('e_enter_source', dtype='int32', vocab=(60, 61, 73, 86), emb_dim=8)),

    # 用户/房主 基础静态属性
    PersonaFeature('u_gender', 0, 'int32', ('user_id', 10001, 'sex'), True,
                   tf_input=SparseFeature('u_gender', dtype='int32', vocab=(0, 1), emb_dim=8)),
    PersonaFeature('i_owner_gender', 0, 'int32', ('item_owner_id', 10001, 'sex'), True,
                   tf_input=SparseFeature('i_owner_gender', dtype='int32', vocab=(0, 1), emb_dim=8)),

    PersonaFeature('u_age', 0, 'float32', ('user_id', 10001, 'age'), True,
                   tf_input=DenseFeature('u_age', dtype='float32', op='normal', emb_dim=8)),
    PersonaFeature('i_owner_age', 0, 'float32', ('item_owner_id', 10001, 'age'), True,
                   tf_input=DenseFeature('i_owner_age', dtype='float32', op='normal', emb_dim=8)),

    PersonaFeature('u_phone', '', 'string', ('user_id', 10017, 'phone'), True,
                   tf_input=SparseFeature('u_phone', dtype='string', vocab=('',), emb_dim=8)),
    PersonaFeature('i_owner_reg_days', 0, 'int32', ('item_owner_id', 10001, 'reg_time'), False,
                   expr='if(equal(i_owner_reg_days, batch(ZERO_INT)), batch(ZERO_INT), get_time_cnt(i_owner_reg_days, DAY))',
                   expr_params={'DAY': 'd', 'ZERO_INT': 0},
                   tf_input=DenseFeature('i_owner_reg_days', dtype='int32', op='log_normal', emb_dim=8)),

    PersonaFeature('u_cpid', '', 'string', ('user_id', 10013, 'cp_id'), True,
                   tf_input=SparseFeature('u_cpid', dtype='string', vocab=(), min_cnt=30, emb_dim=8,
                                          before_vocab='split_dot_get_last', oov_equals_default=True)),
    # 需要预处理 split.取最后一段（备选：pkg_channel）
    PersonaFeature('i_owner_cpid', '', 'string', ('item_owner_id', 10013, 'cp_id'), True,
                   tf_input=SparseFeature('i_owner_cpid', dtype='string', vocab=(), min_cnt=30, emb_dim=8,
                                          before_vocab='split_dot_get_last', oov_equals_default=True)),
    # 需要预处理 split.取最后一段（备选：pkg_channel）
    # u_media_name, i_owner_media_name

    PersonaFeature('u_city', '', 'string', ('user_id', 110100200, 'city'), True,
                   tf_input=SparseFeature('u_city', dtype='string', vocab=(), min_cnt=30, emb_dim=8)),
    PersonaFeature('u_province', '', 'string', ('user_id', 110100200, 'province'), True,
                   tf_input=SparseFeature('u_province', dtype='string', vocab=(), min_cnt=100, emb_dim=8)),

    # 房间 动态特征(房间人数/麦上人数/是否唱歌/合唱/房主段位/房主称号)
    PersonaFeature('i_user_num', 0, 'float32', ('item_id', 10101, 'user_num'), True,
                   tf_input=DenseFeature('i_user_num', dtype='float32', op='min_max', op_num_x=0, op_num_y=15,
                                         emb_dim=8)
                   ),
    PersonaFeature('i_mic_num', 0, 'float32', ('item_id', 10101, 'mic_num'), True,
                   tf_input=DenseFeature('i_mic_num', dtype='float32', op='min_max', op_num_x=0, op_num_y=10, emb_dim=8)
                   ),
    PersonaFeature('i_mic_female_num', 0, 'float32', ('item_id', 10101, 'mic_female_num'), True,
                   tf_input=DenseFeature('i_mic_female_num', dtype='float32', op='min_max', op_num_x=0, op_num_y=10,
                                         emb_dim=8)
                   ),
    PersonaFeature('i_ktv_song_name', '', 'string', ('item_id', 110200900, 'song_title'), True,
                   tf_input=SparseFeature('i_ktv_song_name', dtype='string', vocab=('',), emb_dim=8)
                   ),
    PersonaFeature('i_ktv_sing_user_num', 0, 'float32', ('item_id', 110200900, 'singer_count'), True,
                   tf_input=DenseFeature('i_ktv_sing_user_num', dtype='float32', op='min_max', op_num_x=0, op_num_y=10,
                                         emb_dim=8)
                   ),
    PersonaFeature('i_owner_star_level', 0, 'float32', ('item_id', 110111700, 'star_level'), True,
                   tf_input=DenseFeature('i_owner_star_level', dtype='float32', op='log_normal', emb_dim=8)
                   ),
    PersonaFeature('i_owner_glory_name', '', 'string', ('item_id', 110200900, 'glory_name'), True,
                   tf_input=SparseFeature('i_owner_glory_name', dtype='string', vocab=('',), emb_dim=8)
                   ),
    # PersonaFeature('i_ktv_song_id', '', 'string', ('item_id', 110200900, 'song_title'), True,
    #    tf_input=SparseFeature('i_ktv_song_id', dtype='string', vocab=('placeholder', ), emb_dim=32)
    # ),
]

# %% main
if __name__ == '__main__':
    dt, n_days = (sys.argv[1], int(sys.argv[2])) if not DEV_MODE else ('20221127', 7)
    start_dt, split_dt, end_dt = get_date(-n_days, dt), get_date(-1, dt), get_date(0, dt)
    print(f'dt={dt}: start_dt={start_dt}, split_dt={split_dt}, end_dt={end_dt}')
    GZIP = "org.apache.hadoop.io.compress.GzipCodec"

    input_path = 'obs://pvc-obs-hw-bj-zt-rcmd-data/user/ljy/dataset/music/quickmatch/temp_v1'
    output_path = f'obs://pvc-obs-hw-bj-zt-rcmd-data/user/zhoutao/dataset/music/quickmatch/temp_v3_tfrecord/dt={end_dt}'
    train_dts = '{%s}' % ','.join(date_range(start_dt, split_dt, end_include=True))

    # train data
    train_df = spark.read.load(input_path).where(
        f"dt>='{start_dt}' and dt<='{split_dt}' and tag_name='一起K歌' and e_enter_source in (60, 61, 73, 86)")
    train_df = train_df.join(
        spark.sql(f'''select dt, user_id, u_p_user_city as u_city, u_p_user_province as u_province
            from ttrecdw.dwd_tt_algo_profile_user_scenario_df where dt>="{start_dt}" and dt<="{split_dt}"'''),
        how='left', on=['dt', 'user_id']
    )
    train_df = force_cast(train_df, {pf.name: pf.datatype for pf in persona_features})  # 强制类型转换
    train_df = fill_na(train_df, {pf.name: pf.default for pf in persona_features})  # 填充缺失值
    train_df = persona_to_feature(train_df, persona_features)  # 画像处理成特征(非通用，需要自定义)

    train_df = train_df.select('user_id', 'duration', *[pf.tf_input.name for pf in persona_features]).repartition(
        140).persist()
    train_df.write.format("tfrecord").option('codec', GZIP).save(os.path.join(output_path, 'train'), mode='overwrite')

    # feature.json
    fill_vocab_or_op(train_df, persona_features)
    save_features_info(persona_features, output_path, dir_name='features_info', version=2, spark=spark)

    # validation data
    val_df = spark.read.load(input_path).where(
        f"dt='{end_dt}' and tag_name='一起K歌' and e_enter_source in (60, 61, 73, 86)")

    val_df = force_cast(val_df, {pf.name: pf.datatype for pf in persona_features})  # 强制类型转换
    val_df = fill_na(val_df, {pf.name: pf.default for pf in persona_features})  # 填充缺失值
    val_df = persona_to_feature(val_df, persona_features)  # 画像处理成特征(非通用，需要自定义)

    val_df = val_df.select('user_id', 'duration', *[pf.tf_input.name for pf in persona_features]).repartition(20)
    val_df.write.format("tfrecord").option('codec', GZIP).save(os.path.join(output_path, 'validation'),
                                                               mode='overwrite')

    # 清理旧数据
    clean_dt = get_date(-30, end_dt)
    os.system(f'hdfs dfs -rm -r {output_path.split("dt=")[0] + "dt=" + clean_dt}')
