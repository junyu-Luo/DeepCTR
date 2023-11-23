from feature_util import DenseFeature, SparseFeature, VarLenFeature

# easy_feature算子支持规划
# map2array(map, keys), map2value(map, key)
# array_concat(array, " ")
# NOW时间戳 - 给定时间戳
# 当前时间(hour, weekday)

MUSIC_FEATURE_COLUMNS = [
    # # 0. 上下文环境
    SparseFeature('e_now_hour', dtype='int32', vocab=tuple(range(0,24)), missing=-1, emb_dim=8), #TODO:field;op

    # # 1. 用户侧
    # ## 1.1 属性类
    SparseFeature('u_gender', dtype='int32', vocab=(0,1), missing=-1, emb_dim=8),
    DenseFeature('u_age', dtype='float32', missing=0, op='normal')
    SparseFeature('u_age_group', dtype='int32', vocab=(0,1,2,3,4,5,6), missing=-1, emb_dim=8),
    SparseFeature('u_platform', dtype='int32', vocab=(0,1,5), missing=-1, emb_dim=8),
    SparseFeature('u_province', dtype='string', vocab=None, missing='', emb_dim=8), # TODO:vocab
    SparseFeature('u_city', dtype='string', vocab=None, missing='', emb_dim=8), # TODO:vocab
    SparseFeature('u_province_code', dtype='int32', vocab=None, missing='', emb_dim=8), # TODO:vocab
    SparseFeature('u_city_code', dtype='int32', vocab=None, missing='', emb_dim=8), # TODO:vocab
    #SparseFeature('u_cpid', dtype='string', vocab=None, missing='', emb_dim=8), # TODO:source-field
    SparseFeature('u_reg_time', dtype='int32', vocab=None, missing='', emb_dim=8), # TODO:vocab;op
    VarLenFeature('u_card_tag_ids', dtype='string', vocab=None, max_len=6, missing='', emb_dim=8), #TODO: vocab

    # # 2. 房间侧
    # ## 2.1 外显
    DenseFeature('i_user_num', dtype='int32', missing=1, op='normal'),
    VarLenFeature('i_title_keywords', dtype='string', vocab=None, max_len=5, missing='', emb_dim=8) # TODO: vocab;op

    # # 3. 房主侧
    SparseFeature('i_owner_gender', dtype='int32', vocab=(0,1), missing=-1, emb_dim=8),
    DenseFeature('i_owner_age', dtype='float32', missing=0, op='normal')
    SparseFeature('i_owner_age_group', dtype='int32', vocab=(0,1,2,3,4,5,6), missing=-1, emb_dim=8),
    SparseFeature('i_owner_platform', dtype='int32', vocab=(0,1,5), missing=-1, emb_dim=8),
    SparseFeature('i_owner_province', dtype='string', vocab=None, missing='', emb_dim=8), # TODO:vocab
    SparseFeature('i_owner_city', dtype='string', vocab=None, missing='', emb_dim=8), # TODO:vocab
    SparseFeature('i_owner_province_code', dtype='int32', vocab=None, missing='', emb_dim=8), # TODO:vocab
    SparseFeature('i_owner_city_code', dtype='int32', vocab=None, missing='', emb_dim=8), # TODO:vocab

]

ENGINE_SOURCE = {
    # 'e_now_hour': None,
    'u_gender': {"key_type": "user_id", "pid": 10001, "field": "sex"},
    'u_age': {"key_type": "user_id", "pid": 10001, "field": "age"},
    'u_age_group': {"key_type": "user_id", "pid": 10001, "field": "age_group"},
    'u_platform': {"key_type": "user_id", "pid": 10002, "field": "last_os_type"},
    'u_province': {"key_type": "user_id", "pid": 10017, "field": "province"},
    'u_city': {"key_type": "user_id", "pid": 10017, "field": "city"},
    'u_province': {"key_type": "user_id", "pid": 10017, "field": "province_code"},
    'u_city': {"key_type": "user_id", "pid": 10017, "field": "city_code"},
    'u_reg_time': {"key_type": "user_id", "pid": 10001, "field": "reg_time"},
    'u_card_tag_ids': {"key_type": "user_id", "pid": 10001, "field": "tag_4"}, # "246:238:237:13"
    '': {},
    '': {},
    'i_user_num': {"key_type": "item_id", "pid": 10101, "field": "user_num"},
    'i_title_keywords': {"key_type": "item_id", "pid": 110200600, "field": "keyword"},
    'i_owner_gender': {"key_type": "item_owner_id", "pid": 10001, "field": "sex"},
    'i_owner_age': {"key_type": "item_owner_id", "pid": 10001, "field": "age"},
    'i_owner_age_group': {"key_type": "item_owner_id", "pid": 10001, "field": "age_group"},
    'i_owner_platform': {"key_type": "item_owner_id", "pid": 10001, "field": "last_os_type"},
    'i_owner_province': {"key_type": "item_owner_id", "pid": 10017, "field": "province"},
    'i_owner_city': {"key_type": "item_owner_id", "pid": 10017, "field": "city"},
    'i_owner_province': {"key_type": "item_owner_id", "pid": 10017, "field": "province_code"},
    'i_owner_city': {"key_type": "item_owner_id", "pid": 10017, "field": "city_code"},
    '': {},
    '': {},
}

