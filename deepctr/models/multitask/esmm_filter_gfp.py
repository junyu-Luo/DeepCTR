import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Multiply

from ...layers.interaction import FM
from ...layers.core import PredictionLayer, DNN
from ...feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from ...layers.featconv import ArithmeticLayer, VocabLayer, AutoDis


# # Util function
def sequence_embs_combiner(fc, emb, mask_init, relavant_obj=None, method=None):
    if method is None:
        method = fc.combiner
    maxlen = fc.maxlen
    fn = fc.name

    if method in ('avg', 'mean', 'sum', 'max', 'test_mean'):
        mask = tf.cast(mask_init, tf.float32)
        if method == 'sum':
            mask = tf.expand_dims(mask, axis=1)
            sumed = tf.matmul(mask, emb)
            merged_emb = sumed
        elif method in ('avg', 'mean'):
            mask = tf.expand_dims(mask, axis=1)
            sumed = tf.matmul(mask, emb)
            avged = sumed / (tf.reduce_sum(mask, axis=-1, keepdims=True) + 1e-6)
            merged_emb = avged
        elif method == 'test_mean':
            #             mask = tf.expand_dims(mask, axis=1)
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keepdims=True)
            print('user_behavior_length:', user_behavior_length.shape)
            mask = tf.expand_dims(mask, axis=2)
            print('mask:', mask.shape)
            emb_size = emb.shape[-1]
            print('emb_size:', emb_size)
            mask = tf.tile(mask, [1, 1, emb_size])
            print('mask:', mask.shape)
            print('emb:', emb.shape)
            sumed = tf.reduce_sum(emb * mask, 1, keepdims=False)
            print('sumed:', sumed.shape)
            avged = tf.divide(sumed, tf.cast(user_behavior_length, tf.float32) + 1e-6)
            print('avged:', avged.shape)
            avged = tf.expand_dims(avged, axis=1)
            merged_emb = avged
        elif method == 'max':
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keepdims=True)
            mask = tf.expand_dims(mask, axis=2)
            emb_size = emb.shape[-1]
            mask = tf.tile(mask, [1, 1, emb_size])
            hist = emb - (1 - mask) * 1e9
            merged_emb = tf.reduce_max(hist, axis=1, keepdims=True)
            print('merged_emb:', merged_emb.shape)
        else:
            raise ValueError('method error')
    elif method == 'weighted_average':
        x = tf.strings.split(relavant_obj, ',', name=f'{fn}_weight_split').to_tensor('0', shape=[None, 1, maxlen],
                                                                                     name=f'{fn}_to_tensor')
        x = tf.compat.v1.string_to_number(tf.squeeze(x, axis=1), out_type=tf.float32, name=f'{fn}_to_number')
        mask = tf.expand_dims(x, axis=1)
        sumed = tf.matmul(mask, emb)
        merged_emb = sumed / (tf.reduce_sum(mask, axis=-1, keepdims=True) + 1e-6)
    elif method == 'attention':
        Q = layers.Dense(emb.shape[-1])(emb)
        K = layers.Dense(emb.shape[-1])(emb)
        V = layers.Dense(emb.shape[-1])(emb)
        embs = tf.keras.layers.Attention(use_scale=False)([Q, V, K], mask=[None, mask_init])
        _merged_emb = embs[:, :1, :]
        merged_emb = tf.keras.layers.Dense(4 * _merged_emb.shape[-1], activation='relu')(_merged_emb)
        merged_emb = tf.keras.layers.Dense(_merged_emb.shape[-1])(merged_emb)
    elif method == 'attention_no_mask':
        Q = layers.Dense(emb.shape[-1])(emb)
        K = layers.Dense(emb.shape[-1])(emb)
        V = layers.Dense(emb.shape[-1])(emb)
        embs = tf.keras.layers.Attention(use_scale=False)([Q, V, K])
        _merged_emb = embs[:, :1, :]
        merged_emb = tf.keras.layers.Dense(4 * _merged_emb.shape[-1], activation='relu')(_merged_emb)
        merged_emb = tf.keras.layers.Dense(_merged_emb.shape[-1])(merged_emb)
    elif method == 'attention_dot':
        attention_score = tf.reduce_sum(tf.multiply(emb, relavant_obj), axis=2) / (emb.get_shape()[-1] ** 0.5)
        score = tf.where(mask_init, attention_score, 1e-12)
        score = tf.nn.softmax(score)
        merged_emb = tf.matmul(tf.expand_dims(score, axis=1), emb)
    elif method == 'attention_din':
        # from tensorflow.keras import backend as K
        attention_hidden_units = (80, 40, 1)
        attention_activation = 'sigmoid'
        relavant_obj = tf.tile(relavant_obj, [1, maxlen, 1],
                               name=f'{fn}_din_attention_tile')  # (batch_size, maxlen * hidden_units)
        concat = K.concatenate([relavant_obj, emb, relavant_obj - emb, relavant_obj * emb],
                               axis=2)  # (batch_size, maxlen, emb_dim * 4) 点乘
        for i in range(len(attention_hidden_units)):
            activation = None if i == 2 else attention_activation
            outputs = layers.Dense(attention_hidden_units[i], activation=activation,
                                   name=f'{fn}_din_attention_dense_{i}')(concat)
            concat = outputs

        attention_score = tf.squeeze(outputs / (emb.get_shape()[-1] ** 0.5))
        score = tf.nn.softmax(tf.where(mask_init, attention_score, 1e-12), name=f'{fn}_din_attention_softmax')
        merged_emb = tf.matmul(tf.expand_dims(score, axis=1), emb)
    else:
        raise ValueError(
            'method must be in (mean, sum, max, min, attention, attention_no_mask, attention_dot, attention_din)')
    return merged_emb


def get_embeddings(feature_columns, inputs_dict, lookup_layer_dict, emb_layer_dict, return_dict=False):
    feature_dict = {fc.name: fc for fc in feature_columns}
    emb_dict = dict()
    for fn in emb_layer_dict.keys():
        if isinstance(feature_dict[fn], SparseFeat):
            if lookup_layer_dict[fn].table.key_dtype == tf.string and feature_dict[fn].dtype == 'int32':
                str_index = tf.strings.as_string(inputs_dict[fn], name=f'key_to_string_{fn}')
                sparse_index = lookup_layer_dict[fn](str_index)
            else:
                sparse_index = lookup_layer_dict[fn](inputs_dict[fn])
            emb = emb_layer_dict[fn](sparse_index)
            emb_dict[fn] = emb
        if isinstance(feature_dict[fn], VarLenSparseFeat):
            #             split_input = tf.keras.layers.Lambda(lambda x:tf.strings.split(x,','))(input_layer)[fn]

            x = tf.strings.split(inputs_dict[fn], ',', maxsplit=feature_dict[fn].maxlen, name=f'{fn}_split')
            x = tf.squeeze(x.to_tensor('', shape=[None, 1, feature_dict[fn].maxlen], name=f'{fn}_to_tensor'), axis=1)
            sparse_index = lookup_layer_dict[fn](x)
            emb = emb_layer_dict[fn](sparse_index)
            pooling_method = feature_dict[fn].combiner
            relavant_obj = None
            if pooling_method == 'weighted_average':
                if inputs_dict.get(fn + '_weight', None) is None:
                    raise ValueError(f'there is no weight column for feature {fn}')
                else:
                    relavant_obj = inputs_dict[fn + '_weight']

            mask_init = tf.not_equal(sparse_index, tf.constant(0, dtype=lookup_layer_dict[fn].table.value_dtype))
            if isinstance(pooling_method, list):
                combiner_emb = sequence_embs_combiner(feature_dict[fn], emb, mask_init, relavant_obj,
                                                      method=pooling_method[0])
                for method in pooling_method[1:]:
                    emb = sequence_embs_combiner(feature_dict[fn], emb, mask_init, relavant_obj, method=method)
                    combiner_emb = tf.concat([combiner_emb, emb], axis=-1)
            else:
                combiner_emb = sequence_embs_combiner(feature_dict[fn], emb, mask_init, relavant_obj,
                                                      method=pooling_method)
            emb_dict[fn] = combiner_emb
    return list(emb_dict.values()) if not return_dict else emb_dict


def ESMM(dnn_inputs, tower_dnn_hidden_units=(256, 128, 64), l2_reg_embedding=0.00001, l2_reg_dnn=0,
         seed=2022, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
         task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'),
         logits_list=[], inputs_dict={}):
    """Instantiates the Entire Space Multi-Task Model architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tower_dnn_hidden_units:  list,list of positive integer or empty list, the layer number and units in each layer of task DNN.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN.
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task_types:  str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss.
    :param task_names: list of str, indicating the predict target of each tasks. default value is ['ctr', 'ctcvr']
    :param logits_list: list of logits
    :param inputs_dict: dict of input features

    :return: A Keras model instance.
    """
    if len(task_names) != 2:
        raise ValueError("the length of task_names must be equal to 2")

    for task_type in task_types:
        if task_type != 'binary':
            raise ValueError("task must be binary in ESMM, {} is illegal".format(task_type))

    #     features = build_input_features(dnn_feature_columns)
    inputs_list = list(inputs_dict.values())

    #     sparse_embedding_list, dense_value_list = input_from_feature_columns(process_features,dnn_feature_columns,
    #                                                                          l2_reg_embedding, seed)

    #     dnn_inputs = combined_dnn_input(sparse_embedding_list, dense_value_list)

    ctr_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                     name='ctr_tower')(
        dnn_inputs)
    cvr_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                     name='cvr_tower')(
        dnn_inputs)

    ctr_logits = Dense(1, use_bias=True)(ctr_output)
    ctr_logits_list = logits_list + [ctr_logits]
    ctr_final_logits = layers.Add(name='ctr_final_logits')(ctr_logits_list) if len(ctr_logits_list) > 1 else \
        ctr_logits_list[0]

    cvr_logits = Dense(1, use_bias=True)(cvr_output)
    cvr_logits_list = logits_list + [cvr_logits]
    cvr_final_logits = layers.Add(name='cvr_final_logits')(cvr_logits_list) if len(cvr_logits_list) > 1 else \
        cvr_logits_list[0]

    ctr_pred = PredictionLayer(task_types[0], name=task_names[0])(ctr_final_logits)
    cvr_pred = PredictionLayer(task_types[1])(cvr_final_logits)

    ctcvr_pred = Multiply(name=task_names[1])([ctr_pred, cvr_pred])  # CTCVR = CTR * CVR

    model = Model(inputs=inputs_list, outputs=[ctr_pred, ctcvr_pred], name='esmm')
    return model


def build_esmm_filter_model(feature_columns, choose_fea, share_embedding_list, fm_fea_groups=None, model_name='DNN',
                dnn_hidden_units=(256,),
                task_names=['ctr', 'cvr'],
                dnn_activation='tanh', emb_dim=None,
                use_linear=True, use_fm=False, use_dnn=True,
                linear_reg=1e-5, emb_reg=1e-5, dnn_reg=1e-4,
                optimizer='adam', loss='binary_crossentropy', metrics=['AUC'],
                dropout_rate=0, use_bn=False):

    def first_use_fea(feas):
        min_index = len(choose_fea)
        min_fea_id = 0
        for i in range(len(feas)):
            if feas[i] in choose_fea:
                if choose_fea.index(feas[i]) < min_index:
                    min_index = choose_fea.index(feas[i])
                    min_fea_id = i
        feas[0], feas[min_fea_id] = feas[min_fea_id], feas[0]
        return feas

    share_embedding = {}
    for features in share_embedding_list:
        # put the feature used first in the first place
        feas = first_use_fea(features)
        if feas:
            for fea in feas[1:]:
                share_embedding[fea] = feas[0]  # 'i_u_p_gender': 'u_p_gender'

    assert (use_linear or use_fm or use_dnn)
    use_dense = any(True if isinstance(fc, DenseFeat) else False for fc in feature_columns)
    use_sparse = any(True if isinstance(fc, SparseFeat) else False for fc in feature_columns)
    logits_list = list()

    # # inputs
    inputs_dict = dict()
    dense_op_layer_dict = dict()
    lookup_layer_dict = dict()
    emb_layer_dict = dict()
    linear_emb_layer_dict = dict()
    for fc in feature_columns:
        if isinstance(fc, DenseFeat):
            inputs_dict[fc.name] = layers.Input(name=fc.name, shape=(1,), dtype=fc.dtype)
            if fc.operation is not None:
                dense_op_layer_dict[fc.name] = ArithmeticLayer(fc.operation, fc.statistics,
                                                               name=f'{fc.operation}_{fc.name}')
        elif isinstance(fc, SparseFeat):
            inputs_dict[fc.name] = layers.Input(name=fc.name, shape=(1,), dtype=fc.dtype)
            if not share_embedding.get(fc.name, None) or not lookup_layer_dict.get(share_embedding[fc.name], None):
                lookup_layer_dict[fc.name] = VocabLayer(fc.vocab, name=f'lookup_{fc.name}')
                emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, fc.embedding_dim, name=f'emb_{fc.name}')
                linear_emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, 1, name=f'linear_emb_{fc.name}',
                                                                  embeddings_regularizer=l2(linear_reg))
            else:
                lookup_layer_dict[fc.name] = lookup_layer_dict[share_embedding[fc.name]]
                emb_layer_dict[fc.name] = emb_layer_dict[share_embedding[fc.name]]
                linear_emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, 1, name=f'linear_emb_{fc.name}',
                                                                  embeddings_regularizer=l2(linear_reg))
        elif isinstance(fc, VarLenSparseFeat):
            inputs_dict[fc.name] = layers.Input(name=fc.name, shape=(1,), dtype=fc.dtype)
            lookup_layer_dict[fc.name] = VocabLayer(fc.vocab, name=f'lookup_{fc.name}')
            emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, fc.embedding_dim, name=f'emb_{fc.name}')

        #     inputs_dict[fc.name] = layers.Input(name=fc.name, shape=(1,), dtype=fc.dtype)
        #     if not fc.is_weight:
        #         if not share_embedding.get(fc.name, None) or not lookup_layer_dict.get(share_embedding[fc.name], None):
        #             lookup_layer_dict[fc.name] = VocabLayer(fc.vocab, name=f'lookup_{fc.name}')
        #             emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, fc.embedding_dim, name=f'emb_{fc.name}')
        #         #                     linear_emb_layer_dict[fc.name] = layers.Embedding(1+fc.vocabulary_size, 1, name=f'linear_emb_{fc.name}', embeddings_regularizer=l2(linear_reg))
        #         else:
        #             lookup_layer_dict[fc.name] = lookup_layer_dict[share_embedding[fc.name]]
        #             emb_layer_dict[fc.name] = emb_layer_dict[share_embedding[fc.name]]
        # #                     linear_emb_layer_dict[fc.name] = layers.Embedding(1+fc.vocabulary_size, 1, name=f'linear_emb_{fc.name}', embeddings_regularizer=l2(linear_reg))
        else:
            raise ValueError('feature_columns contains unknown Featrue-Type')

    if use_dense:
        normal_dense_list = [
            AutoDis(num_buckets=16, emb_dim=8, keepdim=True)(dense_op_layer_dict[fc.name](inputs_dict[fc.name]))
            if fc.operation is not None else AutoDis(num_buckets=16, emb_dim=8, keepdim=True)(inputs_dict[fc.name])
            for fc in feature_columns if isinstance(fc, DenseFeat)]
        concat_denses = layers.Concatenate(name='concat_denses')(normal_dense_list)
    if use_sparse:
        emb_list = get_embeddings(feature_columns, inputs_dict, lookup_layer_dict, emb_layer_dict)

    # Linear
    if use_linear:
        if use_dense:
            linear_dense_logits = layers.Dense(1, kernel_regularizer=l2(linear_reg), name='linear_dense_logits')(
                concat_denses)
        if use_sparse:
            weight_sparses = get_embeddings(feature_columns, inputs_dict, lookup_layer_dict, linear_emb_layer_dict)
            linear_sparse_logits = layers.Flatten(name='linear_sparse_logits')(
                layers.Add(name='linear_sparse_logits_')(weight_sparses))
        if use_dense and use_sparse:
            linear_logits = layers.Add(name='linear_logits')([linear_dense_logits, linear_sparse_logits])
        elif use_dense:
            linear_logits = linear_dense_logits
        elif use_sparse:
            linear_logits = linear_sparse_logits
        else:
            raise Exception('no linear_logits')
        logits_list.append(linear_logits)

    # fm
    if use_fm and use_sparse:
        emb_dict = get_embeddings(feature_columns, inputs_dict, lookup_layer_dict, emb_layer_dict, return_dict=True)
        for i, fea_group in enumerate(fm_fea_groups):
            fm_emb_list = [emb_dict[fea] for fea in fea_group]
            embs_concat = layers.Concatenate(axis=1, name=f'embs_concat_{i}')(fm_emb_list)
            fm_logits = FM(name=f'fm_{i}')(embs_concat)
            logits_list.append(fm_logits)

    # dnn
    if use_sparse and use_dense:
        concat_sparse = layers.Flatten(name='embs_flatten')(
            layers.Concatenate(axis=-1, name='embs_concat_last')(emb_list))
        dnn_inputs = layers.Concatenate(name='dnn_inputs')([concat_denses, concat_sparse])
    elif use_sparse:
        dnn_inputs = layers.Flatten(name='embs_flatten')(layers.Concatenate(axis=-1, name='embs_concat_last')(emb_list))
    elif use_dense:
        dnn_inputs = concat_denses
    else:
        raise Exception('no dnn_inputs')

    if model_name == 'DNN':
        if use_dnn:
            _hidden = DNN(dnn_hidden_units, activation=dnn_activation, l2_reg=dnn_reg, dropout_rate=dropout_rate,
                          use_bn=use_bn, name='dnn_hidden')(dnn_inputs)
            dnn_logits = layers.Dense(1, name='dnn_logits')(_hidden)
            logits_list.append(dnn_logits)

        # final sigmoid output
        final_logits = layers.Add(name='final_logits')(logits_list) if len(logits_list) > 1 else logits_list[0]
        sigmoid_output = layers.Activation('sigmoid', name='sigmoid_output')(final_logits)

        # build and compile model
        model = models.Model(inputs=inputs_dict, outputs=[sigmoid_output], name=model_name)

    elif model_name == 'ESMM':
        model = ESMM(dnn_inputs, tower_dnn_hidden_units=dnn_hidden_units, l2_reg_embedding=emb_reg, l2_reg_dnn=dnn_reg,
                     seed=2022, dnn_dropout=dropout_rate, dnn_activation=dnn_activation, dnn_use_bn=use_bn,
                     task_types=('binary', 'binary'), task_names=task_names,
                     logits_list=logits_list, inputs_dict=inputs_dict)
    else:
        print(f'Error: model_name was not defined: {model_name}, it should be one of (DNN, ESMM, PLE)')
        model = None

    return model

    # model.compile(optimizer=optimizer,
    #               loss={label: loss for label in labels},
    #               metrics={label: metrics for label in labels})
