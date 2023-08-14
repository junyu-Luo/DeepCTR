# -*- coding: utf-8 -*-
import os
import tensorflow as tf

try:
    from tensorflow_serving.apis import model_pb2
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_log_pb2
except Exception as e:
    print('WARN: save tfx_warmup failed, please install tensorflow_serving')


# warm up
def generate_tfserving_warmup(X: dict, savedmodel_path, n=100):
    extra_path = os.path.join(savedmodel_path, 'assets.extra')
    warmup_file = os.path.join(extra_path, 'tf_serving_warmup_requests')
    if not os.path.exists(extra_path):
        os.makedirs(extra_path)

    with tf.io.TFRecordWriter(warmup_file) as writer:
        request = predict_pb2.PredictRequest(
            inputs={k: tf.make_tensor_proto(v[:n]) for k, v in X.items()}
        )
        log = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())
    return extra_path


def tfserving_warmup(test_x, save_model_path, n=100):
    generate_tfserving_warmup(test_x, save_model_path, n=n)
    print('warmup done')
