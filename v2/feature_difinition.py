class MetaFeature:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        class_name = self.__class__.__name__.split('.')[-1]
        s = class_name + '('
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            elif k == 'vocab':
                s += f'{k}={v if len(v)<=4 else v[:2]+v[-2:]}, '
            elif len(str(v)) > 200:
                s += f'{k}={str(v)[:20]}... '
            else:
                s += f'{k}={v}, '
        s = s.strip(', ') + ')'
        return s

    def __repr__(self) -> str:
        return self.__str__()

class DenseFeature(MetaFeature):
    ''' 实数特征：dense
        - name     ：特征列名
        - shape    ：输入大小, 默认长度为1
        - dtype    ：输入数据类型，默认float32
        - op       ：op可以做特征处理，如min_max_scale, normal_scale, log_normal_scale
        - op_num_x ：传递给op的第一个参数
        - op_num_y ：传递给op的第二个参数
        - missing  ：暂无用，仅用于备注
    '''
    def __init__(self, name, shape=1, dtype='float32', op=None, op_num_x=None, op_num_y=None, emb_dim=None, missing=None, engine=None, **kwargs):
        self.name = name
        self.shape = shape  # (n, )
        if isinstance(shape, int):
            self.shape = (shape, )
        self.dtype = dtype  # string

        self._allow_op_ = ('normal', 'min_max', 'log', 'log_normal', 'ratio')
        assert op is None or op in self._allow_op_, f'{op} not in {self._allow_op_}'
        self.op = op
        self.op_num_x = op_num_x
        self.op_num_y = op_num_y
        self.emb_dim = emb_dim  # for AutoDis

        self.missing = missing
        if engine is None:
            self.engine = dict()
        else:
            assert isinstance(engine, dict), 'engine must be dict'
            self.engine = engine

        self.kwargs = kwargs

class SparseFeature(MetaFeature):
    ''' 类别特征：sparse / category
        - name     ：特征列名
        - vocab    ：List, 枚举所有类别
        - emb_dim  ：embedding dimension
        - dtype    ：默认string，可选int32
        - missing  ：暂无用，仅用于备注
    '''
    def __init__(self, name, vocab=(), emb_dim=8, share_emb=None, dtype='int32', missing=None, min_cnt=1, engine=None, **kwargs):
        assert dtype in ('string', 'int32'), 'dtype only accept (string, int32)'
        self.name = name
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.emb_dim = emb_dim  # 6 * int(pow(vocabulary_size, 0.25))
        self.share_emb = share_emb
        self.shape = (1, )
        self.dtype = dtype
        self.missing = missing
        self.min_cnt = min_cnt
        if engine is None:
            self.engine = dict()
        else:
            assert isinstance(engine, dict), 'engine must be dict'
            self.engine = engine

        self.kwargs = kwargs

class VarLenFeature(MetaFeature):
    ''' 变长类别特征
        - name     ：特征列名
        - vocab    ：List, 枚举所有类别
        - emb_dim  ：embedding dimension
        - dtype    ：默认string，可选int32
        - missing  ：暂无用，仅用于备注
    '''
    def __init__(self, name, vocab=(), max_len=9, emb_dim=8, share_emb=None, pooling='mean', dtype='int32', missing=None, sep=' ', min_cnt=1, engine=None, **kwargs):
        assert dtype in ('string', 'int32'), 'dtype only accept (string, int32)'
        self.name = name
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_len = max_len
        self.shape = (max_len, ) if dtype not in ('str', 'string') else (1,)
        self.emb_dim = emb_dim  # 6 * int(pow(vocabulary_size, 0.25))
        self.share_emb = share_emb
        self.pooling = pooling
        self.dtype = dtype
        self.missing = missing
        self.sep = sep
        self.min_cnt = min_cnt
        if engine is None:
            self.engine = dict()
        else:
            assert isinstance(engine, dict), 'engine must be dict'
            self.engine = engine

        self.kwargs = kwargs


class PersonaFeature:
    def __init__(self, name, default, datatype, source, is_feature=False,
                 expr=None, expr_params=None, expr_out=None,
                 tf_input=None, ):
        self.name = name         # hive画像名：线上服务时，pb画像取回来后重命名成这个名字
        self.default = default   # 缺失值
        self.datatype = datatype       # 从pb取回画像后，强转成的类型（double-->float, int-->float, map<string, int> --> map<string, float>）
        self.source = source     # 定义pb画像
        self.is_feature = is_feature  # True说明这个画像donothing即可变成特征

        self.expr = expr
        self.expr_params = expr_params
        self.expr_out = expr_out

        self.tf_input = tf_input

        # 检查输入参数之间是否合法
        self.check()

        if source is None:  # 无需画像的特征，如e_hour
            self.schema = None
        else:
            self.schema = {
                "type": self.format_datatype(self.datatype),
                "default": self.default,
                "source": self.format_source(self.source),
            }

        if is_feature: # 画像即特征，大部分非map的离线画像
            self.features = {'expr': self.name, 'out': self.name}
        else:
            self.features = {'expr': self.expr, 'out': self.expr_out or self.name}
            if self.expr_params is not None:
                self.features['consts'] = dict()
                for key, value in self.expr_params.items():
                    value_type, value = self.infer_datatype(value)
                    self.features['consts'][key] = {'value': value, 'type': value_type}

    def check(self):
        if self.source is not None:
            # 检查datatype和source
            datatype = self.format_datatype(self.datatype)  # exists check
            source = self.format_source(self.source)  # exists check

            # 检查default
            if isinstance(self.default, str):
                assert datatype == 'String', f'{self.name}: default和datatype的类型不兼容，请检查'
            elif isinstance(self.default, int):
                assert datatype in ('Int', 'Float32'), f'{self.name}: default和datatype的类型不兼容，请检查'
            elif isinstance(self.default, float):
                assert datatype in ('Int', 'Float32'), f'{self.name}: default和datatype的类型不兼容，请检查'
            elif isinstance(self.default, list) or isinstance(self.default, tuple):
                assert datatype.endswith('Vec'), f'{self.name}: default和datatype的类型不兼容，请检查'
                # TODO: 检查内部数据类型
                self.default = list(self.default)  # tuple-->list
            elif isinstance(self.default, dict):
                assert datatype.startswith('Map'), f'{self.name}: default和datatype的类型不兼容，请检查'
                # TODO: 检查内部数据类型
            else:
                raise ValueError(f'{self.name}: default must be String/Int/Float32/VecXXX/MapXXXYYY，请检查')
        
            # 检查is_feature
            if self.is_feature and (isinstance(self.default, dict) or datatype.startswith('Map')):
                raise ValueError(f'{self.name}: is_feature为True时，default和datatype都不能为dict/Map')
        
        if self.is_feature and (self.expr or self.expr_params or self.expr_out):
            raise ValueError(f'{self.name}: is_feature为True时，expr相关参数必须为空')

        if not self.is_feature and (
                self.expr is None
                or not isinstance(self.expr, str)
                or (self.source is not None and self.name not in self.expr)
        ):
            raise ValueError(f'{self.name}: is_feature为False时，expr不能为空必须是string，且画像名必须被expr引用')

        # 检查expr_params
        if self.expr_params is not None:
            assert isinstance(self.expr_params, dict), f'{self.name}: expr_params必须是dict'
            if len(self.expr_params) == 0:
                raise ValueError(f'{self.name}: expr_params为空dict，请传入None或非空dict')
            for k in self.expr_params.keys():
                if k not in self.expr:
                    raise ValueError(f'{self.name}: expr_params中的key({k})未被expr引用')

        # TODO: 检查expr: expr_params
        
        # 检查tf_input
        assert self.tf_input.name == (self.expr_out or self.name), 'self.tf_input.name != (self.expr_out or self.name)'

    def format_source(self, source):
        if isinstance(source, tuple) or isinstance(source, list):
            assert len(source) in (3, 4), f'{self.name}: len(source) in (3,4)'
            source = {
                'key_type': source[0],
                'pid': source[1],
                'field': source[2],
            }
            if len(source) == 4:
                source['no_cache'] = source[3]
        elif isinstance(source, dict):
            pass # do nothing
        else:
            raise ValueError(f'{self.name}: source must be tuple/list/dict')
        
        assert isinstance(source['key_type'], str) \
            and isinstance(source['pid'], int) \
            and isinstance(source['field'], str) \
            and isinstance(source.get('no_cache', False), bool), \
            f'{self.name}: source字段数据类型错误：key_type[0] is str, pid[1] is int, field[2] is str, no_cache[3] is bool'
        return source

    def format_datatype(self, datatype):
        if datatype.lower() in ('int', 'int32'):
            return 'Int'
        elif datatype.lower() in ('float', 'float32'):
            return 'Float32'
        elif datatype.lower() in ('str', 'string'):
            return 'String'
        elif datatype in ('MapStringFloat32', 'MapIntFloat32', 'Float32Vec', 'StringVec'):
            return datatype
        elif datatype.lower() in ('float64', 'int64', 'double', 'long'):
            raise ValueError(f'{self.name}: 暂不支持64位数据类型datatype，请使用32位')
        else:
            raise ValueError(f'{self.name}: "{datatype}" is not supported. Please use Int/Float32/String/Map/Vec')

    def infer_datatype(self, value):
        if isinstance(value, str): return 'String', value
        if isinstance(value, int): return 'Int', value
        if isinstance(value, float): return 'Float32', value
        if isinstance(value, list) or isinstance(value, tuple):
            if len(value) == 0: raise ValueError(f'{self.name}: expr_params为列表时，暂不支持为空')
            if not isinstance(value[0], int) and not isinstance(value[0], float) and not isinstance(value[0], str):
                raise ValueError(f'{self.name}: expr_params列表内的元素只支持Int/Float32/String基本类型')
            for i in range(1, len(value)):
                if type(value[i])!=type(value[i-1]):
                    raise ValueError(f'{self.name}: expr_params列表内的元素存在多种数据类型({type(value[i])!=type(value[i-1])})')
            return self.infer_datatype(value[0])[0] + 'Vec', list(value)
        if isinstance(value, dict):
            raise ValueError(f'{self.name}: expr_params暂不支持为dict')

    def __str__(self) -> str:
        return str({
            'name': self.name,
            'default': self.default,
            'datatype': self.datatype,
            'source': self.source,
            'is_feature': self.is_feature,
            'expr': self.expr,
            'expr_params': self.expr_params,
            'expr_out': self.expr_out,
            'tf_input': self.tf_input
        })

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def gen_feature_json(persona_and_features, version=1):
        import json
        schema_json, features_json = dict(), list()
        for pf in persona_and_features:
            if pf.schema is not None:
                schema_json[pf.name] = pf.schema
            if pf.features is not None:
                features_json.append(pf.features)
        feature_json = {'schema': schema_json, 'features': features_json}
        if version == 2:
            from copy import deepcopy
            feature_json_v2 = {'version': 2}
            feature_json_v2.update(deepcopy(feature_json))
            for name, info in feature_json_v2['schema'].items():
                if 'pid' in info['source']:
                    pid = info['source'].pop('pid')
                    if pid > 0:
                        info['source']['field'] = str(pid) + ':' + info['source']['field']
            return feature_json_v2
        else:
            return feature_json

