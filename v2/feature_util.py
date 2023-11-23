import sys

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
    def __init__(self, name, shape=1, dtype='float32', op=None, op_num_x=None, op_num_y=None, emb_dim=None, missing=None, engine=None):
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

class SparseFeature(MetaFeature):
    ''' 类别特征：sparse / category
        - name     ：特征列名
        - vocab    ：List, 枚举所有类别
        - emb_dim  ：embedding dimension
        - dtype    ：默认string，可选int32
        - missing  ：暂无用，仅用于备注
    '''
    def __init__(self, name, vocab=(), emb_dim=8, share_emb=None, dtype='int32', missing=None, min_cnt=1, engine=None):
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

class VarLenFeature(MetaFeature):
    ''' 变长类别特征
        - name     ：特征列名
        - vocab    ：List, 枚举所有类别
        - emb_dim  ：embedding dimension
        - dtype    ：默认string，可选int32
        - missing  ：暂无用，仅用于备注
    '''
    def __init__(self, name, vocab=(), max_len=9, emb_dim=8, share_emb=None, pooling='mean', dtype='int32', missing=None, sep=' ', min_cnt=1, engine=None):
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

vocab_dict = {
    "media_name": [
        "快手", "VIVO", "OPPO", "华为", "头条", "小米", "应用宝", "其它", "广点通", "百度搜索",
        "魅族", "360手助", "百度品专", "百度手助", "三星", "百度", "消消乐游戏", "ASA", "UC市场", "快手聚星",
        "快手kol", "CPA", "品牌推广", "喜马拉雅", "头条kol", "头条星图", "哔哩哔哩", "华为信息流", "京盟世纪", "233小游戏",
        "趣头条", "北京脉络", "联众至尚", "寸心", "拼多多", "上海欧佑"
    ],
    # city/province: ttdw.dim_province_city_level
    "province": [
        '北京', '上海', '天津', '重庆', '香港', '澳门', '河北', '山西', '内蒙古', '辽宁',
        '吉林', '黑龙江', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北',
        '湖南', '广东', '广西', '海南', '四川', '贵州', '云南', '西藏', '陕西', '甘肃',
        '青海', '宁夏', '新疆', '台湾',
    ],
    "province_long": [
        '北京市', '上海市', '天津市', '重庆市', '香港特别行政区', '澳门特别行政区', '河北省', '山西省', '内蒙古自治区', '辽宁省',
        '吉林省', '黑龙江省', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省',
        '湖南省', '广东省', '广西壮族自治区', '海南省', '四川省', '贵州省', '云南省', '西藏自治区', '陕西省', '甘肃省',
        '青海省', '宁夏回族自治区', '新疆维吾尔自治区', '台湾省',
    ],
    "province_code": [
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43,
    ],
    "city": [
        '北京市', '上海市', '天津市', '重庆市', '香港', '澳门', '石家庄市', '沧州市', '承德市', '秦皇岛市',
        '唐山市', '保定市', '廊坊市', '邢台市', '衡水市', '张家口市', '邯郸市', '任丘市', '太原市', '长治市',
        '大同市', '阳泉市', '朔州市', '临汾市', '晋城市', '忻州市', '运城市', '晋中市', '吕梁市', '呼和浩特市',
        '包头市', '赤峰市', '鄂尔多斯市', '呼伦贝尔市', '临河市', '阿拉善', '乌兰浩特市', '通辽市', '乌海市', '集宁市',
        '锡林浩特市', '乌兰察布市', '兴安盟', '巴彦淖尔市', '锡林郭勒盟', '阿拉善盟', '沈阳市', '大连市', '本溪市', '阜新市',
        '葫芦岛市', '盘锦市', '铁岭市', '丹东市', '锦州市', '营口市', '鞍山市', '辽阳市', '抚顺市', '朝阳市',
        '长春市', '白城市', '白山市', '吉林市', '辽源市', '四平市', '通化市', '松原市', '延边', '哈尔滨市',
        '大庆市', '大兴安岭', '鹤岗市', '黑河市', '鸡西市', '佳木斯市', '牡丹江市', '七台河市', '双鸭山市', '齐齐哈尔市',
        '伊春市', '绥化市', '南京市', '苏州市', '扬州市', '无锡市', '南通市', '常州市', '连云港市', '徐州市',
        '镇江市', '淮安市', '宿迁市', '泰州市', '太仓市', '盐城市', '杭州市', '宁波市', '温州市', '丽水市',
        '奉化市', '宁海市', '临海市', '三门市', '绍兴市', '舟山市', '义乌市', '北仑市', '慈溪市', '象山市',
        '余姚市', '天台市', '温岭市', '仙居市', '台州市', '嘉兴市', '湖州市', '衢州市', '金华市', '合肥市',
        '黄山市', '芜湖市', '蚌埠市', '淮南市', '马鞍山市', '淮北市', '铜陵市', '安庆市', '滁州市', '阜阳市',
        '宿州市', '巢湖市', '六安市', '亳州市', '池州市', '宣城市', '福州市', '厦门市', '泉州市', '漳州市',
        '三明市', '莆田市', '南平市', '龙岩市', '宁德市', '南昌市', '九江市', '赣州市', '景德镇市', '萍乡市',
        '新余市', '吉安市', '宜春市', '抚州市', '上饶市', '鹰潭市', '济南市', '青岛市', '烟台市', '威海市',
        '潍坊市', '德州市', '滨州市', '东营市', '聊城市', '菏泽市', '济宁市', '临沂市', '淄博市', '泰安市',
        '枣庄市', '日照市', '莱芜市', '郑州市', '安阳市', '济源市', '鹤壁市', '焦作市', '开封市', '濮阳市',
        '三门峡市', '驻马店市', '商丘市', '新乡市', '信阳市', '许昌市', '周口市', '南阳市', '洛阳市', '平顶山市',
        '漯河市', '武汉市', '十堰市', '宜昌市', '黄石市', '襄阳市', '荆州市', '荆门市', '鄂州市', '孝感市',
        '黄冈市', '咸宁市', '随州市', '恩施', '仙桃市', '天门市', '潜江市', '神农架林区市', '长沙市', '张家界市',
        '株洲市', '韶山市', '衡阳市', '郴州市', '湘潭市', '邵阳市', '岳阳市', '常德市', '益阳市', '永州市',
        '怀化市', '娄底市', '吉首市', '湘西', '云浮市', '广州市', '深圳市', '珠海市', '东莞市', '佛山市',
        '潮州市', '汕头市', '湛江市', '中山市', '惠州市', '河源市', '韶关市', '梅州市', '汕尾市', '江门市',
        '阳江市', '茂名市', '肇庆市', '清远市', '揭阳市', '南宁市', '柳州市', '北海市', '百色市', '梧州市',
        '贺州市', '玉林市', '河池市', '桂林市', '钦州市', '防城港市', '贵港市', '来宾市', '崇左市', '海口市',
        '三亚市', '五指山市', '琼海市', '儋州市', '文昌市', '万宁市', '东方市', '临高县', '乐东', '保亭',
        '定安县', '屯昌县', '昌江', '澄迈县', '琼中', '白沙', '陵水', '三沙市', '成都市', '内江市',
        '峨眉山市', '绵阳市', '宜宾市', '泸州市', '攀枝花市', '自贡市', '资阳市', '崇州市', '德阳市', '南充市',
        '广元市', '遂宁市', '乐山市', '广安市', '达州市', '巴中市', '雅安市', '眉山市', '阿坝', '甘孜',
        '凉山', '贵阳市', '六盘水市', '遵义市', '安顺市', '铜仁市', '毕节市', '兴义市', '凯里市', '都匀市',
        '黔东南', '黔南', '黔西南', '昆明市', '西双版纳', '大理', '曲靖市', '玉溪市', '保山市', '昭通市',
        '普洱市', '临沧市', '丽江市', '文山', '楚雄', '德宏', '怒江', '红河', '迪庆', '林芝市',
        '拉萨市', '那曲市', '昌都市', '山南市', '日喀则市', '阿里', '西安市', '宝鸡市', '铜川市', '咸阳市',
        '渭南市', '延安市', '汉中市', '榆林市', '安康市', '商洛市', '兰州市', '白银市', '天水市', '嘉峪关市',
        '武威市', '张掖市', '平凉市', '酒泉市', '庆阳市', '定西市', '陇南市', '临夏', '甘南', '金昌市',
        '西宁市', '海东市', '海北', '黄南', '果洛', '海西', '玉树', '海南', '银川市', '石嘴山市',
        '吴忠市', '固原市', '中卫市', '乌鲁木齐市', '克拉玛依市', '石河子市', '图木舒克市', '吐鲁番市', '哈密市', '和田',
        '喀什', '昌吉', '阿图什市', '库尔勒市', '博乐市', '伊宁市', '阿拉尔市', '阿克苏', '五家渠市', '伊犁',
        '克孜勒苏', '博尔塔拉', '塔城', '巴音郭楞', '阿勒泰', '台北', '台中', '台南', '高雄', '云林',
        '南投县', '台东', '嘉义', '基隆', '宜兰县', '屏东县', '彰化县', '新竹', '桃园', '澎湖县',
        '花莲县', '苗栗县',
    ],
    "city_long": [
        '北京市', '上海市', '天津市', '重庆市', '香港', '澳门', '石家庄市', '沧州市', '承德市', '秦皇岛市',
        '唐山市', '保定市', '廊坊市', '邢台市', '衡水市', '张家口市', '邯郸市', '任丘市', '太原市', '长治市',
        '大同市', '阳泉市', '朔州市', '临汾市', '晋城市', '忻州市', '运城市', '晋中市', '吕梁市', '呼和浩特市',
        '包头市', '赤峰市', '鄂尔多斯市', '呼伦贝尔市', '临河市', '阿拉善左旗', '乌兰浩特市', '通辽市', '乌海市', '集宁市',
        '锡林浩特市', '乌兰察布市', '兴安盟', '巴彦淖尔市', '锡林郭勒盟', '阿拉善盟', '沈阳市', '大连市', '本溪市', '阜新市',
        '葫芦岛市', '盘锦市', '铁岭市', '丹东市', '锦州市', '营口市', '鞍山市', '辽阳市', '抚顺市', '朝阳市',
        '长春市', '白城市', '白山市', '吉林市', '辽源市', '四平市', '通化市', '松原市', '延边州', '哈尔滨市',
        '大庆市', '大兴安岭', '鹤岗市', '黑河市', '鸡西市', '佳木斯市', '牡丹江市', '七台河市', '双鸭山市', '齐齐哈尔市',
        '伊春市', '绥化市', '南京市', '苏州市', '扬州市', '无锡市', '南通市', '常州市', '连云港市', '徐州市',
        '镇江市', '淮安市', '宿迁市', '泰州市', '太仓市', '盐城市', '杭州市', '宁波市', '温州市', '丽水市',
        '奉化市', '宁海市', '临海市', '三门市', '绍兴市', '舟山市', '义乌市', '北仑市', '慈溪市', '象山市',
        '余姚市', '天台市', '温岭市', '仙居市', '台州市', '嘉兴市', '湖州市', '衢州市', '金华市', '合肥市',
        '黄山市', '芜湖市', '蚌埠市', '淮南市', '马鞍山市', '淮北市', '铜陵市', '安庆市', '滁州市', '阜阳市',
        '宿州市', '巢湖市', '六安市', '亳州市', '池州市', '宣城市', '福州市', '厦门市', '泉州市', '漳州市',
        '三明市', '莆田市', '南平市', '龙岩市', '宁德市', '南昌市', '九江市', '赣州市', '景德镇市', '萍乡市',
        '新余市', '吉安市', '宜春市', '抚州市', '上饶市', '鹰潭市', '济南市', '青岛市', '烟台市', '威海市',
        '潍坊市', '德州市', '滨州市', '东营市', '聊城市', '菏泽市', '济宁市', '临沂市', '淄博市', '泰安市',
        '枣庄市', '日照市', '莱芜市', '郑州市', '安阳市', '济源市', '鹤壁市', '焦作市', '开封市', '濮阳市',
        '三门峡市', '驻马店市', '商丘市', '新乡市', '信阳市', '许昌市', '周口市', '南阳市', '洛阳市', '平顶山市',
        '漯河市', '武汉市', '十堰市', '宜昌市', '黄石市', '襄阳市', '荆州市', '荆门市', '鄂州市', '孝感市',
        '黄冈市', '咸宁市', '随州市', '恩施州', '仙桃市', '天门市', '潜江市', '神农架林区市', '长沙市', '张家界市',
        '株洲市', '韶山市', '衡阳市', '郴州市', '湘潭市', '邵阳市', '岳阳市', '常德市', '益阳市', '永州市',
        '怀化市', '娄底市', '吉首市', '湘西州', '云浮市', '广州市', '深圳市', '珠海市', '东莞市', '佛山市',
        '潮州市', '汕头市', '湛江市', '中山市', '惠州市', '河源市', '韶关市', '梅州市', '汕尾市', '江门市',
        '阳江市', '茂名市', '肇庆市', '清远市', '揭阳市', '南宁市', '柳州市', '北海市', '百色市', '梧州市',
        '贺州市', '玉林市', '河池市', '桂林市', '钦州市', '防城港市', '贵港市', '来宾市', '崇左市', '海口市',
        '三亚市', '五指山市', '琼海市', '儋州市', '文昌市', '万宁市', '东方市', '临高县', '乐东黎族自治县', '保亭黎族苗族自治县',
        '定安县', '屯昌县', '昌江黎族自治县', '澄迈县', '琼中黎族苗族自治县', '白沙黎族自治县', '陵水黎族自治县', '三沙市', '成都市', '内江市',
        '峨眉山市', '绵阳市', '宜宾市', '泸州市', '攀枝花市', '自贡市', '资阳市', '崇州市', '德阳市', '南充市',
        '广元市', '遂宁市', '乐山市', '广安市', '达州市', '巴中市', '雅安市', '眉山市', '阿坝州', '甘孜州',
        '凉山州', '贵阳市', '六盘水市', '遵义市', '安顺市', '铜仁市', '毕节市', '兴义市', '凯里市', '都匀市',
        '黔东南州', '黔南州', '黔西南州', '昆明市', '西双版纳州', '大理州', '曲靖市', '玉溪市', '保山市', '昭通市',
        '普洱市', '临沧市', '丽江市', '文山州', '楚雄州', '德宏州', '怒江州', '红河州', '迪庆州', '林芝市',
        '拉萨市', '那曲市', '昌都市', '山南市', '日喀则市', '阿里地区', '西安市', '宝鸡市', '铜川市', '咸阳市',
        '渭南市', '延安市', '汉中市', '榆林市', '安康市', '商洛市', '兰州市', '白银市', '天水市', '嘉峪关市',
        '武威市', '张掖市', '平凉市', '酒泉市', '庆阳市', '定西市', '陇南市', '临夏州', '甘南州', '金昌市',
        '西宁市', '海东市', '海北州', '黄南州', '果洛州', '海西州', '玉树州', '海南州', '银川市', '石嘴山市',
        '吴忠市', '固原市', '中卫市', '乌鲁木齐市', '克拉玛依市', '石河子市', '图木舒克市', '吐鲁番市', '哈密市', '和田地区',
        '喀什地区', '昌吉州', '阿图什市', '库尔勒市', '博乐市', '伊宁市', '阿拉尔市', '阿克苏地区', '五家渠市', '伊犁州',
        '克孜勒苏州', '博尔塔拉州', '塔城地区', '巴音郭楞州', '阿勒泰地区', '台北市', '台中市', '台南市', '高雄市', '云林市',
        '南投市', '台东市', '嘉义市', '基隆市', '宜兰县', '屏东县', '彰化县', '新竹市', '桃园市', '澎湖县',
        '花莲县', '苗栗县',
    ],
    "city_code": [
        10, 11, 12, 13, 14, 15, 1601, 1602, 1603, 1604,
        1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1701, 1702,
        1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1801,
        1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811,
        1812, 1813, 1814, 1815, 1816, 1817, 1901, 1902, 1903, 1904,
        1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914,
        2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2101,
        2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111,
        2112, 2113, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208,
        2209, 2210, 2211, 2212, 2213, 2214, 2301, 2302, 2303, 2304,
        2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314,
        2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2401,
        2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411,
        2412, 2413, 2414, 2415, 2416, 2417, 2501, 2502, 2503, 2504,
        2505, 2506, 2507, 2508, 2509, 2601, 2602, 2603, 2604, 2605,
        2606, 2607, 2608, 2609, 2610, 2611, 2701, 2702, 2703, 2704,
        2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714,
        2715, 2716, 2717, 2801, 2802, 2803, 2804, 2805, 2806, 2807,
        2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817,
        2818, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909,
        2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 3001, 3002,
        3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012,
        3013, 3014, 3015, 3016, 3101, 3102, 3103, 3104, 3105, 3106,
        3107, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117,
        3118, 3119, 3120, 3121, 3122, 3201, 3202, 3203, 3204, 3205,
        3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3301,
        3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311,
        3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3401, 3402,
        3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412,
        3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422,
        3423, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509,
        3510, 3511, 3512, 3601, 3602, 3604, 3605, 3606, 3607, 3608,
        3609, 3610, 3611, 3612, 3614, 3615, 3616, 3617, 3618, 3701,
        3702, 3703, 3704, 3705, 3706, 3707, 3801, 3802, 3803, 3804,
        3805, 3806, 3807, 3808, 3809, 3810, 3901, 3902, 3903, 3904,
        3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914,
        4001, 4002, 4003, 4004, 4006, 4007, 4008, 4009, 4101, 4102,
        4103, 4104, 4105, 4201, 4202, 4203, 4204, 4205, 4206, 4207,
        4208, 4209, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217,
        4218, 4219, 4220, 4221, 4222, 4301, 4302, 4303, 4304, 4305,
        4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314, 4315,
        4316, 4317,
    ],
    "hw_tag": [
        '人_人', '人体器官_人脸', '其他_地图', '树_树', '电子类_手机截图', '学科科学_互联网', '学习办公类_海报', '自然风景_云', '服饰穿戴类_帽子', '学习办公类_信', '学习办公类_白板',
        '自然风景_天空', '服饰穿戴类_长裤', '自然风景_雪', '学习办公类_书', '其他_玩具', '文化艺术类_动漫', '文化艺术类_音乐会', '其他_记分牌', '卡证文档_证件', '服饰穿戴类_眼镜',
        '自然风景_月亮', '服饰穿戴类_长大衣', '花_花', '动物_猫', '电子类_显示器', '服饰穿戴类_婚纱', '其他_水槽', '教育_报纸', '陆地交通_汽车', '服饰穿戴类_围巾',
        '服饰穿戴类_连衣裙', '人工场景_摩天大楼', '自然风景_雾', '自然风景_湖', '电子类_计算机', '动物_狗', '家居类_相框', '人工场景_酒吧', '服饰穿戴类_衬衫', '家居类_餐桌',
        '服饰穿戴类_牛仔裤', '人工场景_窗口', '文化艺术类_雕像', '其他_轮子', '电子类_笔记本电脑', '家居类_椅子', '食物_蛋糕', '学习办公类_便利贴', '服饰穿戴类_外套',
        '自然风景_山', '动物_鸟', '家居类_镜子', '家居类_杯子', '电子类_平板电脑', '其他日常使用_行李箱', '动物_兔', '服饰穿戴类_高跟鞋', '花_玫瑰', '其他_条幅',
        '水上交通_船', '电子类_移动电话', '家居类_枕头', '学习办公类_工卡', '其他_星星', '厨房类_勺', '服饰穿戴类_短裙', '卡证文档_证件照', '人工场景_烟花',
        '学习办公类_票据', '家居类_日历', '家居类_床', '其他_瓶子', '家居类_伞', '其他_卡通', '学习办公类_纸', '人工场景_路灯', '食物_啤酒', '空中交通_飞机',
        '其他_星球', '厨房类_盘子', '动物_鱼', '其他_身份证', '食物_沙拉', '服饰穿戴类_领带', '树_圣诞树', '其他_台球桌', '运动类_滑板', '其他_香烟', '家居类_楼梯',
        '服饰穿戴类_头盔', '动物_水母', '其他_幕布', '卡证文档_合照', '运动类_球', '自然风景_洞穴', '服饰穿戴类_迷你裙', '人工场景_碑文', '家居类_门', '食物_大米', '花_樱花',
        '家居类_沙发', '其他_安全带', '情感_可爱', '家居类_工艺品', '服饰穿戴类_服装', '服饰穿戴类_短裤', '陆地交通_自行车', '家居类_壁纸', '食物_面包', '学习办公类_直尺',
        '金融商业_时尚', '花_盆栽', '家居类_手电筒', '其他_硬币', '人工场景_便利店', '厨房类_叉子', '食物_三明治', '服饰穿戴类_礼服', '学习办公类_信封', '食物_面条', '其他_肖像',
        '食物_草莓', '食物_煎饼', '陆地交通_摩托车', '家居类_长凳', '学习办公类_黑板', '其他植物_植物', '服饰穿戴类_游泳衣', '服饰穿戴类_头戴耳机', '服饰穿戴类_睡衣',
        '服饰穿戴类_手表', '动物_海豚', '动物_鼠', '武器_刀', '文化艺术类_涂鸦', '动物_羊驼', '食物_蛋', '文化艺术类_写生', '家居类_毛巾', '花_桃花', '服饰穿戴类_太阳镜',
        '服饰穿戴类_背包', '学习办公类_胶带', '陆地交通_面包车', '花_油菜花', '食物_披萨', '学科科学_教育', '家居类_帐篷', '人工场景_雪人', '医疗医护_药', '医疗医护_听诊器',
        '动作_单板滑雪', '食物_冰淇淋', '厨房类_擦洗刷', '人工场景_游泳池', '家居类_浴缸', '服饰穿戴类_泳装', '食物_海鲜', '服饰穿戴类_项链', '学习办公类_信笺', '化妆类_化妆品',
        '自然风景_彩虹', '动物_章鱼', '学习办公类_旗帜', '家居类_婴儿床', '人_成年人', '食物_胡萝卜', '食物_火腿', '动物_海鸥', '人_儿童', '情感_爱', '陆地交通_交通标志', '服饰穿戴类_凉鞋',
        '动物_动物', '动作_庆祝', '人工场景_广告牌', '食物_盒饭', '文化艺术类_宗教', '动物_蝴蝶', '文化艺术类_吉他', '动物_猪', '家居类_灯泡', '食物_汉堡包', '食物_咖啡', '其他_机器人',
        '服饰穿戴类_假发', '厨房类_煎锅', '食物_粥', '服饰穿戴类_靴子', '家居类_遥控器', '家居类_发票', '服饰穿戴类_跑步鞋', '其他_性感', '其他_气球', '家居类_吹风机', '服饰穿戴类_滑板鞋',
        '服饰穿戴类_拐杖', '家居类_灯', '其他_火', '动物_恐龙', '花_郁金香', '人工场景_城市', '化妆类_唇膏', '运动类_球拍', '草_芦苇', '人工场景_服装店', '运动类_哑铃', '服饰穿戴类_T恤',
        '厨房类_切肉刀', '家居类_柜子', '食物_寿司', '动物_鹿', '服饰穿戴类_球服', '文化艺术类_雕塑', '其他_旅行', '家居类_雨衣', '草_爬山虎', '医疗医护_氧气面罩', '服饰穿戴类_耳环',
        '动物_象', '食物_大枣', '动物_鸭', '运动类_冲浪板', '食物_鸡大腿', '动物_蛇', '食物_南瓜', '厨房类_碗', '食物_橙子', '服饰穿戴类_和服', '学习办公类_文件夹', '食物_香肠',
        '自然风景_瀑布', '花_荷花', '食物_米粉', '其他_花盆', '花_康乃馨', '学习办公类_打印机', '其他_蜡烛', '服饰穿戴类_风衣', '医疗医护_创可贴', '动物_狐狸', '家居类_高脚杯',
        '家居类_花瓶', '人工场景_人行横道', '其他_秋千', '家居类_电风扇', '家居类_熨斗', '厨房类_锅', '事件_运动', '社会_古代', '食物_葡萄干', '食物_华夫饼', '文化艺术类_钢琴',
        '电子类_电脑键盘', '动物_虾', '家居类_架子', '家居类_塑料袋', '武器_火箭', '服饰穿戴类_开衫', '家居类_睡袋', '电子类_鼠标', '家居类_窗帘', '运动类_弓箭', '服饰穿戴类_裙子',
        '其他_制造业', '人工场景_灯塔', '动物_马', '电子类_麦克风', '服饰穿戴类_袜子', '动物_浣熊', '厨房类_冰箱', '食物_饺子', '服饰穿戴类_大褂', '食物_食物', '人工场景_厕所',
        '食物_饼干', '化妆类_肥皂', '自然风景_海洋', '其他_漫画', '动物_鱼子', '电子类_充电器', '卡证文档_证章', '动作_烤肉', '陆地交通_火车', '食物_曲奇饼', '服饰穿戴类_罩裙',
        '食物_香蕉', '食物_牛肉', '家居类_浴室', '人工场景_舞台', '电子类_屏幕', '人工场景_办公室', '服饰穿戴类_钱包', '动物_蝙蝠', '电子类_台式电脑', '食物_桑葚', '动物_狮子',
        '服饰穿戴类_皮带', '厨房类_烤箱', '家居类_时钟', '动物_麻雀', '陆地交通_后视镜', '学习办公类_放大镜', '动物_蜘蛛', '食物_苹果', '食物_柠檬', '花_月季', '食物_西兰花',
        '文化艺术类_拨弦片', '文化艺术类_十字架', '人工场景_儿童房', '家居类_橱柜', '食物_羊肉', '服饰穿戴类_太阳帽', '服饰穿戴类_学位袍', '空中交通_热气球', '动物_虎', '树_枫树',
        '食物_意大利面', '食物_卷心菜', '食物_辣椒', '家居类_百叶窗', '人工场景_水族馆', '动物_鲨鱼', '家居类_水壶', '家居类_被子', '其他_旗杆', '电子类_智能手机', '动物_鸽子',
        '人工场景_公墓', '动物_海马', '人工场景_风车', '动物_毛虫', '其他_服务', '动物_鸡', '食物_火龙果', '家居类_拖把', '人工场景_医院的房间', '生活_都市', '动物_小鸡', '人工场景_喷泉',
        '家居类_牙刷', '厨房类_炖锅', '食物_芒果', '文化艺术类_铜版画', '人工场景_工作室', '运动类_乒乓球', '花_向日葵', '家居类_钟', '食物_西瓜', '服饰穿戴类_防毒面具', '动作_书写',
        '家居类_秒表', '自然风景_地球', '家居类_洗衣机', '文化艺术类_口琴', '教育_纵横字谜',
    ],
}

# 定义所有特征
ALL_FEATURE_COLUMNS = [
    # # 1. 用户侧
    # ## 1.1 属性类
    SparseFeature('u_gender',                       vocab=(0,1),                    emb_dim=8, dtype='int32',   missing=-1),
    SparseFeature('u_age_group',                    vocab=(0,1,2,3,4,5,6),          emb_dim=8, dtype='int32',   missing=0),
    SparseFeature('u_platform',                     vocab=(0,1,5),                  emb_dim=8, dtype='int32',   missing=-1),
    SparseFeature('u_user_last_login_city_level',   vocab=(1,2,3,4,5,10),           emb_dim=8, dtype='int32',   missing=0),
    SparseFeature('u_tg_channel_type',              vocab=tuple(range(1,17)),       emb_dim=8, dtype='int32',   missing=0),
    SparseFeature('u_is_new',                       vocab=(0,1),                    emb_dim=8, dtype='int32',   missing=-1),
    SparseFeature('u_media_name',                   vocab=vocab_dict['media_name'], emb_dim=8, dtype='string',  missing=''),
    SparseFeature('u_user_last_login_city_id',      vocab=vocab_dict['city_code'],  emb_dim=8, dtype='int32',   missing=0),
    SparseFeature('u_user_last_login_province',     vocab=vocab_dict['province_long'],emb_dim=8,dtype='string', missing=''),
    DenseFeature('u_age',                               dtype='float32', op='normal',        missing=0),

    # ## 1.2 统计类
    # ### 1.2.1 普通
    *[DenseFeature(f'u_log_days_cnt_{n}d',              dtype='float32', op='log_normal',    missing=0) for n in (14,7,3)],
    *[DenseFeature(f'u_view_post_cnt_{n}d',             dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'u_click_poster_ratio_{n}d',        dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'u_comment_poster_ratio_{n}d',      dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'u_view_rec_post_cnt_{n}d',         dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'u_click_rec_poster_ratio_{n}d',    dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'u_comment_rec_poster_ratio_{n}d',  dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'u_avg_view_time_{n}d',             dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'u_rec_avg_view_time_{n}d',         dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'u_valid_view_post_ratio_{n}d',     dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'u_valid_view_rec_post_ratio_{n}d', dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    # ### 1.2.2 偏好
    #TODO: DenseFeature('u_post_type_click_ratio', shape=4, dtype='float32', op=None), missing=0,   # 2:0.875,3:0.125 {1,2,3,5}
    *[DenseFeature(f'u_ctr_p_male_{n}d',                dtype='float32', op='normal',     missing=0.166667) for n in(14,7,3)],
    *[DenseFeature(f'u_ctr_p_female_{n}d',              dtype='float32', op='normal',     missing=0.166667) for n in(14,7,3)],
    *[DenseFeature(f'u_ctr_odd_p_female_{n}d',          dtype='float32', op='normal',     missing=1.0) for n in (14,7,3)],
    VarLenFeature('u_thump_up_image_hw_tag_seq', dtype='string', vocab=vocab_dict['hw_tag'], max_len=20, emb_dim=8, pooling='attention', missing=''),
    VarLenFeature('u_comment_image_hw_tag_seq', dtype='string', vocab=vocab_dict['hw_tag'], max_len=20, emb_dim=8, pooling='attention', missing=''),
    # ## 1.3 社交类
    DenseFeature('u_social_friend_cnt',                 dtype='float32', op='log_normal',    missing=0),
    DenseFeature('u_social_male_friend_cnt',            dtype='float32', op='log_normal',    missing=0),
    DenseFeature('u_fan_cnt',                           dtype='float32', op='log_normal',    missing=0),
    DenseFeature('u_male_fan_cnt',                      dtype='float32', op='log_normal',    missing=0),
    DenseFeature('u_follow_cnt',                        dtype='float32', op='log_normal',    missing=0),
    DenseFeature('u_female_follow_cnt',                 dtype='float32', op='log_normal',    missing=0),
    DenseFeature('u_related_ucnt',                      dtype='float32', op='log_normal',    missing=0),


    # # 2. 物品侧
    # ## 2.1 帖子
    # ### 2.1.1 属性类
    SparseFeature('i_type_id',                      vocab=(1,2,3,5),                emb_dim=8, dtype='int32',   missing=0),
    SparseFeature('i_diff',                         vocab=(0,1,2,3),                emb_dim=8, dtype='int32',   missing=-1),
    SparseFeature('i_std_platform',                 vocab=(1,2),                    emb_dim=8, dtype='int32',   missing=0),
    DenseFeature('i_text_emb',      shape=64,           dtype='float32', op=None,             missing=[0]*64),
    DenseFeature('i_text_humanlen',                     dtype='float32', op='log_normal',     missing=0),
    VarLenFeature('i_images_hw_tags', dtype='string', vocab=vocab_dict['hw_tag'], max_len=9, emb_dim=8, pooling='attention_no_mask', missing=''),
    # ### 2.1.2 统计类
    *[DenseFeature(f'i_view_cnt_{n}d',                  dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'i_click_ratio_{n}d',               dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'i_comment_ratio_{n}d',             dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'i_avg_stay_time_{n}d',             dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'i_valid_view_ratio_{n}d',          dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'i_v_publisher_click_cnt_{n}d',     dtype='float32', op='log_normal',    missing=0) for n in (7,3)],
    DenseFeature('i_view_cnt_12h',                      dtype='float32', op='log_normal',     missing=0),
    DenseFeature('i_thump_up_cnt_12h',                  dtype='float32', op='log_normal',     missing=0),
    DenseFeature('i_comment_cnt_12h',                   dtype='float32', op='log_normal',     missing=0),

    # ## 2.2 发布者
    # ### 2.2.1 属性类
    SparseFeature('p_gender',                       vocab=(0,1),                    emb_dim=8, dtype='int32',   missing=-1),
    SparseFeature('p_age_group',                    vocab=(0,1,2,3,4,5,6),          emb_dim=8, dtype='int32',   missing=0),
    SparseFeature('p_platform',                     vocab=(0,1,5),                  emb_dim=8, dtype='int32',   missing=-1),
    SparseFeature('p_user_last_login_city_level',   vocab=(1,2,3,4,5,10),           emb_dim=8, dtype='int32',   missing=0),
    SparseFeature('p_tg_channel_type',              vocab=tuple(range(1,17)),       emb_dim=8, dtype='int32',   missing=0),
    SparseFeature('p_is_new',                       vocab=(0,1),                    emb_dim=8, dtype='int32',   missing=-1),
    SparseFeature('p_media_name',                   vocab=vocab_dict['media_name'], emb_dim=8, dtype='string',  missing=''),
    SparseFeature('p_user_last_login_city_id',      vocab=vocab_dict['city_code'],  emb_dim=8, dtype='int32',   missing=0),
    SparseFeature('p_user_last_login_province',     vocab=vocab_dict['province_long'],emb_dim=8,dtype='string', missing=''),
    DenseFeature('p_age',                               dtype='float32', op='normal',        missing=0),
    # ### 2.2.2 统计类
    *[DenseFeature(f'p_publish_post_cnt_{n}d',          dtype='float32', op='log_normal',    missing=0) for n in (28,14,7)],
    *[DenseFeature(f'p_viewed_post_cnt_{n}d',           dtype='float32', op='log_normal',    missing=0) for n in (28,14,7)],
    *[DenseFeature(f'p_personal_page_view_ucnt_{n}d',   dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)], #drop28 for all -1
    DenseFeature('p_comment_cnt_sum_28d',               dtype='float32', op='log_normal',    missing=0),
    DenseFeature('p_reply_comment_ratio_28d',           dtype='float32', op='normal',        missing=0),
    DenseFeature('p_choice_post_cnt_28d',               dtype='float32', op='log_normal',    missing=0),
    DenseFeature('p_avg_view_click_ratio_28d',          dtype='float32', op='normal',        missing=0),
    DenseFeature('p_avg_view_comment_ratio_28d',        dtype='float32', op='normal',        missing=0),
    DenseFeature('p_max_view_click_ratio_28d',          dtype='float32', op='normal',        missing=0),
    DenseFeature('p_max_view_comment_ratio_28d',        dtype='float32', op='normal',        missing=0),
    *[DenseFeature(f'p_active_im_ucnt_{n}d',            dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'p_active_im_reply_ratio_{n}d',     dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'p_passive_im_ucnt_{n}d',           dtype='float32', op='log_normal',    missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'p_passive_im_reply_ratio_{n}d',    dtype='float32', op='normal',        missing=0) for n in (14,7,3,1)],
    *[DenseFeature(f'p_valid_viewed_post_ratio_{n}d',   dtype='float32', op='normal',        missing=0) for n in (28,14,7)],
    # ### 2.2.3 社交类
    DenseFeature('p_social_friend_cnt_his',             dtype='float32', op='log_normal',    missing=0),
    DenseFeature('p_social_male_friend_cnt_his',        dtype='float32', op='log_normal',    missing=0),
    DenseFeature('p_fan_cnt',                           dtype='float32', op='log_normal',    missing=0),
    DenseFeature('p_male_fan_cnt',                      dtype='float32', op='log_normal',    missing=0),
    DenseFeature('p_follow_cnt',                        dtype='float32', op='log_normal',    missing=0),
    DenseFeature('p_female_follow_cnt',                 dtype='float32', op='log_normal',    missing=0),
    DenseFeature('p_related_ucnt',                      dtype='float32', op='log_normal',    missing=0),

    # # 3. 其他
    SparseFeature('min_position_id', vocab=list(range(0, 300)), emb_dim=1, dtype='int32', missing=-1),  # 曝光位置，用于消除位置偏差
]
ALL_FEATURE_COLUMNS_DICT = {fc.name:fc for fc in ALL_FEATURE_COLUMNS}

# 定义engine字段{source, ...}
ENGINE_SOURCE = {
    'u_gender':                     {"key_type": "user_id",         "pid": 10001,       "field": "sex"},
    'u_age_group':                  {"key_type": "user_id",         "pid": 10001,       "field": "age_group"},
    'u_platform':                   {"key_type": "user_id",         "pid": 10002,       "field": "last_os_type"},
    'u_user_last_login_city_level': {"key_type": "user_id",         "pid": 11039,       "field": "u_user_last_login_city_level"},
    'u_ctr_p_male_14d':             {"key_type": "user_id",         "pid": 110110600,   "field": "u_ctr_p_male_14d"},
    'u_ctr_p_male_3d':              {"key_type": "user_id",         "pid": 110110600,   "field": "u_ctr_p_male_3d"},
    'u_ctr_p_female_14d':           {"key_type": "user_id",         "pid": 110110600,   "field": "u_ctr_p_female_14d"},
    'u_ctr_p_female_3d':            {"key_type": "user_id",         "pid": 110110600,   "field": "u_ctr_p_female_3d"},
    'u_ctr_odd_p_female_14d':       {"key_type": "user_id",         "pid": 110110600,   "field": "u_ctr_odd_p_female_14d"},
    'u_ctr_odd_p_female_3d':        {"key_type": "user_id",         "pid": 110110600,   "field": "u_ctr_odd_p_female_3d"},
    'u_social_friend_cnt':          {"key_type": "user_id",         "pid": 10008,       "field": "friend_num"},
    'u_social_male_friend_cnt':     {"key_type": "user_id",         "pid": 10008,       "field": "friend_num_1"},
    'u_view_post_cnt_14d':          {"key_type": "user_id",         "pid": 11039,       "field": "u_view_post_cnt_14d"},
    'u_view_rec_post_cnt_14d':      {"key_type": "user_id",         "pid": 11039,       "field": "u_view_rec_post_cnt_14d"},
    'u_click_poster_ratio_14d':     {"key_type": "user_id",         "pid": 11039,       "field": "u_click_poster_ratio_14d"},
    'u_click_poster_ratio_3d':      {"key_type": "user_id",         "pid": 11039,       "field": "u_click_poster_ratio_3d"},
    'u_click_rec_poster_ratio_14d': {"key_type": "user_id",         "pid": 11039,       "field": "u_click_rec_poster_ratio_14d"},
    'u_click_rec_poster_ratio_3d':  {"key_type": "user_id",         "pid": 11039,       "field": "u_click_rec_poster_ratio_3d"},
    'i_type_id':                    {"key_type": "item_id",         "pid": 11001,       "field": "post_type"},
    'i_diff':                       {"key_type": "item_id",         "pid": 11041,       "field": "i_diff"},
    'i_text_humanlen':              {"key_type": "item_id",         "pid": 110313000,   "field": "i_text_humanlen"},
    'i_text_emb':                   {"key_type": "item_id",         "pid": 110313000,   "field": "i_text_emb"},
    'i_view_cnt_14d':               {"key_type": "item_id",         "pid": 11041,       "field": "i_view_cnt_14d"},
    'i_view_cnt_1d':                {"key_type": "item_id",         "pid": 11041,       "field": "i_view_cnt_1d"},
    'i_click_ratio_14d':            {"key_type": "item_id",         "pid": 11041,       "field": "i_click_ratio_14d"},
    'i_click_ratio_1d':             {"key_type": "item_id",         "pid": 11041,       "field": "i_click_ratio_1d"},
    'i_comment_ratio_14d':          {"key_type": "item_id",         "pid": 11041,       "field": "i_comment_ratio_14d"},
    'i_comment_ratio_7d':           {"key_type": "item_id",         "pid": 11041,       "field": "i_comment_ratio_7d"},
    'i_comment_ratio_3d':           {"key_type": "item_id",         "pid": 11041,       "field": "i_comment_ratio_3d"},
    'i_comment_ratio_1d':           {"key_type": "item_id",         "pid": 11041,       "field": "i_comment_ratio_1d"},
    'p_gender':                     {"key_type": "item_owner_id",   "pid": 10001,       "field": "sex"},
    'p_age_group':                  {"key_type": "item_owner_id",   "pid": 10001,       "field": "age_group"},
    'p_platform':                   {"key_type": "item_owner_id",   "pid": 10002,       "field": "last_os_type"},
    'p_user_last_login_city_level': {"key_type": "item_owner_id",   "pid": 11040,       "field": "p_user_last_login_city_level"},
    'p_fan_cnt':                    {"key_type": "item_owner_id",   "pid": 11040,       "field": "p_fan_cnt"},
    'p_male_fan_cnt':               {"key_type": "item_owner_id",   "pid": 11040,       "field": "p_male_fan_cnt"},
    'p_personal_page_view_ucnt_14d':{"key_type": "item_owner_id",   "pid": 11040,       "field": "p_personal_page_view_ucnt_14d"},
    'p_personal_page_view_ucnt_3d': {"key_type": "item_owner_id",   "pid": 11040,       "field": "p_personal_page_view_ucnt_3d"},
    'p_comment_cnt_sum_28d':        {"key_type": "item_owner_id",   "pid": 11040,       "field": "p_comment_cnt_sum_28d"},
    'p_avg_view_click_ratio_28d':   {"key_type": "item_owner_id",   "pid": 11040,       "field": "p_avg_view_click_ratio_28d"},
    'p_max_view_click_ratio_28d':   {"key_type": "item_owner_id",   "pid": 11040,       "field": "p_max_view_click_ratio_28d"},
    'p_avg_view_comment_ratio_28d': {"key_type": "item_owner_id",   "pid": 11040,       "field": "p_avg_view_comment_ratio_28d"},
    'p_max_view_comment_ratio_28d': {"key_type": "item_owner_id",   "pid": 11040,       "field": "p_max_view_comment_ratio_28d"},
    'i_view_cnt_12h':               {"key_type": "item_id",         "pid": 110312000,   "field": "i_view_cnt_12h"},
    'i_comment_cnt_12h':            {"key_type": "item_id",         "pid": 110312000,   "field": "i_comment_cnt_12h"},
    'i_thump_up_cnt_12h':           {"key_type": "item_id",         "pid": 110312000,   "field": "i_thump_up_cnt_12h"},
    'i_images_hw_tags':             {"key_type": "item_id",         "pid": 11001,       "field": "image_tags"},
    'u_thump_up_image_hw_tag_seq':  {"key_type": "user_id",         "pid": 110111300,   "field": "u_thump_up_image_hw_tag_seq"},
    'u_comment_image_hw_tag_seq':   {"key_type": "user_id",         "pid": 110111300,   "field": "u_comment_image_hw_tag_seq"},
}
for fn, fc in ALL_FEATURE_COLUMNS_DICT.items():
    if fn in ENGINE_SOURCE:
        fc.engine['source'] = ENGINE_SOURCE[fn]


def features2columns(features, all_feature_columns_dict=ALL_FEATURE_COLUMNS_DICT, ver=None, warning=True):
    fcs = list()
    for feat in parse_raw_features(features):
        if feat[:2] in ('u_', 'p_', 'i_') and feat != 'u_post_type_prefer' and not feat.startswith('u_p_'):
            fcs.append(all_feature_columns_dict[feat])
        elif feat in ('min_position_id', ):
            fcs.append(all_feature_columns_dict[feat])
        else:
            if warning:
                print(f'WARNING: ver={ver}: {feat} ignore')
    return fcs

def parse_raw_features(raw_features, verbose=False):
    import re
    re_pattern = re.compile(r'\{.+?\}')
    features = list()
    for x in raw_features:
        feat_name = x.strip()
        if feat_name == '' or feat_name.startswith('#'):
            if verbose and feat_name:
                print('%46s --> %s' % (feat_name, "[drop for comment(#)]"))
            continue
        if '{' not in feat_name:
            features.append(feat_name)
            if verbose:
                print('%46s --> %s' % (feat_name, feat_name))
        else:
            feat_name_list = [feat_name]
            patterns = re_pattern.findall(feat_name)
            for ps_str in patterns:
                ps = [p.strip().strip(r'{}') for p in ps_str.split('/')]
                l = len(feat_name_list)
                for i in range(l):
                    fn = feat_name_list.pop(0)
                    for p in ps:
                        feat_name_list.append(fn.replace(ps_str, p))
            features.extend(feat_name_list)
            if verbose:
                print('%46s --> %s' % (feat_name, feat_name_list))
    return features


# 定义所有特征组版本
# 每个模型需要定义一个特征组版本
feature_versions = {
    'xgb_v1': [
        'u_gender', 'u_age_group', 'u_platform', 'u_social_friend_cnt', 'u_social_male_friend_cnt', 'u_tg_channel_type',
        'u_click_rec_poster_ratio_7d', 'u_click_poster_ratio_7d', 'u_click_rec_poster_ratio_3d', 'u_click_poster_ratio_3d',
        'u_user_last_login_city_level', 'u_view_post_cnt_7d', 'u_view_rec_post_cnt_7d', 'u_view_post_cnt_3d', 'u_view_rec_post_cnt_3d',
        'u_log_days_cnt_7d', 'u_log_days_cnt_3d', 'u_related_ucnt', 'i_comment_ratio_14d', 'i_click_ratio_14d', 'i_click_ratio_3d',
        'i_comment_ratio_3d', 'i_diff', 'i_v_publisher_click_cnt_3d', 'i_v_publisher_click_cnt_7d', 'i_view_cnt_14d', 'i_view_cnt_3d',
        'i_type_id', 'p_gender', 'p_age_group', 'p_platform', 'p_social_friend_cnt_his', 'p_social_male_friend_cnt_his', 'p_tg_channel_type',
        'p_avg_view_click_ratio_28d', 'p_reply_comment_ratio_28d', 'p_max_view_click_ratio_28d', 'p_user_last_login_city_level',
        'p_publish_post_cnt_28d', 'p_viewed_post_cnt_28d', 'p_fan_cnt', 'p_male_fan_cnt',
    ],
    'xgb_v3': [
        'u_gender', 'u_age_group', 'u_platform', 'u_social_friend_cnt', 'u_social_male_friend_cnt', 'u_tg_channel_type',
        'u_click_rec_poster_ratio_7d', 'u_click_poster_ratio_7d', 'u_click_rec_poster_ratio_3d', 'u_click_poster_ratio_3d',
        'u_user_last_login_city_level', 'u_view_post_cnt_7d', 'u_view_rec_post_cnt_7d', 'u_view_post_cnt_3d',
        'u_view_rec_post_cnt_3d', 'u_log_days_cnt_7d', 'u_log_days_cnt_3d', 'u_related_ucnt', 'i_diff', 'i_type_id',
        'i_click_ratio_1d', 'i_click_ratio_3d', 'i_comment_ratio_3d', 'i_comment_ratio_1d', 'i_v_publisher_click_cnt_3d',
        'i_v_publisher_click_cnt_7d', 'i_view_cnt_1d', 'i_view_cnt_3d', 'p_gender', 'p_age_group', 'p_platform', 'p_social_friend_cnt_his',
        'p_social_male_friend_cnt_his', 'p_tg_channel_type', 'p_avg_view_click_ratio_28d', 'p_reply_comment_ratio_28d',
        'p_max_view_click_ratio_28d', 'p_user_last_login_city_level', 'p_publish_post_cnt_28d', 'p_viewed_post_cnt_28d', 'p_fan_cnt',
        'p_male_fan_cnt', 'u_post_type_prefer', 'is_same_province', 'is_same_city', 'age_diffs', 'recall_type', 'u_p_view_p_cnt_1d',
        'u_p_click_p_cnt_1d', 'u_p_thump_up_p_cnt_1d', 'u_p_view_time_1d', 'u_p_view_p_cnt_3d', 'u_p_click_p_cnt_3d',
        'u_p_thump_up_p_cnt_3d', 'u_p_view_time_3d',
    ],
    'xgb_v5': [
        'u_gender', 'u_age_group', 'u_platform', 'u_social_friend_cnt', 'u_social_male_friend_cnt', 'u_tg_channel_type',
        'u_click_rec_poster_ratio_7d', 'u_click_poster_ratio_7d', 'u_click_rec_poster_ratio_3d', 'u_click_poster_ratio_3d',
        'u_user_last_login_city_level', 'u_view_post_cnt_7d', 'u_view_rec_post_cnt_7d', 'u_view_post_cnt_3d',
        'u_view_rec_post_cnt_3d', 'u_log_days_cnt_7d', 'u_log_days_cnt_3d', 'u_related_ucnt', 'i_diff', 'i_type_id',
        'i_click_ratio_1d', 'i_click_ratio_3d', 'i_comment_ratio_3d', 'i_comment_ratio_1d', 'i_v_publisher_click_cnt_3d',
        'i_v_publisher_click_cnt_7d', 'i_view_cnt_1d', 'i_view_cnt_3d', 'p_gender', 'p_age_group', 'p_platform', 'p_social_friend_cnt_his',
        'p_social_male_friend_cnt_his', 'p_tg_channel_type', 'p_avg_view_click_ratio_28d', 'p_reply_comment_ratio_28d',
        'p_max_view_click_ratio_28d', 'p_user_last_login_city_level', 'p_publish_post_cnt_28d', 'p_viewed_post_cnt_28d', 'p_fan_cnt',
        'p_male_fan_cnt', 'u_post_type_prefer', 'is_same_province', 'is_same_city', 'age_diffs', 'recall_type', 'u_p_view_p_cnt_1d',
        'u_p_click_p_cnt_1d', 'u_p_thump_up_p_cnt_1d', 'u_p_view_time_1d', 'u_p_view_p_cnt_3d', 'u_p_click_p_cnt_3d',
        'u_p_thump_up_p_cnt_3d', 'u_p_view_time_3d', 'u_ctr_p_male_14d', 'u_ctr_p_female_14d', 'u_ctr_odd_p_female_14d',
        'u_ctr_p_male_7d', 'u_ctr_p_female_7d', 'u_ctr_odd_p_female_7d', 'u_ctr_p_male_3d', 'u_ctr_p_female_3d', 'u_ctr_odd_p_female_3d',
    ],
    'nn_v1': [
        'u_gender', 'u_age_group', 'u_platform', 'u_user_last_login_city_level',
        'u_social_friend_cnt', 'u_social_male_friend_cnt',
        'u_click_rec_poster_ratio_7d', 'u_click_poster_ratio_7d', 'u_click_rec_poster_ratio_3d', 'u_click_poster_ratio_3d',
        'u_view_post_cnt_7d', 'u_view_rec_post_cnt_7d', 'u_view_post_cnt_3d', 'u_view_rec_post_cnt_3d',
        'u_log_days_cnt_7d', 'u_log_days_cnt_3d', 'u_related_ucnt', 'i_comment_ratio_14d', 'i_click_ratio_14d', 'i_click_ratio_3d',
        'i_comment_ratio_3d', 'i_diff', 'i_v_publisher_click_cnt_3d', 'i_v_publisher_click_cnt_7d',

        'i_type_id', 'i_view_cnt_14d', 'i_view_cnt_3d',

        'p_gender', 'p_age_group', 'p_platform', 'p_user_last_login_city_level',
        'p_social_friend_cnt_his', 'p_social_male_friend_cnt_his', 'p_fan_cnt', 'p_male_fan_cnt',
        'p_avg_view_click_ratio_28d', 'p_reply_comment_ratio_28d', 'p_max_view_click_ratio_28d',
        'p_publish_post_cnt_28d', 'p_viewed_post_cnt_28d',
    ]
}
feature_columns_versions = {
    version:features2columns(features, ver=version, warning=False) for version, features in feature_versions.items()
}


if __name__ == '__main__':
    feature_columns = ALL_FEATURE_COLUMNS

    FEATURES = [x.name for x in feature_columns]
    LABELS = ['label']
    OTHERS = ['dt', 'min_hr', 'min_recall_type', 'user_id']
    ALL_COLUMNS = FEATURES + LABELS + OTHERS

    sparse_features = [fc for fc in feature_columns if isinstance(fc, SparseFeature)]
    dense_features = [fc for fc in feature_columns if isinstance(fc, DenseFeature)]
    print(f'len(sparse_features) = {len(sparse_features)}, len(dense_features) = {len(dense_features)}')
    print(f'len(FEATURES) = {len(FEATURES)}, len(ALL_COLUMNS) = {len(ALL_COLUMNS)}')




# u_is_new, u_platform, u_gender, u_age, u_age_group, u_media_name, u_user_last_login_city_id, u_user_last_login_city_level,
# u_user_last_login_province,
# u_social_friend_cnt, u_social_male_friend_cnt, u_fan_cnt, u_male_fan_cnt, u_follow_cnt, u_female_follow_cnt, u_related_ucnt,
# u_post_type_click_ratio, u_log_days_cnt_{14/7/3}d, u_view_post_cnt_{14/7/3/1}d, u_{click/comment}_poster_ratio_{14/7/3/1}d,
# u_view_rec_post_cnt_{14/7/3/1}d, u_{click/comment}_rec_poster_ratio_{14/7/3/1}d, p_favorite_post_type_id,
#
# i_type_id, i_std_platform, i_diff, i_view_cnt_{14/7/3/1}d, i_{click/comment}_ratio_{14/7/3/1}d, i_v_publisher_click_cnt_3d,
# i_avg_stay_time_{14/7/3/1}d, i_valid_view_ratio_{14/7/3/1}d,
#
# p_is_new, p_platform, p_gender, p_age, p_age_group, p_media_name, p_user_last_login_province,
# p_user_last_login_city_id, p_user_last_login_city_level, p_social_friend_cnt_his, p_social_male_friend_cnt_his,
# p_fan_cnt, p_male_fan_cnt, p_follow_cnt, p_female_follow_cnt, p_related_ucnt,
# p_publish_post_cnt_{28/14/7}d, p_viewed_post_cnt_{28/14/7}d, p_personal_page_view_ucnt_{28/14/7/3/1}d,
# p_comment_cnt_sum_28d, p_reply_comment_ratio_28d, p_choice_post_cnt_28d, p_avg_view_{click/comment}_ratio_28d,
# p_max_view_{click/comment}_ratio_28d, p_active_im_ucnt_{14/7/3/1}d, p_active_im_reply_ratio_{14/7/3/1}d,
# p_passive_im_ucnt_{14/7/3/1}d, p_passive_im_reply_ratio_{14/7/3/1}d, u_avg_view_time_{14/7/3/1}d, u_rec_avg_view_time_{14/7/3/1}d,
# u_valid_view_post_ratio_{14/7/3/1}d, u_valid_view_rec_post_ratio_{14/7/3/1}d,
# p_valid_viewed_post_ratio_{28/14/7}d,
#
# {min/max}_client_timestamp,{min/max}_hr,{min/max}_position_id,{min/max}_recall_type
# clicked, click_cnt,thump_up, enter_personal_page_cnt, comment_cnt,chitchat_cnt, fst_comment_cnt, sec_comment_cnt, comment_up_cnt,
# share_cnt, attention_cnt, collect_cnt, stay_time, label

# 以后分析：user_id, u_reg_time, u_birthday, u_is_gameprint, u_raw_cpid, u_cpid, u_cmid, u_click_time, u_agent_name,
#               u_tag_social, u_tag_gamecard_list, u_tag_personality_list, u_active_hour, u_enter_room_cnt, u_create_room_cnt,
#               u_active_im_ucnt_{14/7/3/1}d, u_active_im_reply_ratio_{14/7/3/1}d,
#               u_passive_im_ucnt_{14/7/3/1}d, u_passive_im_reply_ratio_{14/7/3/1}d,
#           post_id, i_publish_time, i_create_time, i_is_robot, i_topic_id, i_tag_name, i_origin_id, i_location_id, i_content
#           publisher_uid, p_is_new, p_reg_time, p_nickname, p_birthday, p_is_gameprint, p_room_id, p_is_black_user,
#               p_raw_cpid, p_cpid, p_cmid, p_click_time, p_agent_name, p_user_last_login_city_id, p_user_last_login_city_level
#               p_tag_social, p_tag_gamecard_list, p_tag_personality_list, p_game_card, p_active_hour, p_kol, p_kol_level,
#               p_enter_room_cnt, p_create_room_cnt,
# 疑问：u_ttid, u_room_id, u_is_black_user, u_is_ban_user, u_channel_flag, u_tg_channel_type, u_tg_user_name, u_tf_content_name,
#       p_ttid, p_channel_flag, p_tg_channel_type, p_tg_user_name, p_tf_content_name,
#
# 丢弃：u_platform_name, u_app_ver, u_nickname, u_user_last_login_city, u_device_id, u_game_card, u_app_id,
#       i_type_name, i_topic_name, i_origin_name, i_std_platform_name, i_app_id,
#       p_app_ver, p_platform_name, p_app_id， p_user_last_login_city, p_device_id

