# -*- coding: utf-8 -*-

from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool

_df4gauc = None


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 5)
    return group_auc


def _cal_auc_len(st, ed):
    _len_list, _auc_list = list(), list()
    for row in _df4gauc.iloc[st:ed].itertuples(index=False):
        _len_list.append(len(row[0]))
        _auc_list.append(roc_auc_score(row[0], row[1]))  # label, pred
    return _len_list, _auc_list


def _parallel_cal_group_auc(labels, preds, user_id_list):
    """ Calculate group auc """
    if len(user_id_list) != len(labels):
        raise ValueError("impression id num should equal to the sample num," \
                         "impression id num is {0}".format(len(user_id_list)))

    global _df4gauc
    _df4gauc = pd.DataFrame.from_dict({'label': labels, 'pred': preds, 'user_id': user_id_list})
    _df4gauc = _df4gauc[_df4gauc['user_id'].duplicated(keep=False)].groupby('user_id').agg(
        {'label': [tuple, 'sum', 'count'], 'pred': tuple})
    _df4gauc = _df4gauc.loc[
        (_df4gauc[('label', 'sum')] > 0) & (_df4gauc[('label', 'sum')] != _df4gauc[('label', 'count')])]
    _df4gauc.columns = ['label', 'drop1', 'drop2', 'pred']
    _df4gauc.drop(['drop1', 'drop2'], axis=1, inplace=True)

    weighted_sum, total_sum = 0, 0
    n_workers = cpu_count() - 1
    with Pool(processes=n_workers) as P:
        indexes = list(range(0, len(_df4gauc), len(_df4gauc) // n_workers))[:n_workers] + [len(_df4gauc)]

        async_results = list()
        for i in range(n_workers): async_results.append(P.apply_async(_cal_auc_len, (indexes[i], indexes[i + 1])))
        P.close()
        P.join()

        for _lens, _aucs in (res.get() for res in async_results):
            weighted_sum += np.array(_aucs).dot(np.array(_lens))
            total_sum += sum(_lens)
    _df4gauc = None
    return round(float(weighted_sum / total_sum), 6)


# 并行计算gauc
def parallel_cal_group_auc(labels, preds, user_id_list):
    """ Calculate group auc """
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(preds, list):
        preds = np.array(preds)

    if hasattr(labels, 'shape') and len(labels.shape) > 1: labels = np.reshape(labels, [-1])
    if hasattr(preds, 'shape') and len(preds.shape) > 1: preds = np.reshape(preds, [-1])

    if len(user_id_list) != len(labels):
        raise ValueError("impression id num should equal to the sample num," \
                         "impression id num is {0}".format(len(user_id_list)))
    if len(user_id_list) >= 100000:
        try:
            return _parallel_cal_group_auc(labels, preds, user_id_list)
        except Exception as e:
            print("WARN: cant parallelize")
            print(e)

    group_truth, group_score = defaultdict(list), defaultdict(list)
    for idx, user_id in enumerate(user_id_list):
        group_truth[user_id].append(labels[idx])
        group_score[user_id].append(preds[idx])

    total_auc, impression_total = 0.0, 0
    for user_id, this_group_truth in group_truth.items():
        if this_group_truth and len(set(np.asarray(this_group_truth))) > 1:
            auc = roc_auc_score(np.asarray(this_group_truth), np.asarray(group_score[user_id]))
            total_auc += auc * len(this_group_truth)
            impression_total += len(this_group_truth)
    return round(float(total_auc) / impression_total, 6)


if __name__ == '__main__':
    # 单元测试只是验证user id 一样的情况下，是否跟auc一样

    y_pred = [
        0.09315896034240723, 0.12490281462669373, 0.11221307516098022, 0.19415467977523804, 0.0419694185256958,
        0.12626835703849792, 0.17528077960014343, 0.08039942383766174, 0.04866880178451538, 0.1617153286933899,
        0.21216043829917908, 0.1219533383846283, 0.0609055757522583, 0.05872851610183716, 0.18362346291542053,
        0.03574401140213013, 0.10407242178916931, 0.15018713474273682, 0.10158365964889526, 0.07824292778968811,
        0.1044837236404419, 0.03757771849632263, 0.11979082226753235, 0.02091875672340393, 0.07370883226394653,
        0.07832187414169312, 0.15056690573692322, 0.02013951539993286, 0.0428619384765625, 0.09616050124168396,
        0.07980287075042725, 0.017647773027420044, 0.728003740310669, 0.33676373958587646, 0.05323329567909241,
        0.33572036027908325, 0.1456538438796997, 0.08585020899772644, 0.021872758865356445, 0.3323047161102295,
        0.14452910423278809, 0.10015317797660828, 0.32870471477508545, 0.058025360107421875, 0.03147372603416443,
        0.049002230167388916, 0.33274340629577637, 0.10543730854988098, 0.2666059136390686, 0.02436751127243042,
        0.06509396433830261, 0.0573955774307251, 0.03739893436431885, 0.10526889562606812, 0.08665534853935242,
        0.053078681230545044, 0.034448057413101196, 0.08272099494934082, 0.00829470157623291, 0.17825818061828613,
        0.03018563985824585, 0.042617738246917725, 0.0412060022354126, 0.11065617203712463, 0.03628072142601013,
        0.07860362529754639, 0.03565672039985657, 0.2563965916633606, 0.31039825081825256, 0.14852997660636902,
        0.310992956161499, 0.048822641372680664, 0.5012142658233643, 0.05741897225379944, 0.5452510118484497,
        0.09434705972671509, 0.3471984267234802, 0.13391819596290588, 0.01376497745513916, 0.06947019696235657,
        0.05096283555030823, 0.11196863651275635, 0.10468575358390808, 0.03975829482078552, 0.08219578862190247,
        0.021490484476089478, 0.054100990295410156, 0.19808629155158997, 0.03799799084663391, 0.11710631847381592,
        0.09993401169776917, 0.12421858310699463, 0.3556922674179077, 0.05815902352333069, 0.3889700770378113,
        0.1024864912033081, 0.03083813190460205, 0.1097012460231781, 0.07143574953079224, 0.0380517840385437,
        0.17310452461242676, 0.07170730829238892, 0.06782540678977966, 0.09334799647331238, 0.0229053795337677,
        0.4178757071495056, 0.3843355178833008, 0.07960736751556396, 0.037320107221603394, 0.11303776502609253,
        0.01938420534133911, 0.014246910810470581, 0.04398199915885925, 0.07310786843299866, 0.16961991786956787,
        0.21252167224884033, 0.0261993408203125, 0.0672747790813446, 0.06077665090560913, 0.05133354663848877,
        0.39845412969589233, 0.12647685408592224, 0.04219260811805725, 0.14048010110855103, 0.25854259729385376,
        0.26810070872306824, 0.1083713173866272, 0.1452491283416748, 0.09804049134254456, 0.024198263883590698,
        0.15427875518798828, 0.02130812406539917, 0.023791790008544922, 0.40602385997772217, 0.045842647552490234,
        0.03870353102684021, 0.04131332039833069, 0.22315645217895508, 0.04205921292304993, 0.022297710180282593,
        0.1515914797782898, 0.020539522171020508, 0.2938770055770874, 0.07053086161613464, 0.42396003007888794,
        0.07368239760398865, 0.062224507331848145, 0.10869163274765015, 0.104286789894104, 0.11505186557769775,
        0.04332664608955383, 0.04293850064277649, 0.07254195213317871, 0.02478930354118347, 0.04191237688064575,
        0.031360119581222534, 0.13027963042259216, 0.3391348123550415, 0.1105160117149353, 0.45123806595802307,
        0.017431676387786865, 0.22518396377563477, 0.12912365794181824, 0.08543810248374939, 0.21663886308670044,
        0.27373769879341125, 0.04655948281288147, 0.11145070195198059, 0.09180060029029846, 0.02538171410560608,
        0.20664548873901367, 0.05106163024902344, 0.1997058391571045, 0.014097005128860474, 0.041274696588516235,
        0.17285513877868652, 0.029011428356170654, 0.23312479257583618, 0.20327505469322205, 0.4578773081302643,
        0.04336729645729065, 0.163496732711792, 0.02882292866706848, 0.08272948861122131, 0.08256059885025024,
        0.03659126162528992, 0.03176999092102051, 0.13102129101753235, 0.02018454670906067, 0.019100069999694824,
        0.031184017658233643, 0.0671597421169281, 0.34097588062286377, 0.019394397735595703, 0.06385248899459839,
        0.026644796133041382, 0.01879405975341797, 0.11601418256759644, 0.18078288435935974, 0.005475908517837524,
        0.07376644015312195, 0.058872729539871216, 0.053471505641937256, 0.12069115042686462, 0.027772098779678345,
        0.0281522274017334, 0.036479830741882324, 0.17244255542755127, 0.10503405332565308, 0.040000081062316895,
        0.06418293714523315, 0.07806980609893799, 0.05498501658439636, 0.04847201704978943, 0.08448508381843567,
        0.20448344945907593, 0.13106653094291687, 0.07255396246910095, 0.34139710664749146, 0.10162389278411865,
        0.14186739921569824, 0.0360734760761261, 0.02508804202079773, 0.008325368165969849, 0.10612133145332336,
        0.030629754066467285, 0.2712938189506531, 0.22198766469955444, 0.017312854528427124, 0.08255508542060852,
        0.1421220600605011, 0.03519216179847717, 0.3711695969104767, 0.14451825618743896, 0.0576956570148468,
        0.04485899209976196, 0.07118123769760132, 0.08619850873947144, 0.08149093389511108, 0.05872383713722229,
        0.1175336241722107, 0.10910636186599731, 0.17398828268051147, 0.051576822996139526, 0.2035215198993683,
        0.08882614970207214, 0.013032019138336182, 0.09882336854934692, 0.21397793292999268, 0.158960223197937,
        0.29391923546791077, 0.06217530369758606, 0.04112255573272705, 0.03889775276184082, 0.01027563214302063,
        0.04201361536979675, 0.04901215434074402, 0.19108247756958008, 0.19621646404266357, 0.059261053800582886,
        0.01856285333633423, 0.014767467975616455, 0.05037647485733032, 0.2124149203300476, 0.020989954471588135,
        0.10288858413696289, 0.08144629001617432, 0.12580490112304688, 0.049946993589401245, 0.2172626554965973,
        0.016939431428909302, 0.02323758602142334, 0.043867141008377075, 0.04414305090904236, 0.12476161122322083,
        0.27574124932289124, 0.13628548383712769, 0.030571430921554565, 0.043754398822784424, 0.14803752303123474,
        0.11102950572967529, 0.020368993282318115, 0.10188350081443787, 0.014186948537826538, 0.09252774715423584,
        0.03957796096801758, 0.06903919577598572, 0.03736865520477295, 0.19056078791618347, 0.0329684317111969,
        0.16132193803787231, 0.2964426875114441, 0.08419609069824219, 0.04146847128868103, 0.22425615787506104,
        0.09347894787788391, 0.1026751697063446, 0.09772053360939026, 0.13194039463996887, 0.10949036478996277,
        0.1003747284412384, 0.4137239158153534, 0.05083689093589783, 0.030933111906051636, 0.17922893166542053,
        0.06168085336685181, 0.6152852773666382, 0.08095639944076538, 0.125868558883667, 0.04482269287109375,
        0.19132491946220398, 0.11587974429130554, 0.04941245913505554, 0.060558855533599854, 0.1383262574672699,
        0.35095536708831787, 0.08137720823287964, 0.03390359878540039, 0.035154879093170166, 0.11303785443305969,
        0.22574618458747864, 0.08505362272262573, 0.022188544273376465, 0.2038358449935913, 0.021757185459136963,
        0.01425120234489441, 0.2924729585647583, 0.022311359643936157, 0.45446813106536865, 0.025530725717544556,
        0.17640942335128784, 0.3803442120552063, 0.07653751969337463, 0.02653980255126953, 0.03471583127975464,
        0.03251582384109497, 0.07914993166923523, 0.040933579206466675, 0.05456683039665222, 0.03605964779853821,
        0.12290331721305847, 0.1599070131778717, 0.03412961959838867, 0.1522689163684845, 0.10768118500709534,
        0.01851290464401245, 0.12107565999031067, 0.16344159841537476, 0.05794167518615723, 0.051277995109558105,
        0.1602592170238495, 0.21965819597244263, 0.03972572088241577, 0.03284502029418945, 0.04214408993721008,
        0.12692254781723022, 0.20529186725616455, 0.06370961666107178, 0.16954919695854187, 0.06653270125389099,
        0.06013152003288269, 0.09853735566139221, 0.08317044377326965, 0.11168467998504639, 0.035727083683013916,
        0.15627598762512207, 0.20520350337028503, 0.11707055568695068, 0.0345531702041626, 0.021687597036361694,
        0.0858355462551117, 0.07526406645774841, 0.09426391124725342, 0.040329545736312866, 0.1363716721534729,
        0.12612327933311462, 0.1388920545578003, 0.027020633220672607, 0.12303677201271057, 0.1717625856399536,
        0.29655468463897705, 0.029080629348754883, 0.12745487689971924, 0.2382175624370575, 0.12143570184707642,
        0.07918092608451843, 0.18801361322402954, 0.5047825574874878, 0.06380897760391235, 0.025305509567260742,
        0.06689617037773132, 0.39729535579681396, 0.07319203019142151, 0.07211479544639587, 0.0746118426322937,
        0.2439441978931427, 0.1379653513431549, 0.08074238896369934, 0.1349763572216034, 0.14945480227470398,
        0.005453169345855713, 0.19957643747329712, 0.3501351773738861, 0.03322100639343262, 0.18682995438575745,
        0.2798541784286499, 0.02958357334136963, 0.04344549775123596, 0.21264496445655823, 0.03303876519203186,
        0.020328104496002197, 0.11280748248100281, 0.2889729142189026, 0.04670155048370361, 0.06174999475479126,
        0.11747616529464722, 0.11150205135345459, 0.2822878062725067, 0.04431813955307007, 0.29289573431015015,
        0.32679373025894165, 0.3223807215690613, 0.02922344207763672, 0.08531844615936279, 0.11818721890449524,
        0.01914113759994507, 0.04917752742767334, 0.05990105867385864, 0.060692548751831055, 0.22156554460525513,
        0.07008504867553711, 0.17427867650985718, 0.047612130641937256, 0.12106022238731384, 0.045282065868377686,
        0.036345839500427246, 0.19596543908119202, 0.07288745045661926, 0.14858776330947876, 0.16519761085510254,
        0.2521715760231018, 0.0254686176776886, 0.05470472574234009, 0.09650164842605591, 0.08878788352012634,
        0.03748667240142822, 0.051008015871047974, 0.05443194508552551, 0.04241684079170227, 0.04374241828918457,
        0.28452712297439575, 0.08765441179275513, 0.26053932309150696, 0.04143276810646057, 0.07654684782028198,
        0.1489804983139038, 0.1029861569404602, 0.31045809388160706, 0.4292628765106201, 0.06493908166885376,
        0.3781900107860565, 0.041293948888778687, 0.2822532653808594, 0.06525599956512451, 0.09398779273033142,
        0.06841090321540833, 0.06418189406394958, 0.05361735820770264, 0.09576413035392761, 0.2915286421775818,
        0.09380581974983215, 0.06717193126678467, 0.08428493142127991, 0.021690458059310913, 0.014697670936584473,
        0.48303574323654175, 0.010870516300201416, 0.02788713574409485, 0.060061484575271606, 0.11098220944404602,
        0.009795695543289185, 0.14355504512786865, 0.03025028109550476, 0.05447790026664734, 0.09215033054351807,
        0.10135716199874878, 0.3639208674430847, 0.46160608530044556, 0.5531855821609497, 0.07329386472702026,
        0.08213689923286438, 0.01976493000984192, 0.01322484016418457, 0.03876376152038574, 0.02648940682411194,
        0.0661439299583435, 0.08544868230819702, 0.11020404100418091, 0.13066565990447998, 0.16958671808242798,
        0.10875752568244934, 0.33334988355636597, 0.03411400318145752, 0.1872604489326477, 0.022700875997543335,
        0.4406962990760803, 0.04029130935668945, 0.026442408561706543, 0.023670613765716553, 0.03541988134384155,
        0.17074644565582275, 0.023440837860107422, 0.03629094362258911, 0.11555233597755432, 0.07105758786201477,
        0.3103915750980377, 0.023726344108581543, 0.11569154262542725, 0.008542239665985107, 0.027744382619857788,
        0.05080804228782654, 0.17309987545013428, 0.08375418186187744, 0.018756389617919922, 0.08050331473350525,
        0.06287828087806702, 0.13412249088287354, 0.06139957904815674, 0.1536692976951599, 0.17818614840507507,
        0.10149109363555908, 0.04875916242599487, 0.08720019459724426, 0.04030376672744751, 0.32001399993896484,
        0.03328442573547363, 0.24633798003196716, 0.13899412751197815, 0.15630239248275757, 0.09330517053604126,
        0.023574888706207275, 0.12918078899383545, 0.05144762992858887, 0.041751205921173096, 0.1123565137386322,
        0.08119797706604004, 0.2558767795562744, 0.10465598106384277, 0.029753148555755615, 0.3883990943431854,
        0.057108551263809204, 0.09070733189582825, 0.16014626622200012, 0.10476714372634888, 0.09281015396118164,
        0.054875582456588745, 0.02888774871826172, 0.04552358388900757, 0.015549838542938232, 0.05297571420669556,
        0.06667596101760864, 0.13174837827682495, 0.053059667348861694, 0.008244991302490234, 0.034251868724823,
        0.08305436372756958, 0.029166847467422485, 0.059966325759887695, 0.3981354236602783, 0.08869081735610962,
        0.1394660770893097, 0.05237942934036255, 0.10496893525123596, 0.048463016748428345, 0.15480318665504456,
        0.2652115225791931, 0.05913981795310974, 0.05809482932090759, 0.03696879744529724, 0.07648435235023499,
        0.06303280591964722, 0.33588939905166626, 0.31156444549560547, 0.016319245100021362, 0.2983359694480896,
        0.03563809394836426, 0.2045241892337799, 0.06060028076171875, 0.1530900001525879, 0.034526705741882324,
        0.041601717472076416, 0.03479751944541931, 0.04026079177856445, 0.2782175540924072, 0.09459742903709412,
        0.3729952573776245, 0.028053313493728638, 0.11024698615074158, 0.2272110879421234, 0.02640455961227417,
        0.05263516306877136, 0.07031092047691345, 0.03220829367637634, 0.0663875937461853, 0.06179201602935791,
        0.30496150255203247, 0.06726345419883728, 0.05104970932006836, 0.4573562741279602, 0.12079176306724548,
        0.03144711256027222, 0.0030919313430786133, 0.041982054710388184, 0.14066442847251892, 0.027859598398208618,
        0.024911433458328247, 0.27355843782424927, 0.04418262839317322, 0.12236326932907104, 0.15400445461273193,
        0.19532766938209534, 0.04125851392745972, 0.03322985768318176, 0.06050032377243042, 0.04797449707984924,
        0.10188013315200806, 0.02661311626434326, 0.042188167572021484, 0.10836797952651978, 0.03513398766517639,
        0.12300679087638855, 0.20660513639450073, 0.2193552553653717, 0.404199481010437, 0.11950421333312988,
        0.051907867193222046, 0.14247936010360718, 0.06009688973426819, 0.09795185923576355, 0.1792689859867096,
        0.1351146399974823, 0.26226720213890076, 0.1333751678466797, 0.07534894347190857, 0.0343022346496582,
        0.2243090569972992, 0.24441903829574585, 0.10473716259002686, 0.02699485421180725, 0.060216039419174194,
        0.0709230899810791, 0.10930219292640686, 0.030017197132110596, 0.15300825238227844, 0.08892539143562317,
        0.17472228407859802, 0.12967798113822937, 0.04034826159477234, 0.06375622749328613, 0.016570329666137695,
        0.04777845740318298, 0.03182128071784973, 0.03491848707199097, 0.17455655336380005, 0.10786125063896179,
        0.06492799520492554, 0.10107776522636414, 0.0295695960521698, 0.06729549169540405, 0.039513856172561646,
        0.04965311288833618, 0.03512251377105713, 0.30647286772727966, 0.31620529294013977, 0.1515924632549286,
        0.019261687994003296, 0.20026171207427979, 0.14718589186668396, 0.28016406297683716, 0.03619244694709778,
        0.03258123993873596, 0.11179885268211365, 0.12621286511421204, 0.07235440611839294, 0.09138721227645874,
        0.07925128936767578, 0.007509768009185791, 0.061692237854003906, 0.11026811599731445, 0.0955309271812439,
        0.19075199961662292, 0.46118631958961487, 0.006866008043289185, 0.182137131690979, 0.06533893942832947,
        0.09895268082618713, 0.01668936014175415, 0.08396059274673462, 0.142553448677063, 0.1999565064907074,
        0.5770849585533142, 0.10462737083435059, 0.11570999026298523, 0.18560564517974854, 0.05579379200935364,
        0.01801815629005432, 0.09985107183456421, 0.16119882464408875, 0.1657351553440094, 0.09527280926704407,
        0.0935506820678711, 0.018661588430404663, 0.06152549386024475, 0.039426177740097046, 0.10038864612579346,
        0.10044515132904053, 0.03020414710044861, 0.20471441745758057, 0.5050371289253235, 0.03436684608459473,
        0.11605656147003174, 0.019187062978744507, 0.0679674744606018, 0.16794803738594055, 0.01322057843208313,
        0.14018678665161133, 0.015832364559173584, 0.02924153208732605, 0.06623956561088562, 0.039527952671051025,
        0.05094084143638611, 0.23424053192138672, 0.0744490921497345, 0.046950697898864746, 0.07591232657432556,
        0.03706720471382141, 0.04736125469207764, 0.06324699521064758, 0.1374143362045288, 0.02473190426826477,
        0.43334734439849854, 0.03159502148628235, 0.08871981501579285, 0.08806866407394409, 0.12311509251594543,
        0.06398865580558777, 0.01825803518295288, 0.7774355411529541, 0.07535529136657715, 0.04467964172363281,
        0.09308764338493347, 0.03290078043937683, 0.03320795297622681, 0.07532462477684021, 0.24808591604232788,
        0.037443071603775024, 0.12758079171180725, 0.018985092639923096, 0.007859617471694946, 0.16902413964271545,
        0.09315037727355957, 0.09406024217605591, 0.2700338065624237, 0.09036645293235779, 0.5100675821304321,
        0.23528248071670532, 0.03389334678649902, 0.402457594871521, 0.046937912702560425, 0.036443084478378296,
        0.19374608993530273, 0.07527473568916321, 0.023050397634506226, 0.002006620168685913, 0.01605314016342163,
        0.3794311285018921, 0.06564787030220032, 0.24055787920951843, 0.18604305386543274, 0.2736677825450897,
        0.3647412657737732, 0.05929386615753174, 0.01872539520263672, 0.2314319610595703, 0.4110865890979767,
        0.04060128331184387, 0.08266064524650574, 0.14100602269172668, 0.4833398461341858, 0.03013324737548828,
        0.4510098099708557, 0.11597788333892822, 0.017244458198547363, 0.017553657293319702, 0.14907178282737732,
        0.04295390844345093, 0.06660860776901245, 0.22991898655891418, 0.10992926359176636, 0.14813673496246338,
        0.4002528190612793, 0.12288889288902283, 0.07660156488418579, 0.14162391424179077, 0.09451150894165039,
        0.42963263392448425, 0.23691004514694214, 0.22713598608970642, 0.031801074743270874, 0.03751799464225769,
        0.2999134659767151, 0.06260618567466736, 0.05831176042556763, 0.16224408149719238, 0.0060076117515563965,
        0.13972455263137817, 0.08220908045768738, 0.12252509593963623, 0.19853639602661133, 0.0399324893951416,
        0.06839737296104431, 0.16653800010681152, 0.04609134793281555, 0.03437802195549011, 0.11378705501556396,
        0.5284417867660522, 0.33410748839378357, 0.18578815460205078, 0.005784958600997925, 0.09956711530685425,
        0.02384677529335022, 0.2485087811946869, 0.06465005874633789, 0.0226747989654541, 0.0742744505405426,
        0.13152092695236206, 0.029460102319717407, 0.03709772229194641, 0.044764935970306396, 0.1387844979763031,
        0.06834599375724792, 0.0759967565536499, 0.043072789907455444, 0.1964310109615326, 0.0935889184474945,
        0.01911604404449463, 0.05914062261581421, 0.036137938499450684, 0.07759320735931396, 0.1475946605205536,
        0.12499254941940308, 0.1906508505344391, 0.1923719346523285, 0.05605432391166687, 0.1896870732307434,
        0.15316417813301086, 0.02163878083229065, 0.03953924775123596, 0.06462469696998596, 0.12510448694229126,
        0.05007848143577576, 0.1038849949836731, 0.02978670597076416, 0.04216626286506653, 0.11211895942687988,
        0.01633802056312561, 0.29378247261047363, 0.06770974397659302, 0.13133791089057922, 0.06958615779876709,
        0.04661795496940613, 0.052022457122802734, 0.03935331106185913, 0.4368076026439667, 0.07143869996070862,
        0.024435341358184814, 0.07856076955795288, 0.012479335069656372, 0.22522321343421936, 0.014615744352340698,
        0.12540924549102783, 0.0330159068107605, 0.11635982990264893, 0.08215337991714478, 0.029590308666229248,
        0.061932146549224854, 0.13729023933410645, 0.05965784192085266, 0.10948508977890015, 0.10730850696563721,
        0.08767110109329224, 0.027631789445877075, 0.02251383662223816, 0.21100392937660217, 0.05479389429092407,
        0.06100955605506897, 0.5808190703392029, 0.24673965573310852, 0.05272877216339111, 0.015547096729278564,
        0.049039095640182495, 0.018476516008377075, 0.02998712658882141, 0.07687008380889893, 0.028773725032806396,
        0.10421919822692871, 0.025012582540512085, 0.022611886262893677, 0.09269052743911743, 0.10829445719718933,
        0.09803459048271179, 0.05517420172691345, 0.1448323130607605, 0.11824542284011841, 0.04030385613441467,
        0.016258955001831055, 0.0515400767326355, 0.11325722932815552, 0.007433056831359863, 0.08921676874160767,
        0.015840083360671997, 0.04918798804283142, 0.0836317241191864, 0.09849369525909424, 0.04079997539520264,
        0.025753170251846313, 0.015715867280960083, 0.02635335922241211, 0.09368175268173218, 0.008910268545150757,
        0.15956035256385803, 0.24033096432685852, 0.027231693267822266, 0.035634756088256836, 0.1788312792778015,
        0.15189412236213684, 0.16572102904319763, 0.01968863606452942, 0.053184181451797485, 0.032855987548828125,
        0.13717401027679443, 0.08045977354049683, 0.0835639238357544, 0.09996744990348816, 0.14712977409362793,
        0.2135409116744995, 0.10238569974899292, 0.030599087476730347, 0.04069709777832031, 0.07273644208908081,
        0.39814329147338867, 0.05966669321060181, 0.1881677806377411, 0.2186327576637268, 0.030352532863616943,
        0.04220014810562134, 0.08685413002967834, 0.050397634506225586, 0.3772004246711731, 0.1612381935119629,
        0.047620534896850586, 0.03251740336418152, 0.007520943880081177, 0.042664796113967896, 0.00968778133392334,
        0.02796351909637451, 0.05803149938583374, 0.0648130476474762, 0.10147994756698608, 0.26437950134277344,
        0.12199869751930237, 0.03904610872268677, 0.1454269289970398, 0.11943987011909485, 0.09483665227890015,
        0.21066462993621826, 0.10229071974754333, 0.35208404064178467, 0.1632043421268463, 0.1290527880191803,
        0.0145893394947052, 0.25217288732528687, 0.06572183966636658, 0.09870937466621399, 0.09108811616897583,
        0.08988097310066223, 0.1588670313358307, 0.02004873752593994, 0.4510363042354584, 0.03776884078979492,
        0.0184938907623291, 0.011885315179824829, 0.038090646266937256, 0.14032500982284546, 0.06488290429115295,
        0.07617771625518799, 0.11613982915878296, 0.01796245574951172, 0.12648728489875793, 0.29066693782806396,
        0.10044780373573303, 0.010604888200759888, 0.06301751732826233, 0.16089549660682678, 0.2136535942554474,
        0.05245161056518555, 0.03391975164413452, 0.09704649448394775, 0.19904744625091553, 0.022246897220611572,
        0.1898413896560669, 0.01210436224937439, 0.14938995242118835, 0.10329365730285645, 0.033585697412490845,
        0.13588494062423706, 0.06065702438354492, 0.05851975083351135, 0.02731439471244812, 0.23709475994110107,
        0.26772865653038025, 0.2500956058502197, 0.24481338262557983, 0.07593458890914917]
    y_test = [
        1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    user_id_list = ['1'] * len(y_test)

    print(parallel_cal_group_auc(y_test, y_pred, user_id_list))
    print(cal_group_auc(y_test, y_pred, user_id_list))
    print(roc_auc_score(y_test, y_pred))