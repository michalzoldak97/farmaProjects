from math import log, e

class Condition:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, val):
        return val[self.column] >= self.value

    def __repr__(self):
        return "Is {} >= {}".format(self.column, self.value)


class FinalLeaf:

    def __init__(self, ds: list):
        self.predictions = _cls_count(ds)


class Node:

    def __init__(self, con: Condition, t_group, f_group):
        self.condition = con
        self.true_group = t_group
        self.false_group = f_group


def _cls_count(ds: list, num_class=6):
    cnt = [0 for _ in range(num_class)]
    for row in ds:
        cnt[int(row[-1])] += 1
    return cnt


def _calc_entropy(ds: list, num_class=6):
    ds_len = float(len(ds))
    cnt = _cls_count(ds, num_class)
    ent = 1.
    for i in cnt:
        prob = i / ds_len
        try:
            ent -= prob * log(prob, 2)
        except ValueError:
            continue

    return ent


def _calc_gini(ds: list, num_class=6):
    ds_len = float(len(ds))
    cnt = _cls_count(ds, num_class)
    gini = 1.
    for i in cnt:
        prob = i / ds_len
        gini -= prob * prob

    return gini


def _calc_info_gain(left: list, right: list, curr_u: float):
    len_l = len(left)
    p = float(len_l / (len_l + len(right)))
    return curr_u - p * _calc_entropy(left) - (1 - p) * _calc_entropy(right)


def _split(ds, con):
    t_rows, f_rows = [], []
    for row in ds:
        if con.match(row):
            t_rows.append(row)
        else:
            f_rows.append(row)

    return t_rows, f_rows


def _get_best_split(ds: list):
    n_feat = len(ds[0]) - 1
    b_info_gain = 0
    b_con = None
    start_u = _calc_entropy(ds)
    for col in range(n_feat):
        vals = set([row[col] for row in ds])
        for val in vals:
            curr_con = Condition(col, val)
            t_group, f_group = _split(ds, curr_con)

            if len(t_group) == 0 or len(f_group) == 0:
                continue

            info_gain = _calc_info_gain(t_group, f_group, start_u)

            if info_gain > b_info_gain:
                b_info_gain, b_con = info_gain, curr_con

    return b_info_gain, b_con


def build_tree(ds: list, depth: int, max_depth=2, min_size=5):
    info_gain, condition = _get_best_split(ds)

    if info_gain == 0 or depth > max_depth or len(ds) < min_size:
        return FinalLeaf(ds)

    t_group, f_group = _split(ds, condition)

    t_branch = build_tree(t_group, depth + 1, max_depth, min_size)
    f_branch = build_tree(f_group, depth + 1, max_depth, min_size)

    return Node(condition, t_branch, f_branch)


def print_tree(node, spacing=""):
    if isinstance(node, FinalLeaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.condition))

    print (spacing + '--> True:')
    print_tree(node.true_group, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_group, spacing + "  ")

#
# test_set = [[751.3130660697681, 5.77333462158451, 31.17356346461413, 3.0], [566.9345911469687, -239.8546809331984, -1.178536600531432, 1.0], [-522.7244224610362, -209.99797853744755, -53.87118217238084, 1.0], [103.57130630111855, 94.33970724695118, 224.2945551453468, 2.0], [-711.3115748148543, -74.75891459336587, -22.39007244751537, 4.0], [-324.46447062726776, -232.2239990362868, -43.60648806085337, 4.0], [421.6895247191794, -268.725678097986, 10.845418831665771, 3.0], [-153.4908215364213, -266.08145529107105, -19.414108774415563, 3.0], [-298.38608691277125, 59.51527093800421, -207.21674419271116, 3.0], [-300.5197321869889, -11.737440369441565, 237.64143630197412, 2.0], [219.51383629713223, 96.00277943690745, 227.4963621030181, 4.0], [767.4340135963466, 123.17668706214144, 189.843288272593, 3.0], [477.57381133577536, 87.57487926999013, 99.03210963105178, 2.0], [400.1188800118566, 90.20483829130976, 32.34229407193549, 2.0], [747.4334972145393, 82.31324355915537, 148.60562345456745, 4.0], [-452.33793538764337, 6.489904954083392, -71.73528718850115, 2.0], [-281.7731574934415, -195.91871923869437, -51.90145253824716, 3.0], [819.3048655002666, -108.81537464744864, 64.38555947609451, 3.0], [353.752704097702, -232.81570046666909, -10.534420263613498, 4.0], [-31.373848783575745, -198.91016722755268, -24.909558813338812, 3.0], [-579.2062627523298, -252.31608403308263, -17.673356193402803, 1.0], [378.0077030660125, 213.52568056742436, -52.68264575499874, 0.0], [539.7850558758294, -204.92580035705367, -19.740698599832506, 2.0], [-47.96945109704043, 310.5650549443592, 174.35205402023715, 4.0], [-702.7377256504957, 1145.5815991108982, 17.156181376254064, 1.0], [-611.4222676670558, -195.70888537169105, -38.39027874011708, 4.0], [31.54073121018666, -211.9095051835829, -13.757067607651912, 1.0], [-285.7309675896843, -266.32219968985083, -20.454168709222017, 2.0], [813.4781952616912, -199.95989903363792, -5.761500283843818, 2.0], [26.60932889062864, 28.53221430340293, -63.48063345276977, 0.0], [-826.7874416865466, -288.5760513251618, -20.376879406140684, 3.0], [919.9395685443994, 39.20805791764382, -136.50633496657883, 2.0], [782.2483680325645, -189.94588753121985, -4.713107827958208, 4.0], [365.85112016526136, -277.09721357123885, 12.559116479241553, 3.0], [-285.8258587778502, -286.444826225131, -5.437246765600447, 5.0], [639.679444448035, -219.98881714816028, -5.17867942594666, 4.0], [616.9583798441083, -207.902648069349, -14.428863523101455, 1.0], [-85.21131689235006, -41.50404217046827, -138.83575796093663, 4.0], [533.2423224680431, -236.35919310703676, 13.047429529770728, 1.0], [475.5786794684502, 118.99033292880324, 181.78356731017917, 3.0], [532.0317964000982, -156.3422500958634, 126.72296120905447, 5.0], [614.2131664426311, 32.75514398070269, 0.3460613351737178, 1.0], [-319.6433199627216, -140.70876992933975, -11.059291543962736, 2.0], [-681.2876017356892, -237.34224706916947, -53.46032499092791, 3.0], [115.75502595596772, -139.04582377010723, -90.89229100974792, 5.0], [525.8364638567456, -7.9668139524706705, 169.5156844056397, 4.0], [-110.92559558266824, -57.30468296807201, 31.991681466420545, 3.0], [770.3187159394843, 322.297419949087, 99.05011041466508, 1.0], [31.980666537894272, -228.70560931620693, -21.960564169603007, 0.0], [-791.4802794688477, -273.5766352770739, -34.420671472857066, 3.0], [-304.7943209815504, -140.07787051942174, -82.25854903952144, 5.0], [750.1845162716854, -201.6013249716988, 6.088615173086434, 4.0], [-397.4659626150228, 87.99395335687994, -206.78464844383976, 3.0], [211.57667615295597, -150.73896217549216, -69.68374501685109, 4.0], [-89.93599963511403, -57.00360955437303, 32.571318932895515, 0.0], [-110.92031304041087, -252.8179110109312, -17.566524624925847, 0.0], [452.6265899660462, 64.64082408915039, -121.08380767827748, 1.0], [699.0865757618519, -253.8521399632031, 4.130213078115462, 5.0], [-590.3962087821227, -137.73449872070137, 40.26326042004117, 0.0], [-311.81327705937, -219.54484854706152, -42.25420826536919, 3.0], [-86.0566284959145, -254.91136423274943, -13.923111683264308, 2.0], [575.8692606930016, 506.6125434715166, 18.383274607934844, 4.0], [-708.0779639540898, -233.90117436659352, -59.12854382580069, 4.0], [-112.35062766912267, 843.9727004429247, 369.7139553386874, 3.0], [-699.3422067956184, -139.2972131060913, 37.25466595119489, 4.0], [-873.3688511505804, -148.6513733193033, -26.350679945254868, 4.0], [-596.6080119111347, 451.1215475303059, -11.511604834512609, 0.0], [-322.83952585342547, -227.40430380627944, -26.46060545378894, 5.0], [-706.8503947129961, -103.96874530886012, 193.1167842611594, 5.0], [22.599959042925217, 980.1811268730391, -424.55638291947946, 0.0], [183.77310118762827, -244.09180981668803, -8.234218315452612, 0.0], [-716.0268365875725, 158.73261969865985, 28.84381339191705, 2.0], [-789.8185434607847, -55.305426423328015, 86.45867272139752, 2.0], [522.0770632461881, -57.06726733124488, 11.670529153918736, 3.0], [740.642896707925, 74.39577456187328, -80.41146703752698, 3.0], [-563.254797679527, 55.71601119370757, -214.53121698394293, 5.0], [293.2502038245667, 30.218865994345737, -153.81265360847445, 4.0], [-515.8522569137176, -167.81687611650418, 69.00162465941507, 3.0], [387.0064102716034, -130.5726316681408, 8.455169827361331, 3.0], [565.1891007360622, -159.1886836535724, 16.720536297938285, 2.0], [-787.1442634968844, -79.69433387351984, -70.68866721430341, 3.0], [-373.4193436525121, 20.65620090209721, -172.2230436150843, 3.0], [323.5627228695368, -241.21443866903988, -0.5692824437386454, 2.0], [749.9454791377598, 95.2227285196261, 42.00291851318503, 2.0], [792.7659710547713, -223.48502520191528, -5.255708622830697, 5.0], [-144.22796463252033, 309.17922120934577, 73.79447794682702, 2.0], [-149.4752361760334, 225.7346501081765, 52.90651365664362, 0.0], [-557.8068810024726, 801.8430231805659, 43.35015038801548, 3.0], [489.0011015583581, -261.5246951316079, 34.670102654443355, 2.0], [-100.648814028678, -245.08066471936797, 3.4554418718157285, 4.0], [729.1845418498564, 254.6809640974768, 230.14057871290012, 3.0], [-286.8946752289548, -245.6388547696573, -26.252949123688268, 3.0], [869.3230496599662, -94.27688878150089, -83.8237623265212, 1.0], [86.11573315617666, -237.8990798812008, -22.019889094522195, 3.0], [-93.99373972934816, -270.2681236924167, -1.0118742539279384, 4.0], [853.6314744096555, -123.46148544908284, -50.26808124692654, 0.0], [318.9423829173405, 834.3759990784231, 12.60595935982334, 1.0], [267.0795350854789, 140.5424945856918, -232.58433365341943, 4.0], [-222.03375852588115, -260.907890300398, -13.949465598027212, 5.0], [377.23346885577433, 15.610517607650422, 316.0161585575776, 1.0]]
# test_set = [[2., 2., 2., 2.], [1.5, 1.5, 1.5, 1.5]]
# print(_calc_gini(test_set))
# print(_calc_entropy(test_set))
# test_tree = build_tree(test_set, 0)
# print_tree(test_tree)