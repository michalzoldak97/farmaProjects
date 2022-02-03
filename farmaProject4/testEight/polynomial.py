class PolynomialFeaturesForDegree:
    
    def __init__(self, deg):
        self.deg = deg
        self.opts = []

    def _get_terms(self, feats):
        m = len(feats[0])
        num = (self.deg + 1) ** m
        ls = [0.0] * m 
        
        for x in range(1, num):
            for idx in range(len(ls)-1, -1, -1):
                if ls[idx] == 0:
                    ls[idx] += 1
                    break
                elif ls[idx] % self.deg == 0:
                    ls[idx] = 0
                    if ls[idx - 1] != self.deg:
                        ls[idx - 1] += 1
                        break
                    else:
                        dfc = -1
                        stop = False
                        while not stop and (idx + dfc) > -1:
                            if ls[idx + dfc] != self.deg:
                                ls[idx + dfc] += 1
                                stop = True
                            else:
                                ls[idx + dfc] = 0
                                dfc -= 1
                        break
                elif idx == (len(ls) - 1):
                    ls[idx] += 1
                    break
            
            if sum(ls) == self.deg:
                self.opts.append(tuple(ls))

    def get_polynomial_features(self, feats):
        self._get_terms(feats)
        poly_features = [[] for _ in enumerate(feats)]
        feature_ls = []
        
        for i in range(len(feats[0])):
            to_add = [row[i] for row in feats]
            feature_ls.append(to_add)

        for i, tup in enumerate(self.opts):
            vec = [1. for _ in enumerate(feats)]
            
            for feature_idx, exp in enumerate(tup):
                for j, el in enumerate(vec):
                    vec[j] = el * (feature_ls[feature_idx][j] ** exp)

            for k, _ in enumerate(poly_features):
                poly_features[k].append(vec[k])
                
        return poly_features


class PolynomialFeatures:

    def __init__(self, deg):
        self.deg = deg
        self.trnsf = []
        
    def fit(self):
        for i in range(1, self.deg + 1):
            self.trnsf.append(PolynomialFeaturesForDegree(2)) # chnged to 2
        
    def transform(self, dataset):
        self.fit()
        temp = [[1.0] for _ in enumerate(dataset)]
        for t in self.trnsf:
            features_for_deg = t.get_polynomial_features(dataset)
            print("Features for deg: {}".format(features_for_deg))
            for i, _ in enumerate(temp):
                for el in features_for_deg[i]:
                    temp[i].append(el)
        return temp
