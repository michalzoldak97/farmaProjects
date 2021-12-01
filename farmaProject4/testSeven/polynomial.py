class PolynomialFeaturesForDegree():
    
    def __init__(self, degree):
        self.degree = degree
        self.options = []
        
    
    def get_terms(self, features):
        m = len(features[0])
        
        num = (self.degree + 1) ** m
        ls = [0.0] * m 
        
        for x in range(1,num):
            for idx in range(len(ls)-1, -1, -1):
                if ls[idx] == 0:
                    ls[idx] += 1
                    break
                elif ls[idx] % self.degree == 0:
                    ls[idx] = 0
                    if ls[idx - 1] != self.degree:
                        ls[idx - 1] += 1
                        break
                    else:
                        deficit = -1
                        stop = False
                        while not stop and (idx + deficit) > -1:
                            if ls[idx + deficit] != self.degree:
                                ls[idx + deficit] += 1
                                stop = True
                            else:
                                ls[idx + deficit] = 0
                                deficit -= 1
                        break
                elif idx == (len(ls) - 1):
                    ls[idx] += 1
                    break
            
            if sum(ls) == self.degree:
                self.options.append(tuple(ls))
        
    
    def get_polynomial_features(self, features):
        self.get_terms(features)
        
        poly_features = [[] for _ in enumerate(features)]
        feature_ls = []
        
        for i in range(len(features[0])):
            to_add = [row[i] for row in features]
            feature_ls.append(to_add)

        for i,tup in enumerate(self.options):
            vector = [1. for _ in enumerate(features)]
            
            for feature_idx, exp in enumerate(tup):
                for j, el in enumerate(vector):
                    vector[j] = el * (feature_ls[feature_idx][j] ** exp)
            
            
            for i, _ in enumerate(poly_features):
                poly_features[i].append(vector[i])
                
        return poly_features



class PolynomialFeatures():

    def __init__(self, degree):
        self.degree = degree
        
    def fit(self):
        self.transformers = []
        for i in range(1, self.degree + 1):
            self.transformers.append(PolynomialFeaturesForDegree(i))
        
    def transform(self,dataset):
        self.fit()
        temp = [[1.0] for _ in enumerate(dataset)]
        for t in self.transformers:
            features_for_deg = t.get_polynomial_features(dataset)
            for i, _ in enumerate(temp):
                for el in features_for_deg[i]:
                    temp[i].append(el)
        return temp