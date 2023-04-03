from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getBestKOptions(self, k):
        # Selects best k features from the data set
        newX = SelectKBest(f_classif, k=k).fit_transform(self.x, self.y)
        return newX

    def getBestOptionUsingVariance(self, varianceTreshold=0):
        # Vlaues whos variance is below the treshHold will be removed
        return VarianceThreshold(threshold=varianceTreshold).fit_transform(self.x)
