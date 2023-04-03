from sklearn.datasets import load_iris
from baseCleaning import clean_dataframe

from featureSelection import FeatureSelector
from dataset.dataloader import Dataloader
import os

os.chdir('./dataset')
data = Dataloader()
df = data.getListings().sample(50)

cleaned_df = clean_dataframe(df)


x, y = load_iris(return_X_y=True)
print(x.shape)
features = FeatureSelector(x, y)
newX = features.getBestKOptions(3)
print(newX.shape)

newX2 = features.getBestOptionUsingVariance(0.5)
print(newX2.shape)
