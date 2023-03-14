from baseCleaning import clean_dataframe
from dataset.dataloader import Dataloader
import os

os.chdir('./dataset')
data = Dataloader()

df = data.getListings().sample(50)
print(df)
cleaned_df = clean_dataframe(df)
print("after:", cleaned_df)
