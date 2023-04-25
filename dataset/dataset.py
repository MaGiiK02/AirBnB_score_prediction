import pandas as pd

# A dataset class that can be used to access to the dataset while granting some extra functions
class Dataset():
  def __init__(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
      self.x = x
      self.y = y

  def get(self):
    return self.x, self.y.review_scores_rating

  def getListings(self):
    conlumns = [ col for col in self.x.columns if "comment" not in col]
    return self.x[conlumns]

  def getListingsEmbeddings(self):
    conlumns = [ col for col in self.x.columns if "embedding" in col and "comment" not in col]
    return self.x[conlumns]

  def getListingsNotEmbeddings(self):
    conlumns = [ col for col in self.x.columns if "embedding" not in col and "comment" not in col ]
    return self.x[conlumns]

  def getComments(self):
    conlumns = [ col for col in self.x.columns if "comment" in col ]
    return self.x[conlumns]

  def getEmbeddings(self):
    conlumns = [ col for col in self.x.columns if "embedding" in col ]
    return self.x[conlumns]

  def getNotEmbeddings(self):
    conlumns = [ col for col in self.x.columns if "embedding" not in col ]
    return self.x[conlumns]

  def getAllScores(self):
    return self.y