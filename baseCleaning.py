import numpy as np


def clean_dataframe(df):
    # Replaces all empty values to Nan
    df.replace('', np.nan, inplace=True)

    # Drops columns that have all Nan Values
    df.dropna(axis=1, how='all', inplace=True)

    # Drops rows that have any Nan values
    df.dropna(inplace=True)

    # Resets indexses
    df.reset_index(drop=True, inplace=True)

    return df
