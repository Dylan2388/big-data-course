import pandas as pd

genres = pd.read_csv('http://www.millionsongdataset.com/sites/default/files/lastfm/lastfm_unique_tags.txt', sep=" ", header=None)
genres.columns = ["genre", "freq"]
print(genres)
