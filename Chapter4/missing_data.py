import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
... 1.0,2.0,3.0,4.0
... 5.0,6.0,,8.0
... 10.0,11.0,12.0,'''
# If you are using Python 2.7, you need
# to convert the string to unicode:
csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))

# To find NaN in each column
# df.isnull().sum()

# To drop rows with NaN
# df.dropna()

# TO drop columsn with NaN
# df.dropna(axis=1)

# only drop rows where all columns are NaN
#df.dropna(how='all')

# drop rows that have not at least 4 non-NaN values
#df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (here: 'C')
#df.dropna(subset=['C'])


# Replace Nan with mean
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
