import pandas as pd
import numpy as np
df = pd.DataFrame(
    [
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
    ])
df.columns = ['color', 'size', 'price', 'classlabel']

# Mapping Ordinal Columns
size_mapping = \
    {
        'XL': 3,
        'L': 2,
        'M': 1
    }
df['size'] = df['size'].map(size_mapping)

# Mapping Class Label (Not Ordered)
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)

# Inverse Mapping of Class Label
#  inv_class_mapping = {v: k for k, v in class_mapping.items()}

# Another Way of Transforming
#  from sklearn.preprocessing import LabelEncoder
#  class_le = LabelEncoder()
#  y = class_le.fit_transform(df['classlabel'].values)

# One Hot Encoding
pd.get_dummies(df[['price', 'color', 'size']])