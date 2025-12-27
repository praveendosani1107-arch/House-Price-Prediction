import numpy as np
import pandas as pd

df = pd.read_csv("bengaluru_house_prices (1).csv")
# print(df.head(10))
# print(df.shape)
df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)
# print(df2.head())
# print(df2.isnull().sum())
df3 = df2.dropna()
# print(df3.isnull().sum())
# print(df3['size'].unique())
df3 = df3.copy()
df3.loc[:, 'bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

# print(df3['bhk'])
# print(df3)
df4 = df3.drop('size', axis=1)
# print(df4)
# def is_float(x):
#     try:
#         float(x)
#         return True
#     except:
#         return False

# print(df4['total_sqft'].apply(is_float).head(10))
# print("hello")
def convert_sqft_to_num(x):
    try:
        # case: range like "2100-2850"
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        
        # case: single number like "1200"
        return float(x)
    
    except:
        return None


df5 = df4.copy()
df5['total_sqft'] = df5['total_sqft'].apply(convert_sqft_to_num)

# print(df5.head(3))
df6=df5.copy()
df6['price_per_sqft']=df5['price']*100000/df5['total_sqft']
# print(df6.head(5))
unique_count = len(df6.drop_duplicates())
# print(f"count: {unique_count}")
df6['location'] = df6['location'].apply(lambda x: x.strip())

location_stats = (
    df6.groupby('location')['location']
       .agg('count')
       .sort_values(ascending=False)
)

# print(location_stats)
len(location_stats[location_stats <= 10])

location_stats_less_than_10 = location_stats[location_stats <= 10]
# print(location_stats_less_than_10)
df6['location'] = df6['location'].apply(
    lambda x: 'other' if x in location_stats_less_than_10.index else x
)
# len(df5['location'].unique())
# print(df6.shape)
df6[df6.total_sqft/df6.bhk<300]
df7 = df6[~(df6.total_sqft / df6.bhk < 300)]
# print(df7.shape)
# print(df7.price_per_sqft.describe())
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[
            (subdf.price_per_sqft > (m - st)) &
            (subdf.price_per_sqft <= (m + st))
        ]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df8 = remove_pps_outliers(df7)
# print(df8.shape)
# Step 1: filter rows
df9 = df8[df8.bath < df8.bhk + 2]

# Step 2: drop unwanted columns
df9 = df9.drop(['price_per_sqft'], axis='columns')

# Step 3: preview
# print(df9.head(3))
# create dummy variables for location
dummies = pd.get_dummies(df9['location'])

# drop 'other' column
dummies = dummies.drop('other', axis=1)

# merge dummies with original dataframe
df10 = pd.concat([df9.drop('location', axis=1), dummies], axis=1)

# print(df10.head())
# split features and target
X = df10.drop('price', axis=1)
y = df10['price']

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

# train linear regression model
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()

lr_clf.fit(X_train, y_train)

# check accuracy (R² score)
# print(lr_clf.score(X_test, y_test))

# print()

def predict_price(location, sqft, bath, bhk):
    x = pd.DataFrame(
        np.zeros((1, len(X.columns))),
        columns=X.columns
    )

    x['total_sqft'] = sqft
    x['bath'] = bath
    x['bhk'] = bhk

    if location in x.columns:
        x[location] = 1

    return lr_clf.predict(x)[0]

print(predict_price('1st Phase JP Nagar', 1000, 2, 2))









