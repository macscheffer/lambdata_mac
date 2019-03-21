import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random



def date_to_features(col, df=df, drop=False):
    
    # a function to split a pandas datetime column into a different date columns (year, month, day, day of week, day of year)
     datetime = pd.to_datetime(df[col])
     df['datetime_year'] = datetime.dt.year
     df['datetime_month'] = datetime.dt.month
     df['datetime_dayofweek'] = datetime.dt.dayofweek
    
     df['datetime_dayofyear'] = datetime.dt.dayofyear
     df['datetime_quarter'] = datetime.dt.quarter
     df['datetime_weekofyear'] = datetime.dt.weekofyear
    
     if drop == True:
          return df.drop(columns=col)
     return df
     
     
     
def square_features(df, features):
    
    # takes in a list of features and squares them creating a new column in a df.
    # returns the dataframe.
    for feature in features:
        df[str(feature) + '_squared'] = df[feature] ** 2
    
    return df
    


def macs_pca_features(df, cols, pcs, clusters, prefix):
     
     #add a function/functionality to 
     #vizualize sufficient pcs

    
     # takes in a df. columns to reduce   dimensions, how many dimensions to reduce to, and how many clusters to make.
     
     
    df = df.copy()

    scaler = StandardScaler()
    std_df = scaler.fit_transform(df[cols])
     
    pca = PCA(pcs)
    pc_df = pca.fit_transform(std_df)
     

    for cluster in clusters:
        km = KMeans(cluster)
        km = km.fit(pc_df)
        df[prefix + str(pcs) + 'PCs_' + str(cluster) + 'Means_Cluster'] = km.labels_

    return df




def mac_labelEncode(df, min_instances, cols):
    
    # encodes columns as long as it means some minumum number of instances
    
    for col in cols:
        
        new_labels = {}
        for k, v in zip(df[col].value_counts().index.tolist(), df[col].value_counts().values.tolist()):
            
            if v > min_instances:
                new_labels[k] = k
            else:
                new_labels[k] = 'misc'
        
        df['Encoded_' + col] = pd.factorize(df[col].map(new_labels))[0]
        
    return df




def mac_labelEncode_linear(df, train, cols, target):
    
     # takes in a list of columns to encode in a linear fashion.
     
    for col in cols:
        
        cur_encodes = pd.factorize(train.pivot_table(index=col, values=target).sort_values(target).index)[1].tolist()
        fnl_encodes = pd.factorize(train.pivot_table(index=col, values=target).sort_values(target).index)[0].tolist()
        
        mapper_dic = dict(zip(cur_encodes, fnl_encodes))
        
        df['MeaningfulEnc_' + col] = df[col].map(mapper_dic)
    
    return df



def mac_create_dummies_meaningful(df, train, cols, target, target_mean_value, cutoff):
    
    # creates dummies if the category meets some information standard.
    
    for col in cols:
        
        for k, v in zip((abs(train.pivot_table(index=col, values=target) - target_mean_value) > cutoff).index.tolist(),
                        (abs(train.pivot_table(index=col, values=target) - target_mean_value) > cutoff)[target].values):
            if v:
                train['dummy_' + col + str(k)] = (train[col] == k)
                df['dummy_' + col + str(k)] = (df[col] == k)
        
    return train, df

