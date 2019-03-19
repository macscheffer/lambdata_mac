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
     
     
def macs_labelEncode_meaningful(df, train, cols, target):
    
     # takes in a list of columns to encode in a linear fashion. 
     
     for col in cols:
        
          cur_encodes = pd.factorize(train.pivot_table(index=col, values=target).sort_values(target).index)[1].tolist()
          fnl_encodes = pd.factorize(train.pivot_table(index=col, values=target).sort_values(target).index)[0].tolist()
        
          mapper_dic = dict(zip(cur_encodes, fnl_encodes))
        
          df['MeaningfulEnc_' + col] = df[col].map(mapper_dic)
        
     return df
     
     
def square_features(df, features):
    
    # takes in a list of features and squares them creating a new column in a df.
    # returns the dataframe.
    for feature in features:
        df[str(feature) + '_squared'] = df[feature] ** 2
    
    return df
    
