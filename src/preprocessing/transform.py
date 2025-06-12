import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def encode_categorical(df, cols):
    encoder = LabelEncoder()
    for col in cols:
        df[col] = encoder.fit_transform(df[col].astype(str))
    return df

def scale_numeric(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def remove_missing_rows(df):
    return df.dropna()

def drop_columns(df, cols):
    return df.drop(columns=cols)

def drop_low_variance(df, threshold=0.0):
    selector = VarianceThreshold(threshold=threshold)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    reduced = selector.fit_transform(numeric_df)
    return pd.DataFrame(reduced, columns=numeric_df.columns[selector.get_support()])

def separate_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y