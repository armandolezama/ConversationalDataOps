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
