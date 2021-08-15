import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def preprocess_productivity_dataset(df: pd.DataFrame, normalize: bool = False):

    # Eliminate not numeric and missing element columns for now
    df_selected = df.drop(['date', 'targeted_productivity', 'idle_time',
                           'idle_men', 'no_of_style_change'], axis=1)
    print(df_selected)

    # Fill empty values in column wip with mean of wip values
    df_selected['wip'].fillna((df_selected['wip'].mean()), inplace=True)
    print(df_selected)

    # Shuffle elements of dataframe
    preprocessed_df = df_selected.sample(frac=1).reset_index(drop=True)
    print(preprocessed_df)

    preprocessed_df = df_selected.copy()

    # Convert string valued columns to numeric values
    # There are whitespaces in department column that interferes with encoding, so strip them first
    preprocessed_df['department'] = preprocessed_df['department'].str.strip()
    enc = OrdinalEncoder()
    enc.fit(preprocessed_df[['department', 'quarter', 'day']])
    preprocessed_df[['department', 'quarter', 'day']] = enc.transform(preprocessed_df[['department', 'quarter', 'day']])
    print(preprocessed_df)

    # There is 1 column which has same fields apart from label value. This causes similarity to
    # become 0, which in turn causes divide by 0 error while calculating weighed mean, so drop it
    preprocessed_df.drop_duplicates(subset=['quarter', 'department', 'day', 'team', 'smv', 'wip',
                                        'over_time', 'incentive', 'no_of_workers'], inplace=True)
    print(preprocessed_df)

    if normalize:
        # Apply min-max normalization
        preprocessed_df.loc[:, preprocessed_df.columns != 'actual_productivity'] = MinMaxScaler(). \
            fit_transform(preprocessed_df.loc[:, preprocessed_df.columns != 'actual_productivity'])
        print(preprocessed_df)
        
    return preprocessed_df
