import time
import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from geopy.geocoders import GoogleV3
from geopy.exc import GeocoderTimedOut

def load_dataset(path, file_name):
    return pd.read_csv(path + file_name)

def rename_columns(df):
    df.columns = df.columns.str.lower()
    df.columns = [col.replace(' ', '_') for col in df.columns]

def change_column_to_datetime(dataframe, column_name):
    try:
        dataframe[column_name] = pd.to_datetime(dataframe[column_name])
    except Exception as e:
                print(f"Error: Could not convert column '{column_name}' to datetime due to {e}.")

def filter_dataframe_by_date(df, date_column, threshold_date, before=True):
    if before:
        filtered_df = df[df[date_column] < threshold_date]
    else:
        filtered_df = df[df[date_column] > threshold_date]
    return filtered_df

def keep_rows_not_following_condition(df, condition):
    filtered_df = df[~condition]
    return filtered_df

def sort_dataframe(dataframe, column_name, ascending=True):
    sorted_df = dataframe.sort_values(by=column_name, ascending=ascending)
    return sorted_df

def remove_from_total(df, condition, column_name):
    df.loc[condition, 'total_amount'] = df['total_amount'] - df[column_name]

def replace_with_zero(df, condition, column_name):
    df.loc[condition, column_name] = 0

def filter_dataframe(df, condition):
    filtered_df = df[condition]
    return filtered_df

def columns_with_negatives(df):
    numeric_columns = df.select_dtypes(include=['int', 'float'])
    columns_with_negatives = numeric_columns.columns[(numeric_columns < 0).any()]
    return columns_with_negatives
    
def min_of_columns(df, columns_list):
    if isinstance(columns_list, pd.Index):
        columns_list = columns_list.to_list()
    min_values = tuple(df[col].min() for col in columns_list)
    return min_values

def count_records_with_negatives(df, columns_with_negatives):
    num_records_with_negatives = (df[columns_with_negatives] <0).any(axis=1).sum()
    return num_records_with_negatives

def delete_records_with_negatives(df, columns_with_negatives):
    df_cleaned = df[~(df[columns_with_negatives] < 0).any(axis=1)]
    return df_cleaned

def add_sum_of_negative_columns(df, neg_columns):
    # Exclude 'total_amount' from negative columns list and calculate sum of columns containing negative values
    if 'total_amount' in neg_columns:
        neg_columns = neg_columns.drop('total_amount')
    df['sum_of_columns'] = df[neg_columns].sum(axis=1)
    return df

def drop_column(df, column):
    if column in df.columns:
        df = df.drop(column, axis=1)
    return df

def create_empty_lookup_table():
    lookup_table = pd.DataFrame(columns=['Column', 'Original_Value', 'Imputed_Value'])
    return lookup_table

def append_lookup_table(lookup_table, column_name, original_value, imputed_value):
    new_row = {'Column': column_name, 'Original_Value': original_value, 'Imputed_Value': imputed_value}
    lookup_table.loc[len(lookup_table.index)] = [column_name, original_value, imputed_value] 
    return lookup_table

def drop_columns(dataframe, columns_to_drop):
    cleaned_dataframe = dataframe.drop(columns_to_drop, axis=1)
    return cleaned_dataframe

def impute_with_median(dataframe, column_name, lookup_table, inplace=True, original_column_value = 'NaN'):
    median_value = dataframe[column_name].median()
    dataframe[column_name].fillna(median_value, inplace=inplace)
    lookup_table = append_lookup_table(lookup_table, column_name, original_column_value, median_value)
    return dataframe, lookup_table

def impute_with_mode(dataframe, column_name, lookup_table, inplace=True, original_column_value = 'NaN'):
    mode_value = dataframe[column_name].mode()[0] 
    dataframe[column_name].fillna(mode_value, inplace=inplace)
    lookup_table = append_lookup_table(lookup_table, column_name, original_column_value, mode_value)
    return dataframe, lookup_table

def impute_with_value(dataframe, column_name, impute_value, lookup_table, inplace=True, original_column_value = 'NaN'):
    dataframe[column_name].fillna(impute_value, inplace=inplace)
    lookup_table = append_lookup_table(lookup_table, column_name, original_column_value, impute_value)
    return dataframe, lookup_table

def replace_missing_value(dataframe, value_to_replace):
    return dataframe.replace(value_to_replace, np.nan)

def get_quantile(df, column_name, quantile):
    return df[column_name].quantile(quantile / 100)

def get_outlier_bounds(dataframe, column_name):
    Q1 = get_quantile(dataframe, column_name, 25)
    Q3 = get_quantile(dataframe, column_name, 75)
    iqr = Q3 - Q1
    cut_off = iqr * 1.5
    lower_bound = Q1 - cut_off
    upper_bound = Q3 + cut_off
    return lower_bound, upper_bound

def get_iqr(df, column_name):
    Q1 = get_quantile(df, column_name, 25)
    Q3 = get_quantile(df, column_name, 75)
    return Q3 - Q1
    
def get_skewiness(df, column_name):
    return df[column_name].skew()

def cap_at_percentile(df, column_name, quantile):
    percentile_value = get_quantile(df, column_name, quantile)
    df[column_name] = np.minimum(df[column_name], percentile_value)
    return df

def impute_outliers_with_mean(df, column_name, lower_bound, upper_bound):
    outliers = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
    df[column_name][outliers] = df[column_name][~outliers].mean()
    return df

def impute_outliers_with_median(df, column_name, lower_bound, upper_bound):
    median_value = df[column_name].median()
    df[column_name][(df[column_name] < lower_bound) | (df[column_name] > upper_bound)] = median_value
    return df

def equal_width_discretization(dataframe, column_name, num_bins, labels = None, new_column_name = None):
    if new_column_name is None:
        new_column_name = f"{column_name}_disc"
    if labels is None:
        labels = [f"Bin_{i+1}" for i in range(num_bins)]
    bins = pd.cut(dataframe[column_name], bins=num_bins, labels = labels)
    dataframe[new_column_name] = bins
    return dataframe

def equal_width_discretization_with_boundaries(dataframe, column_name, num_bins, new_column_name = None):
    if new_column_name is None:
        new_column_name = f"{column_name}_disc"
    bins = pd.cut(dataframe[column_name], bins=num_bins)
    dataframe[new_column_name] = bins
    return dataframe

def label_encode_column(dataframe, column_name, lookup_table):
    original_values = dataframe[column_name].unique()
    label_encoder = LabelEncoder()
    # Adjust dataframe
    dataframe[column_name] = label_encoder.fit_transform(dataframe[column_name])

    # Adjust lookup_table
    encoded_values = label_encoder.fit_transform(original_values)
    for original_value, encoded_value in zip(original_values, encoded_values):
        lookup_table = append_lookup_table(lookup_table, column_name, original_value, encoded_value)
    return lookup_table

def one_hot_encode_columns(dataframe, columns_list):
    for column_name in columns_list:
        encoded_columns = pd.get_dummies(dataframe[column_name], prefix=column_name, drop_first=True)
        dataframe = pd.concat([dataframe, encoded_columns], axis=1)
        # dataframe.drop(column_name, axis=1, inplace=True)
    return dataframe

def get_coordinates(geocoder, location):
    try:
        location_info = geocoder.geocode(location)
        if location_info:
            return location_info.latitude, location_info.longitude
    except GeocoderTimedOut:
        time.sleep(1)
        return get_coordinates(location)
    return None, None

def create_empty_location_df():
    return pd.DataFrame(columns=['Location', 'Latitude', 'Longitude'])

def append_to_location_df (location_df, location, latitude, longitude):
    new_row = {'Location': location, 'Latitude': latitude, 'Longitude': longitude}
    location_df.loc[len(location_df.index)] = [location, latitude, longitude] 
    return location_df

def add_location_entry(geocoder, location, location_df):
    latitude, longitude = get_coordinates(geocoder, location)
    location_df = append_to_location_df(location_df, location, latitude, longitude)
    return location_df

def export_csv(df, output_path, include_index = False):
    df.to_csv(output_path, index=False)

def get_coordinates_from_location(location, location_df):
    df = location_df[location_df['Location'] == location]
    return (df.iloc[0][1], df.iloc[0][2])

def create_coordinates_column(df, old_column, new_column, location_df):
    mapping_dict = location_df.set_index('Location').to_dict()
    latitiude_series = df[old_column].map(mapping_dict['Latitude'])
    longitude_series = df[old_column].map(mapping_dict['Longitude'])
    df[new_column] = list(zip(latitiude_series, longitude_series))






