import functions as f
from ingest import upload_csv_to_postgres
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime
from geopy.geocoders import GoogleV3

data_directory_path = './data/'
cleaned_file_path = os.path.join(data_directory_path, 'green_trip_data_2015-04_clean.csv')

if not os.path.exists(cleaned_file_path):
    taxis_df = f.load_dataset(data_directory_path,'green_tripdata_2015-04.csv')

    ## Inconsistencies

    taxis_df_cleaned = taxis_df.copy()
    f.rename_columns(taxis_df_cleaned)


    f.change_column_to_datetime(taxis_df_cleaned, 'lpep_pickup_datetime')
    f.change_column_to_datetime(taxis_df_cleaned, 'lpep_dropoff_datetime')


    threshold_start_date = datetime(2015, 4, 1)
    threshold_end_date = datetime(2015, 4, 30)

    condition = (taxis_df_cleaned['lpep_pickup_datetime'] > threshold_end_date)
    taxis_df_cleaned = f.keep_rows_not_following_condition(taxis_df_cleaned, condition)

    condition = (taxis_df_cleaned['lpep_dropoff_datetime'] > threshold_end_date)
    taxis_df_cleaned = f.keep_rows_not_following_condition(taxis_df_cleaned, condition)

    taxis_df_cleaned = taxis_df_cleaned.drop_duplicates()

    neg_columns = f.columns_with_negatives(taxis_df_cleaned) 
    taxis_df_cleaned = f.add_sum_of_negative_columns(taxis_df_cleaned, neg_columns)


    condition = (taxis_df_cleaned['sum_of_columns'] == taxis_df_cleaned['total_amount']) & (taxis_df_cleaned['sum_of_columns'] < 0)
    taxis_df_cleaned.loc[condition, neg_columns] = taxis_df_cleaned.loc[condition, neg_columns].abs() # get absolute of those negatives

    taxis_df_cleaned = f.drop_column(taxis_df_cleaned, 'sum_of_columns') # Remove this temp column from our datraframe
    taxis_df_cleaned = f.delete_records_with_negatives(taxis_df_cleaned, neg_columns)


    condition = (taxis_df_cleaned['extra'] != 0.5) & (taxis_df_cleaned['extra'] != 1) & pd.notna(taxis_df_cleaned['extra'])
    f.remove_from_total(taxis_df_cleaned, condition, 'extra')
    f.replace_with_zero(taxis_df_cleaned, condition, 'extra')


    ## Missing Values
    lookup_table = f.create_empty_lookup_table()

    taxis_df_cleaned = f.drop_columns(taxis_df_cleaned, ['ehail_fee', 'congestion_surcharge'])

    taxis_df_cleaned, lookup_table = f.impute_with_median(taxis_df_cleaned, 'passenger_count', lookup_table)
    taxis_df_cleaned['passenger_count'] = taxis_df_cleaned.passenger_count.astype(int)

    taxis_df_cleaned = f.replace_missing_value(taxis_df_cleaned,'Uknown')
    taxis_df_cleaned, lookup_table = f.impute_with_mode(taxis_df_cleaned, 'payment_type', lookup_table, original_column_value = 'Uknown')
    taxis_df_cleaned.loc[taxis_df_cleaned['tip_amount'] > 0, 'payment_type'] = 'Credit card'

    taxis_df_cleaned, lookup_table = f.impute_with_value(taxis_df_cleaned, 'extra', 0, lookup_table)

    taxis_df_cleaned = f.replace_missing_value(taxis_df_cleaned,'Unknown')
    taxis_df_cleaned, lookup_table = f.impute_with_mode(taxis_df_cleaned, 'rate_type', lookup_table, original_column_value = 'Unknown')

    taxis_df_cleaned, lookup_table = f.impute_with_mode(taxis_df_cleaned, 'trip_type', lookup_table)


    ## Outliers

    taxis_df_cleaned = taxis_df_cleaned[taxis_df_cleaned['passenger_count'] != 444]

    taxis_df_cleaned = f.cap_at_percentile(taxis_df_cleaned, 'tolls_amount', 99)

    lower_bound, upper_bound = f.get_outlier_bounds(taxis_df_cleaned, 'fare_amount')
    taxis_df_cleaned = f.impute_outliers_with_mean(taxis_df_cleaned, 'fare_amount', lower_bound, upper_bound)


    lower_bound, upper_bound = f.get_outlier_bounds(taxis_df_cleaned, 'trip_distance')
    taxis_df_cleaned = f.impute_outliers_with_mean(taxis_df_cleaned, 'trip_distance', lower_bound, upper_bound)

    lower_bound, upper_bound = f.get_outlier_bounds(taxis_df_cleaned, 'tip_amount')
    taxis_df_cleaned = f.impute_outliers_with_mean(taxis_df_cleaned, 'tip_amount', lower_bound, upper_bound)


    ## Dsicretization

    taxis_df_cleaned = f.equal_width_discretization(taxis_df_cleaned, 'trip_distance', 3, labels = ["short", "medium", "long"] )

    taxis_df_cleaned = f.equal_width_discretization(taxis_df_cleaned, 'fare_amount', 4, labels = ["Low Fare", "Medium Fare", "High Fare", "Very High Fare"] )

    taxis_df_cleaned = f.equal_width_discretization(taxis_df_cleaned, 'tip_amount', 3, labels = ["Low Tip", "Moderate Tip", "Good Tip"] )

    taxis_df_cleaned = f.equal_width_discretization(taxis_df_cleaned, 'passenger_count', 3, labels = ["Few Passengers", "Average Passenger Count", "Lots of Passengers"] )

    taxis_df_cleaned = f.equal_width_discretization(taxis_df_cleaned, 'tolls_amount', 3, labels = ["Low Tolls Amount", "Average Tolls Amount", "High Tolls Amount"] )

    labels = [i+1 for i in range(5)]
    taxis_df_cleaned = f.equal_width_discretization(taxis_df_cleaned, 'lpep_pickup_datetime', 5, new_column_name = 'week_number', labels=labels)

    taxis_df_cleaned = f.equal_width_discretization_with_boundaries(taxis_df_cleaned, 'lpep_pickup_datetime', 5, new_column_name = 'date_range')

    ## Feature Engineering

    taxis_df_cleaned['trip_duration_mins'] = round((taxis_df_cleaned['lpep_dropoff_datetime'] - taxis_df_cleaned['lpep_pickup_datetime']).dt.total_seconds() / 60)
    taxis_df_cleaned['weekend_trip'] = (taxis_df_cleaned['lpep_pickup_datetime'].dt.dayofweek >= 5).astype(int)

    taxis_df_cleaned['pickup_hour'] = taxis_df_cleaned['lpep_pickup_datetime'].dt.hour
    conditions = [
        (taxis_df_cleaned['pickup_hour'] >= 5) & (taxis_df_cleaned['pickup_hour'] < 12),
        (taxis_df_cleaned['pickup_hour'] >= 12) & (taxis_df_cleaned['pickup_hour'] < 18)
    ]
    time_of_day_labels = ['Morning', 'Afternoon']
    taxis_df_cleaned['time_of_day'] = np.select(conditions, time_of_day_labels, default='Night')
    taxis_df_cleaned = f.drop_column(taxis_df_cleaned, 'pickup_hour') # Remove this temporary column

    ## GPS Coordinates

    geocoder = GoogleV3(api_key='AIzaSyASSdXSSWKeX3lQswJ6mKIlcg3BD429EZI')

    try:
        location_df = pd.read_csv(data_directory_path + 'coordinates.csv')
    except FileNotFoundError:
        location_df = f.create_empty_location_df()
        # Fill dataframe using API
        for location in taxis_df['PU Location'].unique():
            location_df = f.add_location_entry(geocoder, location, location_df)

        for location in taxis_df['DO Location'].unique():
            location_df = f.add_location_entry(geocoder, location, location_df)

        # Save to CSV
        output_path = os.path.join(data_directory_path, 'coordinates.csv')
        f.export_csv(location_df, output_path)

    f.create_coordinates_column (taxis_df_cleaned, 'pu_location', 'pickup_coordinates', location_df)
    f.create_coordinates_column (taxis_df_cleaned, 'do_location', 'dropoff_coordinates', location_df)
    
    ## Encoding

    lookup_table = f.label_encode_column(taxis_df_cleaned, 'pu_location', lookup_table)
    lookup_table = f.label_encode_column(taxis_df_cleaned, 'do_location', lookup_table)

    columns_to_encode = ['vendor', 'payment_type', 'trip_type', 'rate_type', 'store_and_fwd_flag']
    taxis_df_cleaned = f.one_hot_encode_columns(taxis_df_cleaned, columns_to_encode)

    ## Exporting
    lookup_output_path = os.path.join(data_directory_path, 'lookup_table_green_taxis.csv')
    try:
        pd.read_csv(lookup_output_path)
    except FileNotFoundError:
        f.export_csv(lookup_table, lookup_output_path)

    cleaned_output_path = os.path.join(data_directory_path,  'green_trip_data_2015-04_clean.csv')
    try:
        pd.read_csv(cleaned_output_path)
    except FileNotFoundError:
        f.export_csv(taxis_df_cleaned, cleaned_output_path)
    
    # Uploading to Postgres
    upload_csv_to_postgres(cleaned_file_path, 'green_taxi_4_2015')
    upload_csv_to_postgres(os.path.join(data_directory_path, 'lookup_table_green_taxis.csv'), 'lookup_green_taxi_4_2015')
else:
    print('Data has been already cleaned.')
    print('Attempting to upload to Postgres...')
    upload_csv_to_postgres(cleaned_file_path, 'green_taxi_4_2015')
    upload_csv_to_postgres(os.path.join(data_directory_path, 'lookup_table_green_taxis.csv'), 'lookup_green_taxi_4_2015')

    