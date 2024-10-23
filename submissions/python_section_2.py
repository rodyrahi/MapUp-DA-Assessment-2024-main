import time
import numpy as np
import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here


    start_ids = df['start_id'].unique()
    end_ids = df['end_id'].unique()
    
    all_ids = sorted(set(start_ids).union(set(end_ids)))
    
    distance_matrix = pd.DataFrame(np.inf, index=all_ids, columns=all_ids)
    
  
    np.fill_diagonal(distance_matrix.values, 0)
    
   
    for _, row in df.iterrows():
        start, end, distance = row['start_id'], row['end_id'], row['distance']
        distance_matrix.loc[start, end] = distance
        distance_matrix.loc[end, start] = distance  # Ensure symmetry
        
   
    for k in all_ids:
        for i in all_ids:
            for j in all_ids:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]

    return distance_matrix



def unroll_distance_matrix(distance_matrix):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    unrolled_data = []
    
   
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  
                unrolled_data.append([id_start, id_end, distance_matrix.loc[id_start, id_end]])
    
    
    result_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    return result_df


# distance_matrix = calculate_distance_matrix('dataset-2.csv')  # Use the distance matrix from Question 9
# result_df = unroll_distance_matrix(distance_matrix)
# print(result_df)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    reference_df = df[df['id_start'] == reference_id]
    
    reference_avg_distance = reference_df['distance'].mean()
    
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1
    
    valid_ids = []
    for id_start in df['id_start'].unique():
        avg_distance = df[df['id_start'] == id_start]['distance'].mean()
        
        if lower_bound <= avg_distance <= upper_bound:
            valid_ids.append(id_start)
    

    return sorted(valid_ids)




def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
     
    weekday_discount_factors = [
        (time(0, 0), time(10, 0), 0.8),
        (time(10, 0), time(18, 0), 1.2),
        (time(18, 0), time(23, 59, 59), 0.8)
    ]
    
 
    weekend_discount_factor = 0.7


    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekends = ["Saturday", "Sunday"]


    df["start_day"] = pd.Series(weekdays * (len(df) // len(weekdays) + 1)).head(len(df))
    df["end_day"] = df["start_day"]  

    df["start_time"] = time(0, 0)
    df["end_time"] = time(23, 59, 59)


    def apply_discount(row):
        day = row['start_day']
    
        if day in weekdays:
            for start, end, factor in weekday_discount_factors:
                if start <= row['start_time'] <= end:
                    return factor
 
        elif day in weekends:
            return weekend_discount_factor
        return 1  

   
    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
        df[vehicle] = df[vehicle] * df.apply(apply_discount, axis=1)

    return df