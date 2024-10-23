from datetime import timedelta
import re
from typing import Dict, List

import numpy as np
import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []
    i = 0
    while i < len(lst):
        group = []

        for j in range(i, min(i + n, len(lst))):
            group.append(lst[j])

        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
        i += n
    return result

# print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here

    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    

    sorted_dict = dict(sorted(length_dict.items()))
    

    return sorted_dict

input1 = ["apple", "bat", "car", "elephant", "dog", "bear"]
input2 = ["one", "two", "three", "four"]

# print(group_by_length(input1))
# print(group_by_length(input2))

def flatten_dict(d, parent_key='', sep='.'):
    flattened = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            # Recursively flatten the nested dictionary
            flattened.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            # Handle lists by adding index to the key
            for idx, item in enumerate(v):
                list_key = f"{new_key}[{idx}]"
                if isinstance(item, dict):
                    flattened.update(flatten_dict(item, list_key, sep=sep))
                else:
                    flattened[list_key] = item
        else:
            # Assign the value to the final key
            flattened[new_key] = v

    return flattened

data = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

# print(flatten_dict(data))



def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start, end):
        if start == end :
     
            result.append(nums[:])
        for i in  range(start, end):
       
            if i >  start and nums[i] == nums[start]:
                continue
   
            nums[start], nums[i] = nums[i], nums[start]
     
            backtrack(start + 1, end)

            nums[start],  nums[i] = nums[i], nums[start]

    result = []
    nums.sort()  
    backtrack(0, len(nums))
    return result

input_list = [1, 1, 2]
unique_permutations = unique_permutations(input_list)
print(unique_permutations)



def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{4}\.\d{2}\.\d{2}\b' 
    ]
    
 
    dates = []
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    
    return dates


text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
valid_dates = find_all_dates(text)
print(valid_dates)





def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    return pd.Dataframe()


def rotate_matrix_90_clockwise(matrix):
    return [list(row[::-1]) for row in zip(*matrix)]

def sum_row_col_excluding_self(matrix):
    n = len(matrix)
    
    result_matrix = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(matrix[i]) - matrix[i][j]
            col_sum = sum(matrix[k][j] for k in range(n)) - matrix[i][j]
            result_matrix[i][j] = row_sum + col_sum
    
    return result_matrix

def transform_matrix(matrix):
    rotated_matrix = rotate_matrix_90_clockwise(matrix)
    
    final_matrix = sum_row_col_excluding_self(rotated_matrix)
    
    return final_matrix

matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

result = transform_matrix(matrix)
print(np.array(result))


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%Y-%m-%d %H:%M:%S')
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%Y-%m-%d %H:%M:%S')

  
    df.set_index(['id', 'id_2'], inplace=True)
    
 
    result = pd.Series(index=df.index.unique(), dtype=bool)
    

    full_week = set(range(7))  
    full_day = timedelta(hours=24)  


    for idx, group in df.groupby(level=['id', 'id_2']):
       
        covered_days = set()
        full_coverage = True
        
       
        for _, row in group.iterrows():
            start_day = row['start_timestamp'].weekday() 
            end_day = row['end_timestamp'].weekday()     
            duration = row['end_timestamp'] - row['start_timestamp']
            
            
            if duration < full_day:
                full_coverage = False
            
          
            covered_days.update(range(start_day, end_day + 1))
        
        
        if covered_days != full_week:
            full_coverage = False
        
       
        result.loc[idx] = not full_coverage
    
    return result
