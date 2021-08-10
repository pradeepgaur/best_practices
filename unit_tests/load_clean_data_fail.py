import pandas as pd

def load_clean_data(file_path):
    df = pd.read_csv(file_path) 
    return df

# writing tests to ensure loaded data is in expected format.

FILE_PATH = "/content/sample_data/california_housing_train.csv"

def test_missing_values_load_clean_data_california():
    df = load_clean_data(FILE_PATH)
    
    #longitude and latitude should not have missing values
    assert df['longitude'].isna().sum() < 1
    assert df['latitude'].isna().sum() < 1

def test_data_shape_load_clean_data_california():
    df = load_clean_data(FILE_PATH)
    
    # testing for any missing or additional columns
    assert df.shape[1] == 10

    # checking if label present 
    assert 'median_house_value' in df.columns
    



