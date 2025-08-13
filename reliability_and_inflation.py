import pandas as pd
import re

only_alphanum = re.compile(r'[\W_]+')  #regular expression pattern to remove any non-alphanumeric characters

def adjust_for_inflation(df, year_col, col, target_year):
    """
    df is a DataFrame
    year_col is a column consisting of years
    col is a column consisting of dollar values in year_col dollars
    target_year is a year

    The function adds a new column to df consisting of col values in target_year dollars
    """
    inflation = pd.read_csv('inflation_data.csv', index_col = 'year')
    inflation = inflation.to_dict()
    inflation = {year:inflation['amount'][target_year]/inflation['amount'][year] for year in inflation['amount']}  #ratio to multiply with to get price in target_year dollars
    df['adjusted_'+col] = df.apply(lambda x: x[col]*inflation[x[year_col]], axis = 1)


def build_reliability_dict():
    reliability_ratings = {}
    years_included = [2019,2020,2021,2022,2023,2024]
    ind_avg_by_year = {2019: 136, 2020:134, 2021: 121, 2022: 192, 2023: 186, 2024: 190}
    ratings_by_year = [pd.read_csv("reliability"+str(year)+".csv") for year in years_included]
    for i in range(len(years_included)):
        ratings_by_year[i]["Score"] = (ind_avg_by_year[years_included[i]] - ratings_by_year[i]["Score"])/ratings_by_year[i]["Score"].std()
    jd_year_ratings = pd.concat(ratings_by_year).reset_index(drop=True)
    jd_year_ratings["Brand"] = jd_year_ratings["Brand"].apply(lambda x: only_alphanum.sub("", x).lower())
    for i in jd_year_ratings.index:
        brand = jd_year_ratings["Brand"][i]
        if brand in reliability_ratings:
            reliability_ratings[brand][0]+= jd_year_ratings["Score"][i]
            reliability_ratings[brand][1]+=1
        else:
            reliability_ratings[brand] = [jd_year_ratings["Score"][i], 1]
    for brand in reliability_ratings:
        reliability_ratings[brand] = reliability_ratings[brand][0] / reliability_ratings[brand][1]
    return reliability_ratings

def get_reliability(brand, reliability_dict):
    brand = only_alphanum.sub("", brand).lower()
    if brand == "land": brand = "landrover"
    return reliability_dict[brand]  if brand in reliability_dict else 0
