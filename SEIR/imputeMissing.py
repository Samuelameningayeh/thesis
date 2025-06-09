import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def impute_value(row = any, delay_by_district = any, overall_delay = float, df=any):
    
    # check if there is a value this column else check if there is a value in the next
    if pd.notna(row["DateOnset"]):
        print(f'Value already exist...!')
        return row["DateOnset"]
    elif pd.notna(row["DateOnsetInferred"]):
        print(f'Imputing missing values with DateOnsetInferred...!')
        return row["DateOnsetInferred"]
    elif pd.notna(row["DateReport"]):
        print(f'Imputing missing values with DateReport...!')
        country = row["Country"]
        district = row["CL_DistrictRes"]
        delay = delay_by_district.get((country, district), overall_delay)
        return row["DateReport"] - pd.Timedelta(days=delay)
    else:
        print(f'Imputing missing values with District median...!')
        country = row["Country"]
        quarter = row["QuarterOnsetInferred"]
        district = row["CL_DistrictRes"]

        # impute using the median from the same district, country, and quarter
        district_country_quarter_data = df[
            (df["Country"] == country) &
            (df["QuarterOnsetInferred"] == quarter) &
            (df["CL_DistrictRes"] == district) &
            (df["DateOnset"].notna())
        ]
        
        if not district_country_quarter_data.empty:
            return district_country_quarter_data["DateOnset"].median()

        # Else use random sampling from the same country and quarter
        # print(f'Imputing missing values with random sample from Country...!')
        # country_quarter_data = df[
        #     (df["Country"] == country) &
        #     (df["QuarterOnsetInferred"] == quarter) &
        #     (df["DateOnset"].notna())
        # ]
        # if not country_quarter_data.empty:
        #     return country_quarter_data["DateOnset"].sample(1).iloc[0]

        # Else use the median from the entire country
        print(f'Imputing missing values with Country median...!')
        country_data = df[
            (df["Country"] == country) &
            (df["DateOnset"].notna())
        ]
        if not country_data.empty:
            return country_data["DateOnset"].median()
        return pd.NaT
print(f'Imputation Done!!!')