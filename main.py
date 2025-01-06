import os
import time
import json
from dotenv import load_dotenv
from google.cloud import bigquery
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import re
import warnings 
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from gspread_dataframe import set_with_dataframe
warnings.filterwarnings('ignore')


load_dotenv()



def adjust_value(value):
    percentage_pattern = re.compile(r'^\d+(\.\d+)?%$')
    currency_pattern = re.compile(r'^\$([\d,]+(\.\d{1,2})?)$')
    str_value = str(value)
    if percentage_pattern.match(str_value):
        return float(str_value.strip('%')) / 100
    elif currency_pattern.match(str_value):
        numeric_value = float(str_value.replace('$', '').replace(',', ''))
        return numeric_value
    else:
        try:
            return float(str_value.replace(',', ''))
        except ValueError:
            return value
        

def adjust_data_partions_rankings_df(df, data_partions_rankings_df):
    valid_partitions = df['partion'].unique()
    
    mask_keep = data_partions_rankings_df['partion'].isin(valid_partitions)
    df_keep = data_partions_rankings_df[mask_keep].copy()
    df_removed = data_partions_rankings_df[~mask_keep].copy()

    if len(df_keep) == len(data_partions_rankings_df):
        return data_partions_rankings_df

    weight_removed = df_removed['weight'].sum()
    current_total_weight = df_keep['weight'].sum()

    if current_total_weight > 0:
        df_keep['weight'] = df_keep['weight'] + (
            df_keep['weight'] / current_total_weight
        ) * weight_removed

    df_keep = df_keep.sort_values(by='rank')
    df_keep['rank'] = range(1, len(df_keep) + 1)

    data_partions_rankings_df = df_keep.reset_index(drop=True)

    total_weight = data_partions_rankings_df['weight'].sum()
    if abs(total_weight - 1.0) > 1e-9:
        data_partions_rankings_df['weight'] = (data_partions_rankings_df['weight'] / total_weight)

    return data_partions_rankings_df


def adjust_variable_rankings_df(df, variable_rankings_df):
    mask_keep = []
    for col in variable_rankings_df['column_name']:
        if df[col].sum() != 0:
            mask_keep.append(True)
        else:
            mask_keep.append(False)
    
    df_keep = variable_rankings_df[mask_keep].copy()
    df_removed = variable_rankings_df[[not x for x in mask_keep]].copy()

    if len(df_keep) == len(variable_rankings_df):
        return variable_rankings_df

    weight_removed = df_removed['weight'].sum()
    current_total_weight = df_keep['weight'].sum()

    if current_total_weight > 0:
        df_keep['weight'] = df_keep['weight'] + (df_keep['weight'] / current_total_weight) * weight_removed

    df_keep = df_keep.sort_values(by='rank')
    df_keep['rank'] = range(1, len(df_keep) + 1)

    variable_rankings_df = df_keep.reset_index(drop=True)
    total_weight = variable_rankings_df['weight'].sum()
    if abs(total_weight - 1.0) > 1e-9:
        variable_rankings_df['weight'] = (variable_rankings_df['weight'] / total_weight)

    return variable_rankings_df

def get_treeData_df(df, 
                    dates_df, 
                    variable_rankings_df, 
                    date_fields_rankings_df, 
                    data_partions_rankings_df, 
                    variable_scoring_df):
    
    if df['order_date'].dtype != 'datetime64[ns]':
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    
    date_columns = ['start_date', 'end_date', 'start_date_comparison', 'end_date_comparison']
    for col in date_columns:
        if dates_df[col].dtype != 'datetime64[ns]':
            dates_df[col] = pd.to_datetime(dates_df[col], errors='coerce')

    cross_df = (
        dates_df.assign(key=1)
        .merge(variable_rankings_df.assign(key=1), on='key', suffixes=('', '_var'))
        .drop(columns=['key'])
    )

    filas_resultantes = []

    for _, row in cross_df.iterrows():
        date_field = row['date_field']
        start_date = row['start_date']
        end_date = row['end_date']
        start_comp = row['start_date_comparison']
        end_comp = row['end_date_comparison']

        column_name = row['column_name']  
        weight_variable_rankings = row['weight'] 

        df_current = df.loc[
            (df['order_date'] >= start_date) &
            (df['order_date'] <= end_date)
        ]


        df_comparison = df.loc[
            (df['order_date'] >= start_comp) &
            (df['order_date'] <= end_comp)
        ]

        current_agg = (
            df_current
            .groupby('partion', as_index=False)[column_name]
            .sum()
            .rename(columns={column_name: 'currentvalue'})
        )

        comparison_agg = (
            df_comparison
            .groupby('partion', as_index=False)[column_name]
            .sum()
            .rename(columns={column_name: 'comparison_value'})
        )

        merged_agg = pd.merge(current_agg, comparison_agg, on='partion', how='outer').fillna(0)

        # Calcular el cambio porcentual
        def calc_change(row_vals):
            comp_val = row_vals['comparison_value']
            curr_val = row_vals['currentvalue']
            if comp_val == 0:
                return 1
            else:
                return ((curr_val - comp_val) / comp_val) 

        merged_agg['change'] = merged_agg.apply(calc_change, axis=1)
        merged_agg['variable'] = column_name
        merged_agg['date_field'] = date_field
        merged_agg['weight_variable_rankings'] = weight_variable_rankings
        filas_resultantes.append(merged_agg)

    result_df = pd.concat(filas_resultantes, ignore_index=True)

    result_df = result_df.merge(
        date_fields_rankings_df[['date_field', 'weight']],
        on='date_field',
        how='left'
    ).rename(columns={'weight': 'weight_date_fields'})


    result_df = result_df.merge(
        data_partions_rankings_df[['partion', 'weight']],
        on='partion',
        how='left'
    ).rename(columns={'weight': 'weight_data_partions'})


    def get_grade_and_eval(row_vals):
        var = row_vals['variable']
        chg = row_vals['change']
        subset = variable_scoring_df[variable_scoring_df['column_name'] == var]
        
        if subset.empty or chg is None:
            return pd.Series([None, None])

        for _, scoring_row in subset.iterrows():
            low = scoring_row['low_interval']
            high = scoring_row['high_interval']
            if low <= chg < high:
                return pd.Series([scoring_row['grade']])
        
        return pd.Series([None, None])

    result_df[['grade']] = result_df.apply(get_grade_and_eval, axis=1)

    columnas_orden = [
        'variable', 'date_field', 'partion',
        'currentvalue', 'comparison_value', 'change',
        'weight_variable_rankings', 'weight_date_fields', 'weight_data_partions',
        'grade'
    ]

    grade_mapping = {
    'A+': 100,
    'A' : 95,
    'A-': 90,
    'B+': 87,
    'B' : 83,
    'B-': 80,
    'C+': 77,
    'C' : 73,
    'C-': 70,
    'D+': 67,
    'D' : 63,
    'D-': 60,
    'F' : 50}

    columnas_finales = [col for col in columnas_orden if col in result_df.columns] + \
                       [col for col in result_df.columns if col not in columnas_orden]
    
    result_df = result_df[columnas_finales]


    filtered_df = result_df.loc[
    result_df['weight_variable_rankings'].isnull() |
    result_df['weight_date_fields'].isnull() |
    result_df['weight_data_partions'].isnull()
    ]

    if len(filtered_df) > 0:
        print("There are missing weights in the data")

    result_df['grade_numeric'] = result_df['grade'].map(grade_mapping)
    result_df['weight'] = result_df['weight_variable_rankings'] * result_df['weight_date_fields'] * result_df['weight_data_partions']
    result_df["math_eval"] = result_df["grade_numeric"] * result_df["weight"] 
    result_df.drop(columns=['grade_numeric', 'weight_variable_rankings' , 'weight_date_fields', 'weight_data_partions'], inplace=True)

    return result_df

def main():
    print("Starting the process")
    project_id = os.environ.get('PROJECT_ID')
    dataset_id = os.environ.get('DATASET_ID')


    gbq_service_account_info = json.loads(os.environ.get('GBQ_SERVICE_ACCOUNT') , strict=False)
    gbq_client = bigquery.Client.from_service_account_info(gbq_service_account_info)

    google_sheets_credentials = json.loads(os.environ.get('GOOGLE_SHEETS_SERVICE_ACCOUNT'))
    sheet_url = os.environ.get('SPREADSHEET_URL')


    scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/spreadsheets'
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_dict(google_sheets_credentials, scope)
    client_spreed_sheet = gspread.authorize(creds)
    sheet = client_spreed_sheet.open_by_url(sheet_url)

    variable_rankings = sheet.worksheet("variable_rankings").get_all_values()
    variable_rankings_df = pd.DataFrame(variable_rankings[1:], columns=variable_rankings[0])
    variable_rankings_df = variable_rankings_df.applymap(adjust_value)
    variable_rankings_df["weight"] = variable_rankings_df["weight"].astype(float)
    variable_rankings_df["rank"] = variable_rankings_df["rank"].astype(int)

    date_fields_rankings = sheet.worksheet("date_fields_rankings").get_all_values()
    date_fields_rankings_df = pd.DataFrame(date_fields_rankings[1:], columns=date_fields_rankings[0])
    date_fields_rankings_df = date_fields_rankings_df.applymap(adjust_value)
    date_fields_rankings_df["weight"] = date_fields_rankings_df["weight"].astype(float)
    date_fields_rankings_df["rank"] = date_fields_rankings_df["rank"].astype(int)

    data_partions_rankings = sheet.worksheet("data_partions_rankings").get_all_values()
    data_partions_rankings_df = pd.DataFrame(data_partions_rankings[1:], columns=data_partions_rankings[0])
    data_partions_rankings_df = data_partions_rankings_df.applymap(adjust_value)
    data_partions_rankings_df["weight"] = data_partions_rankings_df["weight"].astype(float)
    data_partions_rankings_df["rank"] = data_partions_rankings_df["rank"].astype(int)

    variable_scoring = sheet.worksheet("variable_scoring").get_all_values()
    variable_scoring_df = pd.DataFrame(variable_scoring[1:], columns=variable_scoring[0])
    variable_scoring_df = variable_scoring_df.applymap(adjust_value)
    variable_scoring_df["low_interval"] = variable_scoring_df["low_interval"].astype(float)
    variable_scoring_df["high_interval"] = variable_scoring_df["high_interval"].astype(float)
    variable_scoring_df["math_eval"] = variable_scoring_df["math_eval"].astype(float)

    input_df = sheet.worksheet("input_df").get_all_values()
    input_df = pd.DataFrame(input_df[1:], columns=input_df[0])
    input_df = input_df.applymap(adjust_value)

    client_id = input_df["client_id"][0]
    input_date = input_df["input_date"][0]
    print("data gotten from the sheet")
    dates_df = generate_comparison_table(input_date)

    df = gbq_client.query(f"select * from `{project_id}.{dataset_id}.vs_revenue_tree_customer_sale_type` where client_id = '{client_id}'").to_dataframe()
    print("data gotten from the bigquery")
    df["partion"] = df["customer_type"] + " " + df["target_type"]
    data_partions_rankings_df = adjust_data_partions_rankings_df(df, data_partions_rankings_df)
    variable_rankings_df = adjust_variable_rankings_df(df, variable_rankings_df) 
    df_treeData = get_treeData_df(df, dates_df, variable_rankings_df, date_fields_rankings_df, data_partions_rankings_df, variable_scoring_df)
    treeMath_score = df_treeData["math_eval"].sum()

    results = {'treeName':'revenue',
                'treeMath_score': treeMath_score,
                'treeData': df_treeData.to_dict(orient='records')}
    

    df_treeData["client_id"] = client_id

    creds = ServiceAccountCredentials.from_json_keyfile_dict(google_sheets_credentials, scope)
    client_spreed_sheet = gspread.authorize(creds)
    sheet = client_spreed_sheet.open_by_url(sheet_url)

    worksheet = sheet.worksheet('results')
    worksheet.clear()

    set_with_dataframe(
        worksheet,
        df_treeData,
        row=1,
        col=1,
        include_index=False,
        include_column_header=True
    )


    worksheet = sheet.worksheet('results_dates')
    worksheet.clear()

    set_with_dataframe(
        worksheet,
        dates_df,
        row=1,
        col=1,
        include_index=False,
        include_column_header=True
    )

    return '200'


def generate_comparison_table(input_date_str: str):

    current_date = pd.to_datetime(input_date_str)
    
    def fmt(date_obj):
        return date_obj.strftime('%Y-%m-%d')
    
    prev_date = current_date - timedelta(days=1)
    prev_year_date = current_date - relativedelta(years=1)
    prev_2year_date = current_date - relativedelta(years=2)
    
    start_of_current_month = current_date.replace(day=1)
    end_of_prev_month = start_of_current_month - timedelta(days=1)
    start_of_prev_month = end_of_prev_month.replace(day=1)

    def shift_day_with_clamp(base_date, target_day):

        year = base_date.year
        month = base_date.month
        
        try:
            new_date = base_date.replace(day=target_day)
        except ValueError:

            new_date = (base_date.replace(day=1) 
                        + relativedelta(months=1) 
                        - relativedelta(days=1))
        return new_date
    

    end_of_prev_month_equivalent = shift_day_with_clamp(start_of_prev_month, current_date.day)
    start_of_last_full_month = start_of_prev_month
    end_of_last_full_month = end_of_prev_month
    
    start_of_month_before_that = start_of_last_full_month - relativedelta(months=1)
    end_of_month_before_that = start_of_month_before_that.replace(day=1) + relativedelta(months=1, days=-1)


    current_quarter = (current_date.month - 1) // 3 + 1
    start_of_current_quarter_month = 3 * (current_quarter - 1) + 1
    start_of_current_quarter = current_date.replace(
        month=start_of_current_quarter_month, 
        day=1
    )

    start_of_previous_quarter = start_of_current_quarter - relativedelta(months=3)
    end_of_previous_quarter_equivalent = shift_day_with_clamp(start_of_previous_quarter, current_date.day)
    
    start_of_current_year = current_date.replace(month=1, day=1)
    start_of_prev_year = start_of_current_year - relativedelta(years=1)
    start_of_prev_2year = start_of_current_year - relativedelta(years=2)
    
    
    def get_last_completed_week_range(reference_date: pd.Timestamp):

        last_completed_sunday = reference_date - timedelta(days=(reference_date.weekday() + 1))
        last_completed_monday = last_completed_sunday - timedelta(days=6)
        return (last_completed_monday, last_completed_sunday)
    

    wow_current_start, wow_current_end = get_last_completed_week_range(current_date)
    wow_prev_start = wow_current_start - timedelta(days=7)
    wow_prev_end = wow_current_end - timedelta(days=7)
    
    def shift_week_by_years(week_start, week_end, years):
        return (week_start - relativedelta(years=years),
                week_end - relativedelta(years=years))
    
    wow_current_start_py, wow_current_end_py = shift_week_by_years(wow_current_start, wow_current_end, 1)
    wow_current_start_p2y, wow_current_end_p2y = shift_week_by_years(wow_current_start, wow_current_end, 2)

    rows = []

    rows.append({
        "date_field": "DoD PD",
        "start_date": fmt(current_date),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(prev_date),
        "end_date_comparison": fmt(prev_date)
    })


    rows.append({
        "date_field": "DoD PY",
        "start_date": fmt(current_date),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(prev_year_date),
        "end_date_comparison": fmt(prev_year_date)
    })

    rows.append({
        "date_field": "DoD P2Y",
        "start_date": fmt(current_date),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(prev_2year_date),
        "end_date_comparison": fmt(prev_2year_date)
    })
    

    rows.append({
        "date_field": "MTD PM",
        "start_date": fmt(start_of_current_month),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(start_of_prev_month),
        "end_date_comparison": fmt(end_of_prev_month_equivalent)
    })


    rows.append({
        "date_field": "MTD PY",
        "start_date": fmt(start_of_current_month),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(start_of_current_month - relativedelta(years=1)),
        "end_date_comparison": fmt(current_date - relativedelta(years=1))
    })

    # MTD P2Y
    rows.append({
        "date_field": "MTD P2Y",
        "start_date": fmt(start_of_current_month),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(start_of_current_month - relativedelta(years=2)),
        "end_date_comparison": fmt(current_date - relativedelta(years=2))
    })

    rows.append({
        "date_field": "MoM PM",
        "start_date": fmt(start_of_last_full_month),
        "end_date": fmt(end_of_last_full_month),
        "start_date_comparison": fmt(start_of_month_before_that),
        "end_date_comparison": fmt(end_of_month_before_that)
    })

    rows.append({
        "date_field": "MoM PY",
        "start_date": fmt(start_of_last_full_month),
        "end_date": fmt(end_of_last_full_month),
        "start_date_comparison": fmt(start_of_last_full_month - relativedelta(years=1)),
        "end_date_comparison": fmt(end_of_last_full_month - relativedelta(years=1))
    })


    rows.append({
        "date_field": "MoM P2Y",
        "start_date": fmt(start_of_last_full_month),
        "end_date": fmt(end_of_last_full_month),
        "start_date_comparison": fmt(start_of_last_full_month - relativedelta(years=2)),
        "end_date_comparison": fmt(end_of_last_full_month - relativedelta(years=2))
    })

    rows.append({
        "date_field": "QTD PQ",
        "start_date": fmt(start_of_current_quarter),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(start_of_previous_quarter),
        "end_date_comparison": fmt(end_of_previous_quarter_equivalent)
    })

    rows.append({
        "date_field": "QTD PY",
        "start_date": fmt(start_of_current_quarter),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(start_of_current_quarter - relativedelta(years=1)),
        "end_date_comparison": fmt(current_date - relativedelta(years=1))
    })

    rows.append({
        "date_field": "QTD P2Y",
        "start_date": fmt(start_of_current_quarter),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(start_of_current_quarter - relativedelta(years=2)),
        "end_date_comparison": fmt(current_date - relativedelta(years=2))
    })

    rows.append({
        "date_field": "YTD PY",
        "start_date": fmt(start_of_current_year),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(start_of_prev_year),
        "end_date_comparison": fmt(current_date - relativedelta(years=1))
    })

    rows.append({
        "date_field": "YTD P2Y",
        "start_date": fmt(start_of_current_year),
        "end_date": fmt(current_date),
        "start_date_comparison": fmt(start_of_prev_2year),
        "end_date_comparison": fmt(current_date - relativedelta(years=2))
    })

    rows.append({
        "date_field": "WoW PW",
        "start_date": fmt(wow_current_start),
        "end_date": fmt(wow_current_end),
        "start_date_comparison": fmt(wow_prev_start),
        "end_date_comparison": fmt(wow_prev_end)
    })

    wow_py_start, wow_py_end = wow_current_start_py, wow_current_end_py
    rows.append({
        "date_field": "WoW PY",
        "start_date": fmt(wow_current_start),
        "end_date": fmt(wow_current_end),
        "start_date_comparison": fmt(wow_py_start),
        "end_date_comparison": fmt(wow_py_end)
    })


    wow_p2y_start, wow_p2y_end = wow_current_start_p2y, wow_current_end_p2y
    rows.append({
        "date_field": "WoW P2Y",
        "start_date": fmt(wow_current_start),
        "end_date": fmt(wow_current_end),
        "start_date_comparison": fmt(wow_p2y_start),
        "end_date_comparison": fmt(wow_p2y_end)
    })


    df = pd.DataFrame(rows, columns=[
        "date_field",
        "start_date",
        "end_date",
        "start_date_comparison",
        "end_date_comparison"
    ])

    return df


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        message = (
            f"Task #1 of downloading reports job for attempt #1 failed: {str(err)}"
        )
        print(json.dumps({"message": message, "severity": "ERROR"}))