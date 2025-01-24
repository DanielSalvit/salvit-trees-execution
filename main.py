import os
import time
import json
from dotenv import load_dotenv
from google.cloud import bigquery
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import re
import warnings 
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from gspread_dataframe import set_with_dataframe
warnings.filterwarnings('ignore')


load_dotenv()




def revenue_tree(
    gbq_client, 
    project_id, 
    dataset_id, 
    client_id,  
    date_str,  
    google_sheets_credentials,
    sheet_url,
    agent,
    filters
):
    print('starting revenue tree process')
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
    variable_rankings_df = variable_rankings_df.applymap(adjust_value_spreedsheet)
    variable_rankings_df["weight"] = variable_rankings_df["weight"].astype(float)
    variable_rankings_df["rank"] = variable_rankings_df["rank"].astype(int)

    date_fields_rankings = sheet.worksheet("date_fields_rankings").get_all_values()
    date_fields_rankings_df = pd.DataFrame(date_fields_rankings[1:], columns=date_fields_rankings[0])
    date_fields_rankings_df = date_fields_rankings_df.applymap(adjust_value_spreedsheet)
    date_fields_rankings_df["weight"] = date_fields_rankings_df["weight"].astype(float)
    date_fields_rankings_df["rank"] = date_fields_rankings_df["rank"].astype(int)

    data_partitions_rankings = sheet.worksheet("data_partitions_rankings").get_all_values()
    data_partitions_rankings_df = pd.DataFrame(data_partitions_rankings[1:], columns=data_partitions_rankings[0])
    data_partitions_rankings_df = data_partitions_rankings_df.applymap(adjust_value_spreedsheet)
    data_partitions_rankings_df["weight"] = data_partitions_rankings_df["weight"].astype(float)
    data_partitions_rankings_df["rank"] = data_partitions_rankings_df["rank"].astype(int)

    variable_scoring = sheet.worksheet("variable_scoring").get_all_values()
    variable_scoring_df = pd.DataFrame(variable_scoring[1:], columns=variable_scoring[0])
    variable_scoring_df = variable_scoring_df.applymap(adjust_value_spreedsheet)
    variable_scoring_df["low_interval"] = variable_scoring_df["low_interval"].astype(float)
    variable_scoring_df["high_interval"] = variable_scoring_df["high_interval"].astype(float)
    print('spreadsheet data loaded')

    if not filters:
        query = f"""
        SELECT * 
        FROM `{project_id}.{dataset_id}.vs_revenue_tree_customer_sale_type` 
        WHERE client_id = '{client_id}'
        """
    else:
        query = f""" 
        SELECT client_id,
        order_date,
        customer_type,
        target_type, 
        sum(order_revenue) as order_revenue,
        sum(costs_of_goods) as costs_of_goods,
        sum(total_costs) as total_costs,
        sum(order_outbound_cost) as order_outbound_cost,
        sum(order_discount) as order_discount,
        sum(item_quantity_with_revenue) as item_quantity_with_revenue,
        sum(num_orders) as num_orders,
        sum(num_orders_with_revenue) as num_orders_with_revenue,
        sum(order_refund) as order_refund,
        sum(order_total_shipping_costs) as order_total_shipping_costs,
        sum(item_quantity) as item_quantity,
        sum(order_shipping_price) as order_shipping_price,
        sum(order_refund_shipping_costs) as order_refund_shipping_costs
        FROM `{project_id}.{dataset_id}.vs_revenue_tree_customer_sale_type_with_filters` 
        WHERE client_id = '{client_id}'
        """

        query = add_filters_to_query(query, filters)
        query += "\n GROUP BY 1, 2, 3, 4"


    df = gbq_client.query(query).to_dataframe()
    df["partition"] = df["customer_type"] + " " + df["target_type"]

    # Asegurar que 'order_date' es datetime
    if df['order_date'].dtype != 'datetime64[ns]':
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # 2. Ajustar dataframes de particiones y variables según datos disponibles
    print('Ajustando dataframes de particiones y variables')
    data_partitions_rankings_df = adjust_revenue_data_partitions_rankings_df(df, data_partitions_rankings_df)
    print('Data partitions rankings ajustado')
    variable_rankings_df = adjust_revenue_variable_rankings_df(df, variable_rankings_df)
    print('Variable rankings ajustado')

    if agent == 'agent_1':
        dates_df = generate_comparison_table(date_str)
        dates_df = dates_df.loc[dates_df["date_field"].isin(date_fields_rankings_df["date_field"].unique())] 
        df_treeData = get_revenue_treeData_df(df, dates_df, variable_rankings_df, date_fields_rankings_df, data_partitions_rankings_df, variable_scoring_df)
        treeMath_score = df_treeData["math_eval"].sum()

        results = {
        'treeName': 'revenue',
        'treeMath_score': treeMath_score,
        'treeData': df_treeData.to_dict(orient='records'),
        'dates_df': dates_df}
        
        return results



    base_date = pd.to_datetime(date_str)
    previous_dates = [(base_date - pd.DateOffset(days=i)).strftime('%Y-%m-%d')  for i in range(0, 29)]

   
    def process_date(current_date_str):
        current_dates_df = generate_comparison_table(current_date_str)
        tree_data_df = get_revenue_treeData_df(
            df, 
            current_dates_df, 
            variable_rankings_df, 
            date_fields_rankings_df, 
            data_partitions_rankings_df, 
            variable_scoring_df
        )
        return current_date_str, tree_data_df

    num_cores = multiprocessing.cpu_count()
    processed_results = Parallel(n_jobs=num_cores)(delayed(process_date)(current_date_str) for current_date_str in previous_dates)


    treeData_dict = {}
    change_dfs = []
    for current_date_str, tree_data_df in processed_results:
        if current_date_str == date_str:
            df_treeData = tree_data_df
            dates_df = generate_comparison_table(current_date_str)  
        else:
            change_dfs.append(tree_data_df)


    change_df = pd.concat(change_dfs, ignore_index=True)
    change_df = (
        change_df
        .groupby(['variable', 'date_field', 'partition'], as_index=False)
        .agg({'change': 'mean'})
        .reset_index(drop=True)
        )

    treeMath_score = df_treeData["math_eval"].sum()

    results = {
        'treeName': 'revenue',
        'treeMath_score': treeMath_score,
        'treeData': df_treeData.to_dict(orient='records'),
        'dates_df': dates_df,
        'change_df': change_df.to_dict(orient='records') 
    }

    return results




def adjust_revenue_data_partitions_rankings_df(df, data_partitions_rankings_df):

    valid_partitions = df['partition'].unique()

    mask_keep = data_partitions_rankings_df['partition'].isin(valid_partitions)
    df_keep = data_partitions_rankings_df[mask_keep].copy()
    df_removed = data_partitions_rankings_df[~mask_keep].copy()

    if len(df_keep) == len(data_partitions_rankings_df):
        return data_partitions_rankings_df

    # Redistribuir pesos de las particiones removidas
    weight_removed = df_removed['weight'].sum()
    current_total_weight = df_keep['weight'].sum()

    if current_total_weight > 0:
        df_keep['weight'] += (df_keep['weight'] / current_total_weight) * weight_removed


    df_keep = df_keep.sort_values(by='rank')
    df_keep['rank'] = range(1, len(df_keep) + 1)

    data_partitions_rankings_df = df_keep.reset_index(drop=True)

    total_weight = data_partitions_rankings_df['weight'].sum()
    if abs(total_weight - 1.0) > 1e-9:
        data_partitions_rankings_df['weight'] = data_partitions_rankings_df['weight'] / total_weight

    return data_partitions_rankings_df

def adjust_revenue_variable_rankings_df(df, variable_rankings_df):
    mask_keep = []
    special_variables = ['aov', 'average_items_per_order', 'discount_rate', 'no_charge_ratio']

    for col in variable_rankings_df['column_name']:
        if col in special_variables:
            mask_keep.append(True)
        else:
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



def get_revenue_treeData_df(
    df, 
    dates_df, 
    variable_rankings_df, 
    date_fields_rankings_df, 
    data_partitions_rankings_df, 
    variable_scoring_df
):

    date_columns = ['start_date', 'end_date', 'start_date_comparison', 'end_date_comparison']
    for col in date_columns:
        if dates_df[col].dtype != 'datetime64[ns]':
            dates_df[col] = pd.to_datetime(dates_df[col], errors='coerce')

    # Crear columnas para filtrar una sola vez
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Crear todas las combinaciones de fechas y variables
    cross_df = (
        dates_df.assign(key=1)
        .merge(variable_rankings_df.assign(key=1), on='key', suffixes=('', '_var'))
        .drop(columns=['key'])
    )

    # Variables especiales y sus funciones
    special_vars = {
        'aov': lambda df_: np.where(
            df_['num_orders_with_revenue'] == 0, 
            0, 
            df_['order_revenue'] / df_['num_orders_with_revenue']
        ),
        'average_items_per_order': lambda df_: np.where(
            df_['num_orders_with_revenue'] == 0,
            0,
            df_['item_quantity_with_revenue'] / df_['num_orders_with_revenue']
        ),
        'discount_rate': lambda df_: np.where(
            (df_['order_discount'] + df_['order_revenue']) == 0,
            0,
            df_['order_discount'] / (df_['order_discount'] + df_['order_revenue'])
        ),
        'no_charge_ratio': lambda df_: np.where(
            df_['item_quantity'] == 0,
            0,
            (df_['item_quantity'] - df_['item_quantity_with_revenue']) / df_['item_quantity']
        )
    }

    # Columns needed for special variables
    columns_needed_for_special = [
        'order_revenue',
        'num_orders_with_revenue',
        'item_quantity_with_revenue',
        'order_discount',
        'item_quantity'
    ]

    rows = []

    for _, row in cross_df.iterrows():
        date_field = row['date_field']
        start_date = row['start_date']
        end_date = row['end_date']
        start_comp = row['start_date_comparison']
        end_comp = row['end_date_comparison']

        column_name = row['column_name']
        weight_variable_rankings = row['weight']

        # Filtrar para período actual y de comparación
        df_current = df.loc[(df['order_date'] >= start_date) & (df['order_date'] <= end_date)]
        df_comparison = df.loc[(df['order_date'] >= start_comp) & (df['order_date'] <= end_comp)]

        # Agrupar 'current'
        if column_name in special_vars:
            current_agg = df_current.groupby('partition', as_index=False)[columns_needed_for_special].sum()
            current_agg['currentvalue'] = special_vars[column_name](current_agg)
            current_agg = current_agg[['partition', 'currentvalue']]
            # Agregar fila 'all'
            df_current_sum = df_current[columns_needed_for_special].sum()
            df_current_sum_df = pd.DataFrame([df_current_sum])
            df_current_sum_df['currentvalue'] = special_vars[column_name](df_current_sum_df)
            all_current_value = df_current_sum_df['currentvalue'].iloc[0]
            current_agg = pd.concat([
                current_agg,
                pd.DataFrame({'partition': ['all'], 'currentvalue': [all_current_value]})
            ], ignore_index=True)
        else:
            current_agg = (
                df_current
                .groupby('partition', as_index=False)[column_name]
                .sum()
                .rename(columns={column_name: 'currentvalue'})
            )
            all_current_value = df_current[column_name].sum()
            current_agg = pd.concat([
                current_agg,
                pd.DataFrame({'partition': ['all'], 'currentvalue': [all_current_value]})
            ], ignore_index=True)

        # Agrupar 'comparison'
        if column_name in special_vars:
            comparison_agg = df_comparison.groupby('partition', as_index=False)[columns_needed_for_special].sum()
            comparison_agg['comparison_value'] = special_vars[column_name](comparison_agg)
            comparison_agg = comparison_agg[['partition', 'comparison_value']]
            # Agregar fila 'all'
            df_comparison_sum = df_comparison[columns_needed_for_special].sum()
            df_comparison_sum_df = pd.DataFrame([df_comparison_sum])
            df_comparison_sum_df['comparison_value'] = special_vars[column_name](df_comparison_sum_df)
            all_comp_value = df_comparison_sum_df['comparison_value'].iloc[0]
            comparison_agg = pd.concat([
                comparison_agg,
                pd.DataFrame({'partition': ['all'], 'comparison_value': [all_comp_value]})
            ], ignore_index=True)
        else:
            comparison_agg = (
                df_comparison
                .groupby('partition', as_index=False)[column_name]
                .sum()
                .rename(columns={column_name: 'comparison_value'})
            )
            all_comp_value = df_comparison[column_name].sum()
            comparison_agg = pd.concat([
                comparison_agg,
                pd.DataFrame({'partition': ['all'], 'comparison_value': [all_comp_value]})
            ], ignore_index=True)

        # Merge ambas agregaciones
        merged_agg = pd.merge(current_agg, comparison_agg, on='partition', how='outer').fillna(0)

        # Calcular 'change' con np.where para evitar división por cero
        merged_agg['change'] = np.where(
            merged_agg['comparison_value'] == 0,
            1,
            (merged_agg['currentvalue'] - merged_agg['comparison_value']) / merged_agg['comparison_value']
        )

        # Asignar variables adicionales
        merged_agg['variable'] = column_name
        merged_agg['date_field'] = date_field
        merged_agg['weight_variable_rankings'] = weight_variable_rankings

        # Agregar a la lista de resultados
        rows.append(merged_agg)

    # Concatenar todas las filas
    result_df = pd.concat(rows, ignore_index=True)

    # Merge con los pesos de date_fields
    result_df = result_df.merge(
        date_fields_rankings_df[['date_field', 'weight']], 
        on='date_field',
        how='inner'
    ).rename(columns={'weight': 'weight_date_fields'})

    # Merge con los pesos de data_partitions
    result_df = result_df.merge(
        data_partitions_rankings_df[['partition', 'weight']], 
        on='partition',
        how='left'
    ).rename(columns={'weight': 'weight_data_partitions'})

    # Asignar 'grade' según variable_scoring_df de forma vectorizada
    # Primero, crear bins y labels para cada variable
    scoring_dict = {}
    for var in variable_scoring_df['column_name'].unique():
        subset = variable_scoring_df[variable_scoring_df['column_name'] == var].sort_values('low_interval')
        bins = subset['low_interval'].tolist() + [np.inf]
        labels = subset['grade'].tolist()
        scoring_dict[var] = (bins, labels)

    # Función vectorizada para asignar 'grade'
    def assign_grade(row):
        var = row['variable']
        chg = row['change']
        if var not in scoring_dict or pd.isnull(chg):
            return np.nan
        bins, labels = scoring_dict[var]
        idx = np.searchsorted(bins, chg, side='right') - 1
        if 0 <= idx < len(labels):
            return labels[idx]
        return np.nan

    result_df['grade'] = result_df.apply(assign_grade, axis=1)

    # Orden de columnas
    columnas_orden = [
        'variable', 'date_field', 'partition',
        'currentvalue', 'comparison_value', 'change',
        'weight_variable_rankings', 'weight_date_fields', 'weight_data_partitions',
        'grade'
    ]
    columnas_finales = [
        col for col in columnas_orden if col in result_df.columns
    ] + [
        col for col in result_df.columns if col not in columnas_orden
    ]

    result_df = result_df[columnas_finales]
    result_df.loc[result_df['partition'] == 'all', 'weight_data_partitions'] = 1
    
    # Verificar pesos nulos
    mask_null_weights = (
        result_df['weight_variable_rankings'].isnull() |
        result_df['weight_date_fields'].isnull() |
        result_df['weight_data_partitions'].isnull()
    )
    if mask_null_weights.any():
        # Puedes cambiarlo a un log o warning si lo prefieres.
        print("Existen pesos nulos en la información de result_df.")

    # Llenar nulos con 0
    result_df['weight_variable_rankings'] = result_df['weight_variable_rankings'].fillna(0)
    result_df['weight_date_fields'] = result_df['weight_date_fields'].fillna(0)
    result_df['weight_data_partitions'] = result_df['weight_data_partitions'].fillna(0)

    # Para la partitions "all", forzar weight_data_partitions = 1
    

    # Mapeo de letras a valores numéricos
    grade_mapping = {
        'A+': 100, 'A': 95, 'A-': 90,
        'B+': 87.5, 'B': 85, 'B-': 80,
        'C+': 77.5, 'C': 75, 'C-': 70,
        'D+': 68.5, 'D': 65, 'D-': 62,
        'F': 50
    }

    # Calcular grade numérico y math_eval
    result_df['grade_numeric'] = result_df['grade'].map(grade_mapping)
    result_df['weight'] = (
        result_df['weight_variable_rankings'] *
        result_df['weight_date_fields'] *
        result_df['weight_data_partitions']
    )
    result_df["math_eval"] = result_df["grade_numeric"] * result_df["weight"]

    result_df.drop(
        columns=[
            'grade_numeric', 
            'weight_variable_rankings', 
            'weight_date_fields', 
            'weight_data_partitions'
        ],
        inplace=True
    )

    result_df = result_df.loc[~ ((result_df['currentvalue'] == 0) & (result_df['comparison_value'] != 0))]

    return result_df


def marketing_tree(
    gbq_client, 
    project_id, 
    dataset_id, 
    client_id,  
    date_str,  
    google_sheets_credentials,
    sheet_url,
    agent,
    filters
):
    print('starting marketing tree process')
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
    variable_rankings_df = variable_rankings_df.applymap(adjust_value_spreedsheet)
    variable_rankings_df["weight"] = variable_rankings_df["weight"].astype(float)
    variable_rankings_df["rank"] = variable_rankings_df["rank"].astype(int)

    date_fields_rankings = sheet.worksheet("date_fields_rankings").get_all_values()
    date_fields_rankings_df = pd.DataFrame(date_fields_rankings[1:], columns=date_fields_rankings[0])
    date_fields_rankings_df = date_fields_rankings_df.applymap(adjust_value_spreedsheet)
    date_fields_rankings_df["weight"] = date_fields_rankings_df["weight"].astype(float)
    date_fields_rankings_df["rank"] = date_fields_rankings_df["rank"].astype(int)

    long_term_weighting = sheet.worksheet("long_term_weighting").get_all_values()
    long_term_weighting_df = pd.DataFrame(long_term_weighting[1:], columns=long_term_weighting[0])
    long_term_weighting_df = long_term_weighting_df.applymap(adjust_value_spreedsheet)
    long_term_weighting_df["weight"] = long_term_weighting_df["weight"].astype(float)
    long_term_weighting_df["rank"] = long_term_weighting_df["rank"].astype(int)

    change_variable_scoring = sheet.worksheet("change_variable_scoring").get_all_values()
    change_variable_scoring_df = pd.DataFrame(change_variable_scoring[1:], columns=change_variable_scoring[0])
    change_variable_scoring_df = change_variable_scoring_df.applymap(adjust_value_spreedsheet)
    change_variable_scoring_df["low_interval"] = change_variable_scoring_df["low_interval"].astype(float)
    change_variable_scoring_df["high_interval"] = change_variable_scoring_df["high_interval"].astype(float)

    absolute_variable_scoring = sheet.worksheet("absolute_variable_scoring").get_all_values()
    absolute_variable_scoring_df = pd.DataFrame(absolute_variable_scoring[1:], columns=absolute_variable_scoring[0])
    absolute_variable_scoring_df = absolute_variable_scoring_df.applymap(adjust_value_spreedsheet)
    absolute_variable_scoring_df["low_interval"] = absolute_variable_scoring_df["low_interval"].astype(float)
    absolute_variable_scoring_df["high_interval"] = absolute_variable_scoring_df["high_interval"].astype(float)
    print('spreadsheet data loaded')

    if not filters:

        query = f"""
        SELECT * 
        FROM `{project_id}.{dataset_id}.vs_marketing_tree` 
        WHERE client_id = '{client_id}'
        """
        df = gbq_client.query(query).to_dataframe()

        
        query = f"""
        SELECT marketing_channel, order_date, marketing_spend   
        FROM `{project_id}.{dataset_id}.vs_marketing_tree_channel_mix` 
        WHERE client_id = '{client_id}'
        """
        df_marketing_channel = gbq_client.query(query).to_dataframe()

    else:
        query = f"""
        SELECT client_id,
        order_date,
        sum(order_revenue) order_revenue,
        sum(new_customers) new_customers,
        sum(new_customers_revenue) new_customers_revenue,
        sum(new_customers_subscribers) new_customers_subscribers,
        sum(marketing_spend) marketing_spend,
        sum(marketing_clicks) marketing_clicks,
        sum(marketing_impressions) marketing_impressions
        FROM `{project_id}.{dataset_id}.vs_marketing_tree_with_filters` 
        WHERE client_id = '{client_id}'
        """
        query = add_filters_to_query(query, filters)
        query += "\n GROUP BY 1, 2 "
        sub_df_1 = gbq_client.query(query).to_dataframe()

        if sub_df_1.empty:
            results = {
        'treeName': 'marketing',
        'response_text': "Marketing data not found for the selected filters. No conclusions needed from this tree. Conclude ignoring this tree"
        }
            return results

        if sub_df_1["markeint_spend"].sum() == 0:
            results = {
            'treeName': 'marketing',
            'response_text': "Marketing data not found for the selected filters. No conclusions needed from this tree. Conclude ignoring this tree"
            }
            return results

        query = f"""
        SELECT * 
        FROM `{project_id}.{dataset_id}.vs_marketing_google_analytics` 
        WHERE client_id = '{client_id}'
        """
        sub_df_2 = gbq_client.query(query).to_dataframe()

        if sub_df_2.empty:
            df = sub_df_1
            df["visits"] = 0 
            df["totalPurchasers"] = 0

        else:
            df = pd.merge(sub_df_1, sub_df_2, on=["client_id","order_date"], how = "left" )

        for column in ["order_revenue","new_customers","new_customers_revenue","new_customers_subscribers","marketing_spend","marketing_clicks","marketing_impressions","visits","totalPurchasers"]:
            df[column] = df[column].fillna(0)


        query = f"""
        SELECT marketing_channel, order_date, sum(marketing_spend) marketing_spend   
        FROM `{project_id}.{dataset_id}.vs_marketing_tree_channel_mix_filters` 
        WHERE client_id = '{client_id}'
        """
        query = add_filters_to_query(query, filters)
        query += "\n GROUP BY 1, 2"
        df_marketing_channel = gbq_client.query(query).to_dataframe()


    if df['order_date'].dtype != 'datetime64[ns]':
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    variable_rankings_df = adjust_marketing_variable_rankings_df(df, variable_rankings_df)
    unique_vars = variable_rankings_df["column_name"].unique()
    change_variable_scoring_df = change_variable_scoring_df[change_variable_scoring_df["column_name"].isin(unique_vars)]
    absolute_variable_scoring_df = absolute_variable_scoring_df[absolute_variable_scoring_df["column_name"].isin(unique_vars)]


    if agent == 'agent_1':
        dates_df = generate_comparison_table(date_str)
        dates_df = dates_df.loc[dates_df["date_field"].isin(date_fields_rankings_df["date_field"].unique())] 
        df_treeData = get_marketing_treeData_df(df, df_marketing_channel, dates_df, variable_rankings_df, date_fields_rankings_df, long_term_weighting_df, change_variable_scoring_df, absolute_variable_scoring_df)
        treeMath_score = df_treeData["math_eval"].sum()

        results = {
        'treeName': 'marketing',
        'treeMath_score': treeMath_score,
        'treeData': df_treeData.to_dict(orient='records'),
        'dates_df': dates_df}
        
        return results



def get_marketing_treeData_df(
    df, 
    df_marketing_channel,  # <--- DataFrame con columns: marketing_channel, order_date, marketing_spend
    dates_df, 
    variable_rankings_df, 
    date_fields_rankings_df,
    long_term_weighting_df,  
    change_variable_scoring_df,
    absolute_variable_scoring_df
):

    # =========================================================================
    # 1) Asegurarnos de que las columnas de fechas sean datetime
    # =========================================================================
    date_cols = ['start_date', 'end_date', 'start_date_comparison', 'end_date_comparison']
    for col in date_cols:
        if dates_df[col].dtype != 'datetime64[ns]':
            dates_df[col] = pd.to_datetime(dates_df[col], errors='coerce')

    # Converting order_date en df principal
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['partition'] = 'all'  # Para este ejemplo, solo una partición

    # Converting order_date en df_marketing_channel
    df_marketing_channel['order_date'] = pd.to_datetime(df_marketing_channel['order_date'], errors='coerce')
    df_marketing_channel['partition'] = 'all'

    # =========================================================================
    # 2) Generar las combinaciones de (fechas, variables)
    # =========================================================================
    cross_df = (
        dates_df.assign(key=1)
        .merge(variable_rankings_df.assign(key=1), on='key', suffixes=('', '_var'))
        .drop(columns=['key'])
    )

    # =========================================================================
    # 3) Definir variables especiales y las columnas necesarias
    # =========================================================================
    special_vars = {
        'mer': lambda df_: np.where(
            df_['marketing_spend'] == 0, 
            0, 
            df_['order_revenue'] / df_['marketing_spend']
        ),
        'cpa': lambda df_: np.where(
            df_['new_customers'] == 0,
            0,
            df_['marketing_spend'] / df_['new_customers']
        ),
        'marketing_spend_ratio': lambda df_: np.where(
            df_['order_revenue'] == 0,
            0,
            df_['marketing_spend'] / df_['order_revenue']
        ),
        'roas': lambda df_: np.where(
            df_['marketing_spend'] == 0,
            0,
            df_['new_customers_revenue'] / df_['marketing_spend']
        ),
        'cvr': lambda df_: np.where(
            df_['visits'] == 0,
            0,
            df_['totalPurchasers'] / df_['visits']
        ),
        'cpm': lambda df_: np.where(
            df_['marketing_impressions'] == 0,
            0,
            df_['marketing_spend'] / df_['marketing_impressions']
        ),
        'cpc': lambda df_: np.where(
            df_['marketing_clicks'] == 0,
            0,
            df_['marketing_spend'] / df_['marketing_clicks']
        )
    }

    columns_needed_for_special = [
        'marketing_spend',
        'order_revenue',
        'new_customers',
        'marketing_clicks',
        'marketing_impressions',
        'visits',
        'totalPurchasers',
        'new_customers_revenue'
    ]

    # Si existe la columna new_customers_subscribers, se agrega a las "special_vars"
    if 'new_customers_subscribers' in df.columns:
        new_subs_total = df["new_customers_subscribers"].sum()
        if new_subs_total > 0:
            special_vars["cts"] = lambda df_: np.where(
                df_['new_customers'] == 0,
                0,
                df_['new_customers_subscribers'] / df_['new_customers']
            )
            columns_needed_for_special.append("new_customers_subscribers")

    # =========================================================================
    # 4A) Bucle para métricas normales (MER, CPA, etc.)
    # =========================================================================
    rows_metrics = []

    for _, row_cross in cross_df.iterrows():
        date_field = row_cross['date_field']
        start_date = row_cross['start_date']
        end_date = row_cross['end_date']
        start_comp = row_cross['start_date_comparison']
        end_comp = row_cross['end_date_comparison']

        column_name = row_cross['column_name']
        weight_variable_rankings = row_cross['weight']

        # Filtrar DF principal para actual y comparación
        df_current = df.loc[(df['order_date'] >= start_date) & (df['order_date'] <= end_date)]
        df_comparison = df.loc[(df['order_date'] >= start_comp) & (df['order_date'] <= end_comp)]

        # Calcular currentvalue
        if column_name in special_vars:
            current_agg = df_current.groupby('partition', as_index=False)[columns_needed_for_special].sum()
            current_agg['currentvalue'] = special_vars[column_name](current_agg)
            current_agg = current_agg[['partition', 'currentvalue']]
        else:
            current_agg = (
                df_current
                .groupby('partition', as_index=False)[column_name]
                .sum()
                .rename(columns={column_name: 'currentvalue'})
            )

        # Calcular comparison_value
        if column_name in special_vars:
            comparison_agg = df_comparison.groupby('partition', as_index=False)[columns_needed_for_special].sum()
            comparison_agg['comparison_value'] = special_vars[column_name](comparison_agg)
            comparison_agg = comparison_agg[['partition', 'comparison_value']]
        else:
            comparison_agg = (
                df_comparison
                .groupby('partition', as_index=False)[column_name]
                .sum()
                .rename(columns={column_name: 'comparison_value'})
            )

        # Combinar y calcular 'change'
        merged_agg = pd.merge(current_agg, comparison_agg, on='partition', how='outer').fillna(0)
        merged_agg['change'] = np.where(
            merged_agg['comparison_value'] == 0,
            1,
            (merged_agg['currentvalue'] - merged_agg['comparison_value']) / merged_agg['comparison_value']
        )

        # Guardar columnas clave
        merged_agg['variable'] = column_name
        merged_agg['date_field'] = date_field
        merged_agg['weight_variable_rankings'] = weight_variable_rankings

        rows_metrics.append(merged_agg)

    # =========================================================================
    # 4B) Bucle para spend mix (separado, para no duplicar)
    # =========================================================================
    rows_spend_mix = []

    for _, row_dates in dates_df.iterrows():
        date_field = row_dates['date_field']
        start_date = row_dates['start_date']
        end_date = row_dates['end_date']
        start_comp = row_dates['start_date_comparison']
        end_comp = row_dates['end_date_comparison']

        # Filtrar df_marketing_channel
        df_mc_current = df_marketing_channel.loc[
            (df_marketing_channel['order_date'] >= start_date) &
            (df_marketing_channel['order_date'] <= end_date)
        ]
        df_mc_comparison = df_marketing_channel.loc[
            (df_marketing_channel['order_date'] >= start_comp) &
            (df_marketing_channel['order_date'] <= end_comp)
        ]

        # -- Actual --
        if not df_mc_current.empty:
            cur_mix = df_mc_current.groupby(['partition','marketing_channel'], as_index=False)['marketing_spend'].sum()
            total_spend_cur = cur_mix.groupby('partition', as_index=False)['marketing_spend'].sum()
            total_spend_cur.rename(columns={'marketing_spend': 'total_spend_current'}, inplace=True)

            cur_mix = cur_mix.merge(total_spend_cur, on='partition', how='left')
            cur_mix['currentvalue'] = np.where(
                cur_mix['total_spend_current'] == 0,
                0,
                cur_mix['marketing_spend'] / cur_mix['total_spend_current']
            )
        else:
            # DataFrame vacío con columnas esperadas
            cur_mix = pd.DataFrame(columns=['partition','marketing_channel','marketing_spend','total_spend_current','currentvalue'])

        # -- Comparación --
        if not df_mc_comparison.empty:
            comp_mix = df_mc_comparison.groupby(['partition','marketing_channel'], as_index=False)['marketing_spend'].sum()
            total_spend_comp = comp_mix.groupby('partition', as_index=False)['marketing_spend'].sum()
            total_spend_comp.rename(columns={'marketing_spend': 'total_spend_comparison'}, inplace=True)

            comp_mix = comp_mix.merge(total_spend_comp, on='partition', how='left')
            comp_mix['comparison_value'] = np.where(
                comp_mix['total_spend_comparison'] == 0,
                0,
                comp_mix['marketing_spend'] / comp_mix['total_spend_comparison']
            )
        else:
            comp_mix = pd.DataFrame(columns=['partition','marketing_channel','marketing_spend','total_spend_comparison','comparison_value'])

        # -- Unir actual y comparación --
        mix_merged = pd.merge(
            cur_mix[['partition','marketing_channel','currentvalue']],
            comp_mix[['partition','marketing_channel','comparison_value']],
            on=['partition','marketing_channel'],
            how='outer'
        ).fillna(0)

        # -- Calcular change --
        mix_merged['change'] = np.where(
            mix_merged['comparison_value'] == 0,
            1,
            (mix_merged['currentvalue'] - mix_merged['comparison_value']) / mix_merged['comparison_value']
        )

        # -- Filtrar canales >= 7%
        mask_7 = (mix_merged['currentvalue'] >= 0.07) | (mix_merged['comparison_value'] >= 0.07)
        mix_merged = mix_merged[mask_7].copy()

        # -- Crear filas con variable = "spend_mix_{channel}"
        for _, row_mix in mix_merged.iterrows():
            rows_spend_mix.append({
                'partition': row_mix['partition'],
                'variable': f"spend_mix_{row_mix['marketing_channel']}",
                'date_field': date_field,
                'currentvalue': row_mix['currentvalue'],
                'comparison_value': row_mix['comparison_value'],
                'change': row_mix['change'],
                'weight_variable_rankings': 0  # Se forzará en 0
            })

    # =========================================================================
    # 5) Unir las filas de métricas normales y las de spend mix
    # =========================================================================
    df_metrics = pd.concat(rows_metrics, ignore_index=True).fillna(0)
    df_spend_mix = pd.DataFrame(rows_spend_mix).fillna(0) if rows_spend_mix else pd.DataFrame()

    result_df = pd.concat([df_metrics, df_spend_mix], ignore_index=True).fillna(0)

    # =========================================================================
    # 6) Identificar variables "long term" vs "short term" y hacer merges
    # =========================================================================
    long_term_vars = long_term_weighting_df["column_name"].unique()

    result_df_long_term = result_df[result_df["variable"].isin(long_term_vars)].copy()
    result_df_short_term = result_df[~result_df["variable"].isin(long_term_vars)].copy()

    # -- Short term: merge con date_fields_rankings_df
    result_df_short_term = result_df_short_term.merge(
        date_fields_rankings_df[['date_field', 'weight']], 
        on='date_field',
        how='inner'
    ).rename(columns={'weight': 'weight_date_fields'})

    # -- Long term: merge con long_term_weighting_df
    df_ltw = long_term_weighting_df.rename(columns={'column_name': 'variable'}).copy()
    result_df_long_term = result_df_long_term.merge(
        df_ltw[['date_field','variable','weight']],
        on=['variable','date_field'],
        how='inner'
    ).rename(columns={'weight': 'weight_date_fields'})

    # -- Unir ambos
    result_df = pd.concat([result_df_short_term, result_df_long_term], ignore_index=True)

    # =========================================================================
    # 7) Scoring para 'change' (change_grade)
    # =========================================================================
    scoring_dict_change = {}
    for var in change_variable_scoring_df['column_name'].unique():
        subset = change_variable_scoring_df[change_variable_scoring_df['column_name'] == var].sort_values('low_interval')
        bins = subset['low_interval'].tolist() + [np.inf]
        labels = subset['grade'].tolist()
        scoring_dict_change[var] = (bins, labels)

    def assign_change_grade(row):
        var = row['variable']
        chg = row['change']
        if var not in scoring_dict_change or pd.isnull(chg):
            return np.nan
        bins, labels = scoring_dict_change[var]
        idx = np.searchsorted(bins, chg, side='right') - 1
        if 0 <= idx < len(labels):
            return labels[idx]
        return np.nan

    result_df['change_grade'] = result_df.apply(assign_change_grade, axis=1)

    # =========================================================================
    # 8) Scoring para 'absolute' (absolute_grade)
    # =========================================================================
    scoring_dict_abs = {}
    for var in absolute_variable_scoring_df['column_name'].unique():
        subset = absolute_variable_scoring_df[absolute_variable_scoring_df['column_name'] == var].sort_values('low_interval')
        bins = subset['low_interval'].tolist() + [np.inf]
        labels = subset['grade'].tolist()
        scoring_dict_abs[var] = (bins, labels)

    def assign_abs_grade(row):
        var = row['variable']
        val = row['currentvalue']
        if var not in scoring_dict_abs or pd.isnull(val):
            return np.nan
        bins, labels = scoring_dict_abs[var]
        idx = np.searchsorted(bins, val, side='right') - 1
        if 0 <= idx < len(labels):
            return labels[idx]
        return np.nan

    result_df['absolute_grade'] = result_df.apply(assign_abs_grade, axis=1)

    # =========================================================================
    # 9) Calcular grade combinando 'change_grade' y 'absolute_grade'
    # =========================================================================
    grade_mapping = {
        'A+': 100, 'A': 95, 'A-': 90,
        'B+': 87.5, 'B': 85, 'B-': 80,
        'C+': 77.5, 'C': 75, 'C-': 70,
        'D+': 68.5, 'D': 65, 'D-': 62,
        'F': 50, 'NG': 0
    }
    grade_mapping_with_ng = {**grade_mapping, 'NG': 0}

    def numeric_to_letter(value):
        if value >= 100:
            return 'A+'
        elif value >= 95:
            return 'A'
        elif value >= 90:
            return 'A-'
        elif value >= 87.5:
            return 'B+'
        elif value >= 85:
            return 'B'
        elif value >= 80:
            return 'B-'
        elif value >= 77.5:
            return 'C+'
        elif value >= 75:
            return 'C'
        elif value >= 70:
            return 'C-'
        elif value >= 68.5:
            return 'D+'
        elif value >= 65:
            return 'D'
        elif value >= 62:
            return 'D-'
        elif value >= 1:
            return 'F'
        else:
            return 'NG'

    result_df['grade_numeric'] = (
        result_df['absolute_grade'].map(grade_mapping_with_ng).fillna(0) * 0.8 +
        result_df['change_grade'].map(grade_mapping_with_ng).fillna(0) * 0.2
    )
    result_df['grade'] = result_df['grade_numeric'].apply(numeric_to_letter)

    # =========================================================================
    # 10) Ordenar columnas y llenar nulos en weight
    # =========================================================================
    columnas_orden = [
        'variable', 'date_field', 
        'currentvalue', 'comparison_value', 'change',
        'weight_variable_rankings', 'weight_date_fields', 'weight_data_partitions',
        'change_grade', 'absolute_grade', 'grade',
        'grade_numeric'
    ]
    columnas_finales = [c for c in columnas_orden if c in result_df.columns] + [
        c for c in result_df.columns if c not in columnas_orden
    ]
    result_df = result_df[columnas_finales]
    result_df['weight_variable_rankings'] = result_df['weight_variable_rankings'].fillna(0)
    if 'weight_date_fields' in result_df.columns:
        result_df['weight_date_fields'] = result_df['weight_date_fields'].fillna(0)
    else:
        result_df['weight_date_fields'] = 0

    # Calcular "weight" total y "math_eval"
    result_df['weight'] = result_df['weight_variable_rankings'] * result_df['weight_date_fields']
    result_df['math_eval'] = result_df['grade_numeric'] * result_df['weight']

    # =========================================================================
    # 11) Forzar weight=0 y grades="NG" para spend_mix
    # =========================================================================
    mask_spend_mix = result_df['variable'].str.startswith('spend_mix_')
    result_df.loc[mask_spend_mix, 'weight'] = 0
    result_df.loc[mask_spend_mix, 'change_grade'] = 'NG'
    result_df.loc[mask_spend_mix, 'absolute_grade'] = 'NG'
    result_df.loc[mask_spend_mix, 'grade'] = 'NG'
    result_df.loc[mask_spend_mix, 'math_eval'] = 0

    # Eliminar columnas auxiliares
    to_drop = ['grade_numeric', 'weight_variable_rankings', 'weight_date_fields']
    for col in to_drop:
        if col in result_df.columns:
            result_df.drop(columns=col, inplace=True)

    result_df = result_df.loc[~ ((result_df['currentvalue'] == 0) & (result_df['comparison_value'] != 0))]
    return result_df




def adjust_marketing_variable_rankings_df(df, variable_rankings_df):

    new_subs = df["new_customers_subscribers"].sum()
    if new_subs != 0:
        return variable_rankings_df
    
    df_keep = variable_rankings_df[variable_rankings_df["column_name"]!='cts'].copy()
    df_removed = variable_rankings_df[variable_rankings_df["column_name"]=='cts'].copy()

    weight_removed = df_removed['weight'].sum()
    current_total_weight = df_keep['weight'].sum()

    if current_total_weight > 0:
        df_keep['weight'] += (df_keep['weight'] / current_total_weight) * weight_removed

    df_keep = df_keep.sort_values(by='rank')
    df_keep['rank'] = range(1, len(df_keep) + 1)
    variable_rankings_df = df_keep.reset_index(drop=True)
    return variable_rankings_df




def add_filters_to_query(base_query: str, filters: dict) -> str:
    """
    Función que agrega filtros adicionales (en formato columna IN (...)) a una query SQL.
    
    :param base_query: Consulta base, que puede contener o no una cláusula WHERE.
    :param filters: Diccionario de filtros. Cada clave es el nombre de columna y el valor es una lista de strings.
    :return: Consulta con los filtros adicionales incluidos.
    """
    if not filters:
        return base_query

    print(base_query, filters)


    additional_conditions = []
    for column, values in filters.items():
        str_values = ", ".join(f"'{value}'" for value in values)
        condition = f"{column} IN ({str_values})"
        additional_conditions.append(condition)
        
    filter_clause = " AND ".join(additional_conditions)
    
    if "WHERE" in base_query.upper():
        final_query = f"{base_query}\n  AND {filter_clause}"
    else:
        # No existe WHERE en la query, lo agregamos
        final_query = f"{base_query}\nWHERE {filter_clause}"
        
    return final_query



def cohort_tree(
    gbq_client, 
    project_id, 
    dataset_id, 
    client_id,  
    date_str,  
    google_sheets_credentials,
    sheet_url,
    agent,
    filters
):
    print('starting cohort tree process')
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
    variable_rankings_df = variable_rankings_df.applymap(adjust_value_spreedsheet)
    variable_rankings_df["weight"] = variable_rankings_df["weight"].astype(float)
    variable_rankings_df["rank"] = variable_rankings_df["rank"].astype(int)

    date_fields_rankings = sheet.worksheet("date_fields_rankings").get_all_values()
    date_fields_rankings_df = pd.DataFrame(date_fields_rankings[1:], columns=date_fields_rankings[0])
    date_fields_rankings_df = date_fields_rankings_df.applymap(adjust_value_spreedsheet)
    date_fields_rankings_df["weight"] = date_fields_rankings_df["weight"].astype(float)
    date_fields_rankings_df["rank"] = date_fields_rankings_df["rank"].astype(int)

    variable_scoring = sheet.worksheet("variable_scoring").get_all_values()
    variable_scoring_df = pd.DataFrame(variable_scoring[1:], columns=variable_scoring[0])
    variable_scoring_df = variable_scoring_df.applymap(adjust_value_spreedsheet)
    variable_scoring_df["low_interval"] = variable_scoring_df["low_interval"].astype(float)
    variable_scoring_df["high_interval"] = variable_scoring_df["high_interval"].astype(float)
    print('spreadsheet data loaded')

    query = f"""
    SELECT * 
    FROM `{project_id}.{dataset_id}.itmd_table_for_cohorts` 
    WHERE client_id = '{client_id}'
    """
    df = gbq_client.query(query).to_dataframe()
    dates_df = generate_comparison_table(date_str) 

    if df['month'].dtype != 'datetime64[ns]':
        df['month'] = pd.to_datetime(df['month'], errors='coerce')

    if df['cohort_date'].dtype != 'datetime64[ns]':
        df['cohort_date'] = pd.to_datetime(df['cohort_date'], errors='coerce')


    if agent == 'agent_1':
        dates_df = generate_comparison_table(date_str)
        dates_df = dates_df.loc[dates_df["date_field"].isin(date_fields_rankings_df["date_field"].unique())] 
        df_treeData = get_cohort_treeData_df(df, dates_df, variable_rankings_df, date_fields_rankings_df, variable_scoring_df)
        treeMath_score = df_treeData["math_eval"].sum()
        treeC7P_df = get_TreeC7P_df(df)

        results = {
        'treeName': 'cohort',
        'treeMath_score': treeMath_score,
        'treeC7P': treeC7P_df.to_dict(orient='records'),
        'treeData': df_treeData.to_dict(orient='records'),
        'dates_df': dates_df}
        
        return results


def get_cohort_treeData_df(
    df, 
    dates_df, 
    variable_rankings_df, 
    date_fields_rankings_df, 
    variable_scoring_df
):

    date_columns = ['start_date', 'end_date', 'start_date_comparison', 'end_date_comparison']
    for col in date_columns:
        if dates_df[col].dtype != 'datetime64[ns]':
            dates_df[col] = pd.to_datetime(dates_df[col], errors='coerce')

    # =========================================================================
    # 1. Definir funciones de agregación para cada variable solicitada
    # =========================================================================
    def cohort_size(df_):
        """
        Sum(clients_of_cohort) cuando customer_order_number = 1
        """
        return df_.loc[df_['customer_order_number'] == 1, 'clients_of_cohort'].sum()

    def cohort_ltv(df_):
        """
        (sum(revenue) - sum(marketing_spend)) / cohort_size
        """
        size = cohort_size(df_)
        if size == 0:
            return 0
        return (df_['revenue'].sum() - df_['marketing_spend'].sum()) / size

    def cohort_initial_revenue(df_):
        """
        sum(revenue) cuando customer_order_number = 1
        """
        return df_.loc[df_['customer_order_number'] == 1, 'revenue'].sum()

    def cohort_supplemental_revenue(df_):
        """
        sum(revenue) cuando customer_order_number > 1
        """
        return df_.loc[df_['customer_order_number'] > 1, 'revenue'].sum()

    def cohort_total_supplemental_orders(df_):
        """
        sum(clients_of_cohort) cuando customer_order_number > 1
        """
        return df_.loc[df_['customer_order_number'] > 1, 'clients_of_cohort'].sum()

    def cohort_total_marketing_spend(df_):
        """
        sum(marketing_spend)
        """
        return df_['marketing_spend'].sum()

    def cohort_ltr(df_):
        """
        sum(revenue) / cohort_size
        """
        size = cohort_size(df_)
        if size == 0:
            return 0
        return df_['revenue'].sum() / size

    def cohort_subscribers(df_):
        """
        sum(clients_of_cohort) cuando customer_order_number = 1 
        & client_type_lifetime = 'Subscriber'
        """
        mask = (
            (df_['customer_order_number'] == 1) &
            (df_['client_type_lifetime'] == 'Subscriber')
        )
        return df_.loc[mask, 'clients_of_cohort'].sum()

    def cohort_one_off(df_):
        """
        sum(clients_of_cohort) cuando customer_order_number = 1 
        & client_type_lifetime = 'One-Off'
        """
        mask = (
            (df_['customer_order_number'] == 1) &
            (df_['client_type_lifetime'] == 'One-Off')
        )
        return df_.loc[mask, 'clients_of_cohort'].sum()

    def cohort_total_AOV(df_):
        """
        sum(revenue) / sum(clients_of_cohort)
        """
        total_clients = df_['clients_of_cohort'].sum()
        if total_clients == 0:
            return 0
        return df_['revenue'].sum() / total_clients

    def cohort_supp_AOV(df_):
        """
        sum(revenue) / sum(clients_of_cohort) cuando customer_order_number > 1
        """
        subset = df_.loc[df_['customer_order_number'] > 1]
        total_clients = subset['clients_of_cohort'].sum()
        if total_clients == 0:
            return 0
        return subset['revenue'].sum() / total_clients

    def cohort_repur_rate(df_):
        """
        sum(clients_of_cohort) cuando customer_order_number = 2 / cohort_size
        """
        size = cohort_size(df_)
        if size == 0:
            return 0
        second_order_clients = df_.loc[df_['customer_order_number'] == 2, 'clients_of_cohort'].sum()
        return second_order_clients / size

    # Diccionario de funciones especiales, mapeando nombre_variable -> función
    special_vars = {
        'cohort_size': cohort_size,
        'cohort_ltv': cohort_ltv,
        'cohort_initial_revenue': cohort_initial_revenue,
        'cohort_supplemental_revenue': cohort_supplemental_revenue,
        'cohort_total_supplemental_orders': cohort_total_supplemental_orders,
        'cohort_total_marketing_spend': cohort_total_marketing_spend,
        'cohort_ltr': cohort_ltr,
        'cohort_subscribers': cohort_subscribers,
        'cohort_one_off': cohort_one_off,
        'cohort_total_AOV': cohort_total_AOV,
        'cohort_supp_AOV': cohort_supp_AOV,
        'cohort_repur_rate': cohort_repur_rate
    }

    # =========================================================================
    # 2. Generar la "cross table" de fechas y variables
    #    (igual que en el ejemplo, cruzamos dates_df con variable_rankings_df)
    # =========================================================================
    cross_df = (
        dates_df.assign(key=1)
        .merge(variable_rankings_df.assign(key=1), on='key', suffixes=('', '_var'))
        .drop(columns=['key'])
    )
    # cross_df contiene ahora todas las combinaciones:
    # [start_date, end_date, start_date_comparison, end_date_comparison] x [column_name]

    # =========================================================================
    # 3. Iterar sobre cada combinación y calcular currentvalue / comparison_value
    # =========================================================================
    rows = []

    for _, row_dates in cross_df.iterrows():
        date_field = row_dates['date_field']        # Por si usas varios 'date_field' en date_fields_rankings_df
        column_name = row_dates['column_name']      # La variable (e.g. 'cohort_size')
        weight_variable_rankings = row_dates['weight']  # Peso de la variable

        # Fechas y rangos
        start_date = row_dates['start_date']
        end_date = row_dates['end_date']
        start_comp = row_dates['start_date_comparison']
        end_comp = row_dates['end_date_comparison']

        # ---------------------------------------------------------------------
        # Filtrar df_current y df_comparison
        # ---------------------------------------------------------------------
        # Asumiendo: 'cohort_date' se filtra con [start_date, end_date]
        # y 'month' se filtra con [start_date_comparison, end_date_comparison].
        df_current = df.loc[
            (df['cohort_date'] >= start_date) & (df['cohort_date'] <= end_date) &
            (df['month'] >= start_date) & (df['month'] <= end_date)
        ]

        df_comparison = df.loc[
            (df['cohort_date'] >= start_comp) & (df['cohort_date'] <= end_comp) &
            # En un caso real, puede que la comparación se base en otras columnas,
            # pero por simplicidad repetimos 'month' con el mismo rango:
            (df['month'] >= start_comp) & (df['month'] <= end_comp)
        ]

        # ---------------------------------------------------------------------
        # Calcular currentvalue y comparison_value
        # ---------------------------------------------------------------------
        if column_name in special_vars:
            func = special_vars[column_name]
            current_value = func(df_current)
            comparison_value = func(df_comparison)
        else:
            # Si hay alguna variable que no está en special_vars, 
            # asumimos un sum() directo
            current_value = df_current[column_name].sum()
            comparison_value = df_comparison[column_name].sum()

        # ---------------------------------------------------------------------
        # Calcular change
        # ---------------------------------------------------------------------
        if comparison_value == 0:
            change = 1  # Siguiendo la lógica original: si comp = 0, cambio = 1
        else:
            change = (current_value - comparison_value) / comparison_value

        # Crear df con la fila de resultado
        data_dict = {
            'variable': column_name,
            'date_field': date_field,
            'currentvalue': current_value,
            'comparison_value': comparison_value,
            'change': change,
            'weight_variable_rankings': weight_variable_rankings
        }
        rows.append(data_dict)

    # Concatenar todas las filas en un DataFrame
    result_df = pd.DataFrame(rows)

    # =========================================================================
    # 4. Merge con los pesos de date_fields_rankings_df (si aplican)
    # =========================================================================
    # Se asume que date_fields_rankings_df = [date_field, weight]
    result_df = result_df.merge(
        date_fields_rankings_df[['date_field', 'weight']], 
        on='date_field',
        how='inner'
    ).rename(columns={'weight': 'weight_date_fields'})

    # =========================================================================
    # 5. Asignar 'grade' según variable_scoring_df
    #    (igual que el ejemplo original)
    # =========================================================================
    # Construimos bins y labels para cada variable
    scoring_dict = {}
    for var in variable_scoring_df['column_name'].unique():
        subset = variable_scoring_df[variable_scoring_df['column_name'] == var].sort_values('low_interval')
        bins = subset['low_interval'].tolist() + [np.inf]
        labels = subset['grade'].tolist()
        scoring_dict[var] = (bins, labels)

    def assign_grade(row):
        var = row['variable']
        chg = row['change']
        if var not in scoring_dict or pd.isnull(chg):
            return np.nan
        bins, labels = scoring_dict[var]
        idx = np.searchsorted(bins, chg, side='right') - 1
        if 0 <= idx < len(labels):
            return labels[idx]
        return np.nan

    result_df['grade'] = result_df.apply(assign_grade, axis=1)
    result_df['weight_variable_rankings'] = result_df['weight_variable_rankings'].fillna(0)
    result_df['weight_date_fields'] = result_df['weight_date_fields'].fillna(0)
    result_df['weight'] = result_df['weight_variable_rankings'] * result_df['weight_date_fields']

    # Mapeo de letras a valores numéricos (igual que ejemplo)
    grade_mapping = {
        'A+': 100, 'A': 95, 'A-': 90,
        'B+': 87.5, 'B': 85, 'B-': 80,
        'C+': 77.5, 'C': 75, 'C-': 70,
        'D+': 68.5, 'D': 65, 'D-': 62,
        'F': 50
    }
    result_df['grade_numeric'] = result_df['grade'].map(grade_mapping)

    result_df['math_eval'] = result_df['grade_numeric'] * result_df['weight']

    # Si quieres limpiar columnas intermedias:
    # (en el ejemplo original se hacía un drop)
    result_df.drop(
        columns=[
            'grade_numeric', 
            'weight_variable_rankings', 
            'weight_date_fields'
        ],
        inplace=True
    )

    # Orden final de columnas (ajústalo a tu preferencia):
    columnas_orden = [
        'variable', 'date_field', 'currentvalue', 'comparison_value', 
        'change', 'weight', 'math_eval', 'grade'
    ]
    otras_columnas = [c for c in result_df.columns if c not in columnas_orden]
    result_df = result_df[columnas_orden + otras_columnas]
    result_df = result_df.loc[~ ((result_df['currentvalue'] == 0) & (result_df['comparison_value'] != 0))]

    return result_df


def get_TreeC7P_df(df):
    df_sorted = df.sort_values(by='cohort_date')
    ultimas_7_fechas = df_sorted['cohort_date'].drop_duplicates().tail(7)
    df_7 = df_sorted[df_sorted['cohort_date'].isin(ultimas_7_fechas)]

    def calcular_metricas_por_cohort(grupo):
        # cohort_size: clientes con customer_order_number == 1
        mask_primera_orden = (grupo['customer_order_number'] == 1)
        cohort_size = grupo.loc[mask_primera_orden, 'clients_of_cohort'].sum()

        mask_subscriber = mask_primera_orden & (grupo['client_type_lifetime'] == 'Subscriber')
        cohort_subscribers = grupo.loc[mask_subscriber, 'clients_of_cohort'].sum()

        mask_one_off = mask_primera_orden & (grupo['client_type_lifetime'] == 'One-Off')
        cohort_one_off = grupo.loc[mask_one_off, 'clients_of_cohort'].sum()

        cohort_initial_revenue = grupo.loc[mask_primera_orden, 'revenue'].sum()
        total_revenue = grupo['revenue'].sum()

        if cohort_size != 0:
            cohort_ltr = total_revenue / cohort_size
        else:
            cohort_ltr = 0

        mask_supp = (grupo['customer_order_number'] > 1)
        cohort_total_supp_orders = grupo.loc[mask_supp, 'clients_of_cohort'].sum()

        total_marketing_spend = grupo['marketing_spend'].sum()
        if cohort_size != 0:
            cohort_ltv = (total_revenue - total_marketing_spend) / cohort_size
        else:
            cohort_ltv = 0

        return pd.Series({
            'Cohort size': cohort_size,
            'Cohort subscribers': cohort_subscribers,
            'Cohort One-off': cohort_one_off,
            'Cohort Initial Revenue': cohort_initial_revenue,
            'Cohort LTR': cohort_ltr,
            'Cohort Total Supplemental Orders': cohort_total_supp_orders,
            'Cohort LTV': cohort_ltv
        })


    df_grouped = df_7.groupby('cohort_date').apply(calcular_metricas_por_cohort)
    df_grouped = df_grouped.sort_index()
    tabla_final = df_grouped.T
    tabla_final.columns = [ts.strftime('%Y-%m-%d') if isinstance(ts, pd.Timestamp) else ts for ts in tabla_final.columns]
    return tabla_final



def generate_comparison_table(input_date_str: str) -> pd.DataFrame:
    """
    Genera un DataFrame con distintos rangos de fechas (MTD, QTD, YTD, etc.)
    comparándolos con períodos previos (año anterior, mes anterior, etc.).
    """
    current_date = pd.to_datetime(input_date_str)

    def fmt(date_obj):
        return date_obj.strftime('%Y-%m-%d')

    prev_date = current_date - timedelta(days=1)
    prev_year_date = current_date - relativedelta(years=1)
    prev_2year_date = current_date - relativedelta(years=2)

    start_of_current_month = current_date.replace(day=1)
    start_of_past_month = start_of_current_month - relativedelta(months=1)

    start_of_past_two_months = start_of_past_month - relativedelta(months=1)
    start_of_past_six_months = start_of_current_month - relativedelta(months=5)
    start_of_past_fourteen_months = start_of_current_month - relativedelta(months=13)

    start_of_current_month_py = start_of_current_month - relativedelta(years=1)
    start_of_past_month_py = start_of_past_month - relativedelta(years=1)

    cohort_rows = [

        {
            "date_field": "recent_trend",
            "start_date": fmt(start_of_past_month),
            "end_date": fmt(start_of_current_month),
            "start_date_comparison": fmt(start_of_past_month_py),
            "end_date_comparison": fmt(start_of_current_month_py)
        },

         {
            "date_field": "6M-to-2M PY",
            "start_date": fmt(start_of_past_six_months),
            "end_date": fmt(start_of_past_two_months),
            "start_date_comparison": fmt(start_of_past_six_months - relativedelta(years=1)),
            "end_date_comparison": fmt(start_of_past_two_months - relativedelta(years=1))
        },

        {
            "date_field": "6M-to-2M P2Y",
            "start_date": fmt(start_of_past_six_months),
            "end_date": fmt(start_of_past_two_months),
            "start_date_comparison": fmt(start_of_past_six_months - relativedelta(years=2)),
            "end_date_comparison": fmt(start_of_past_two_months - relativedelta(years=2))
        },

        {
            "date_field": "TTM-min2 PY",
            "start_date": fmt(start_of_past_fourteen_months),
            "end_date": fmt(start_of_past_six_months),
            "start_date_comparison": fmt(start_of_past_fourteen_months - relativedelta(years=1)),
            "end_date_comparison": fmt(start_of_past_six_months - relativedelta(years=1))
        }

    ]






    end_of_prev_month = start_of_current_month - timedelta(days=1)
    start_of_prev_month = end_of_prev_month.replace(day=1)

    def shift_day_with_clamp(base_date, target_day):
        """ Ajusta el día de 'base_date' a 'target_day', controlando fin de mes. """
        try:
            new_date = base_date.replace(day=target_day)
        except ValueError:
            new_date = (
                base_date.replace(day=1) 
                + relativedelta(months=1) 
                - relativedelta(days=1)
            )
        return new_date

    end_of_prev_month_equivalent = shift_day_with_clamp(start_of_prev_month, current_date.day)
    start_of_last_full_month = start_of_prev_month
    end_of_last_full_month = end_of_prev_month

    start_of_month_before_that = start_of_last_full_month - relativedelta(months=1)
    end_of_month_before_that = start_of_month_before_that.replace(day=1) + relativedelta(months=1, days=-1)

    # Quarter
    current_quarter = (current_date.month - 1) // 3 + 1
    start_of_current_quarter_month = 3 * (current_quarter - 1) + 1
    start_of_current_quarter = current_date.replace(
        month=start_of_current_quarter_month, 
        day=1
    )

    start_of_previous_quarter = start_of_current_quarter - relativedelta(months=3)
    end_of_previous_quarter_equivalent = shift_day_with_clamp(start_of_previous_quarter, current_date.day)

    # Year
    start_of_current_year = current_date.replace(month=1, day=1)
    start_of_prev_year = start_of_current_year - relativedelta(years=1)
    start_of_prev_2year = start_of_current_year - relativedelta(years=2)

    # Weeks (WoW)
    def get_last_completed_week_range(reference_date: pd.Timestamp):
        last_completed_sunday = reference_date - timedelta(days=(reference_date.weekday() + 1))
        last_completed_monday = last_completed_sunday - timedelta(days=6)
        return (last_completed_monday, last_completed_sunday)

    wow_current_start, wow_current_end = get_last_completed_week_range(current_date)
    wow_prev_start = wow_current_start - timedelta(days=7)
    wow_prev_end = wow_current_end - timedelta(days=7)

    def shift_week_by_years(week_start, week_end, years):
        return (week_start - relativedelta(years=years), week_end - relativedelta(years=years))

    wow_current_start_py, wow_current_end_py = shift_week_by_years(wow_current_start, wow_current_end, 1)
    wow_current_start_p2y, wow_current_end_p2y = shift_week_by_years(wow_current_start, wow_current_end, 2)

    # L3D
    l3d_end_date = current_date - timedelta(days=1)
    l3d_start_date = current_date - timedelta(days=4)
    l3d_end_date_comp = l3d_start_date - timedelta(days=1)
    l3d_start_date_comp = l3d_end_date_comp - timedelta(days=2)

    rows = [
        {
            "date_field": "DoD PD",
            "start_date": fmt(current_date),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(prev_date),
            "end_date_comparison": fmt(prev_date)
        },
        {
            "date_field": "DoD PY",
            "start_date": fmt(current_date),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(prev_year_date),
            "end_date_comparison": fmt(prev_year_date)
        },
        {
            "date_field": "DoD P2Y",
            "start_date": fmt(current_date),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(prev_2year_date),
            "end_date_comparison": fmt(prev_2year_date)
        },
        {
            "date_field": "MTD PM",
            "start_date": fmt(start_of_current_month),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(start_of_prev_month),
            "end_date_comparison": fmt(end_of_prev_month_equivalent)
        },
        {
            "date_field": "MTD PY",
            "start_date": fmt(start_of_current_month),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(start_of_current_month - relativedelta(years=1)),
            "end_date_comparison": fmt(current_date - relativedelta(years=1))
        },
        {
            "date_field": "MTD P2Y",
            "start_date": fmt(start_of_current_month),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(start_of_current_month - relativedelta(years=2)),
            "end_date_comparison": fmt(current_date - relativedelta(years=2))
        },
        {
            "date_field": "MoM PM",
            "start_date": fmt(start_of_last_full_month),
            "end_date": fmt(end_of_last_full_month),
            "start_date_comparison": fmt(start_of_month_before_that),
            "end_date_comparison": fmt(end_of_month_before_that)
        },
        {
            "date_field": "MoM PY",
            "start_date": fmt(start_of_last_full_month),
            "end_date": fmt(end_of_last_full_month),
            "start_date_comparison": fmt(start_of_last_full_month - relativedelta(years=1)),
            "end_date_comparison": fmt(end_of_last_full_month - relativedelta(years=1))
        },
        {
            "date_field": "MoM P2Y",
            "start_date": fmt(start_of_last_full_month),
            "end_date": fmt(end_of_last_full_month),
            "start_date_comparison": fmt(start_of_last_full_month - relativedelta(years=2)),
            "end_date_comparison": fmt(end_of_last_full_month - relativedelta(years=2))
        },
        {
            "date_field": "QTD PQ",
            "start_date": fmt(start_of_current_quarter),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(start_of_previous_quarter),
            "end_date_comparison": fmt(end_of_previous_quarter_equivalent)
        },
        {
            "date_field": "QTD PY",
            "start_date": fmt(start_of_current_quarter),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(start_of_current_quarter - relativedelta(years=1)),
            "end_date_comparison": fmt(current_date - relativedelta(years=1))
        },
        {
            "date_field": "QTD P2Y",
            "start_date": fmt(start_of_current_quarter),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(start_of_current_quarter - relativedelta(years=2)),
            "end_date_comparison": fmt(current_date - relativedelta(years=2))
        },
        {
            "date_field": "YTD PY",
            "start_date": fmt(start_of_current_year),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(start_of_prev_year),
            "end_date_comparison": fmt(current_date - relativedelta(years=1))
        },
        {
            "date_field": "YTD P2Y",
            "start_date": fmt(start_of_current_year),
            "end_date": fmt(current_date),
            "start_date_comparison": fmt(start_of_prev_2year),
            "end_date_comparison": fmt(current_date - relativedelta(years=2))
        },
        {
            "date_field": "WoW PW",
            "start_date": fmt(wow_current_start),
            "end_date": fmt(wow_current_end),
            "start_date_comparison": fmt(wow_prev_start),
            "end_date_comparison": fmt(wow_prev_end)
        },
        {
            "date_field": "WoW PY",
            "start_date": fmt(wow_current_start),
            "end_date": fmt(wow_current_end),
            "start_date_comparison": fmt(wow_current_start_py),
            "end_date_comparison": fmt(wow_current_end_py)
        },
        {
            "date_field": "WoW P2Y",
            "start_date": fmt(wow_current_start),
            "end_date": fmt(wow_current_end),
            "start_date_comparison": fmt(wow_current_start_p2y),
            "end_date_comparison": fmt(wow_current_end_p2y)
        },
        {
            "date_field": "L3D P3D",
            "start_date": fmt(l3d_start_date),
            "end_date": fmt(l3d_end_date),
            "start_date_comparison": fmt(l3d_start_date_comp),
            "end_date_comparison": fmt(l3d_end_date_comp)
        },
        {
            "date_field": "L3D PY",
            "start_date": fmt(l3d_start_date),
            "end_date": fmt(l3d_end_date),
            "start_date_comparison": fmt(l3d_start_date - relativedelta(years=1)),
            "end_date_comparison": fmt(l3d_end_date - relativedelta(years=1))
        },
    ]

    rows += cohort_rows

    df_result = pd.DataFrame(rows, columns=[
        "date_field", 
        "start_date", 
        "end_date", 
        "start_date_comparison", 
        "end_date_comparison"
    ])
    return df_result

def get_tree_data_by_id(tree_id, *args):
    trees = {
        'revenue': revenue_tree,
        'marketing': marketing_tree
    }
    if tree_id in trees:
        return trees[tree_id](*args)
    else:
        raise ValueError("Invalid tree_id")



def adjust_value_spreedsheet(value):
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



def main():
    print("Starting the process")
    project_id = os.environ.get('PROJECT_ID')
    dataset_id = os.environ.get('DATASET_ID')


    gbq_service_account_info = json.loads(os.environ.get('GBQ_SERVICE_ACCOUNT') , strict=False)
    gbq_client = bigquery.Client.from_service_account_info(gbq_service_account_info)

    google_sheets_credentials = json.loads(os.environ.get('GOOGLE_SHEETS_SERVICE_ACCOUNT'))
    main_sheet_url = os.environ.get('SPREADSHEET_URL')

    scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/spreadsheets'
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_dict(google_sheets_credentials, scope)
    client_spreed_sheet = gspread.authorize(creds)
    sheet = client_spreed_sheet.open_by_url(main_sheet_url)

    input_worksheet = sheet.worksheet("input_df").get_all_values()
    input_df = pd.DataFrame(input_worksheet[1:], columns=input_worksheet[0])
    input_df = input_df.applymap(adjust_value_spreedsheet)

    client_id = input_df["client_id"].iloc[0]
    input_date = input_df["input_date"].iloc[0]
    tree_name = input_df["tree_name"].iloc[0]

    agents_url = sheet.worksheet("production_agents").get_all_values()
    agents_url_df = pd.DataFrame(agents_url[1:], columns=agents_url[0])
    agents_url_df = agents_url_df.applymap(adjust_value_spreedsheet)

    sheet_url = agents_url_df[agents_url_df["super_agent"] == tree_name]["url"].iloc[0]

    tree_result = get_tree_data_by_id(
        tree_name,
        gbq_client,  
        project_id,
        dataset_id,
        client_id,
        input_date,
        google_sheets_credentials,
        sheet_url,
        "agent_1"
    )

    df_treeData  = pd.DataFrame(tree_result["treeData"])
    dates_df = tree_result["dates_df"]
    
    df_treeData["client_id"] = client_id

    creds = ServiceAccountCredentials.from_json_keyfile_dict(google_sheets_credentials, scope)
    client_spreed_sheet = gspread.authorize(creds)
    sheet = client_spreed_sheet.open_by_url(main_sheet_url)

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

    l3d_end_date = current_date - timedelta(days=1)
    l3d_start_date = current_date - timedelta(days=4)
    l3d_end_date_comp = l3d_start_date - timedelta(days=1)
    l3d_start_date_comp = l3d_end_date_comp - timedelta(days=2)
    
    rows.append({
        "date_field": "L3D P3D",
        "start_date": fmt(l3d_start_date),
        "end_date": fmt(l3d_end_date),
        "start_date_comparison": fmt(l3d_start_date_comp),
        "end_date_comparison": fmt(l3d_end_date_comp)
    })


    rows.append({
        "date_field": "L3D PY",
        "start_date": fmt(l3d_start_date),
        "end_date": fmt(l3d_end_date),
        "start_date_comparison": fmt(l3d_start_date - relativedelta(years=1)),
        "end_date_comparison": fmt(l3d_end_date - relativedelta(years=1))
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