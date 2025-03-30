import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime, timedelta


# Helper function for coalesce
def coalesce(*args):
    res = args[0]
    for df in args[1:]:
        res = res.fillna(df)
        
# Helper function for sum that allows missing values
def sum_sas(*args):
    valid_args = [arg for arg in args if pd.notna(arg)]
    return sum(valid_args) if valid_args else np.nan


def quarterize(df: pd.DataFrame,
              id_vars: List[str] = ['gvkey', 'fyr'],
              fyear: str = 'fyearq',
              fqtr: str = 'fqtr') -> pd.DataFrame:
    """
    Convert year-to-date (YTD) flow variables to quarterly values and standardize quarterly data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing quarterly financial data
    id_vars : List[str], default ['gvkey', 'fyr']
        List of identifier variables for grouping
    fyear : str, default 'fyearq'
        Name of fiscal year column
    fqtr : str, default 'fqtr'
        Name of fiscal quarter column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with quarterized values
    """
    # Sort the data by company and time
    df = df.sort_values(id_vars + [fyear, fqtr])
    
    # Get list of YTD variables (ending with 'y')
    ytd_vars = [col for col in df.columns if col.endswith('y')]
    
    # Function to quarterize YTD values
    def quarterize_ytd(group: pd.DataFrame, var: str) -> pd.Series:
        ytd = group[var].copy()
        qtr = pd.Series(index=ytd.index, dtype=float)
        
        # Group by fiscal year
        for year in group[fyear].unique():
            year_mask = group[fyear] == year
            year_data = ytd[year_mask]
            year_qtrs = group.loc[year_mask, fqtr]
            for q in range(4):
                q_mask = year_qtrs == q
                if not any(q_mask):
                    continue
                    
                if q == 0:
                    # Q1: quarterly value = YTD value
                    qtr.loc[q_mask] = year_data.loc[q_mask]
                else:
                    # Q2-Q4: quarterly value = current YTD - previous quarter YTD
                    prev_q_mask = year_qtrs == (q - 1)
                    if any(prev_q_mask):
                        qtr.loc[q_mask] = year_data.loc[q_mask].values - year_data.loc[prev_q_mask].values                    
        return qtr
    
    # Process each YTD variable
    for ytd_var in ytd_vars:
        base_var = ytd_var[:-1]  # Remove 'y' suffix
        qtr_var = f"{base_var}q"  # Variable name with 'q' suffix
        temp_var = f"{ytd_var}_q"  # Temporary quarterized variable name
        
        # Calculate quarterized values
        df[temp_var] = df.groupby(id_vars, group_keys=False).apply(
            lambda x: quarterize_ytd(x, ytd_var)
        )
        # Replace missing quarterly values with quarterized values
        if qtr_var in df.columns:
            df[qtr_var] = df[qtr_var].fillna(df[temp_var])
        else:
            df[qtr_var] = df[temp_var] 
        # Drop temporary variable
        df = df.drop(columns=[temp_var])
            
    return df


def compustat_fx(exrt_dly: pd.DataFrame) -> pd.DataFrame:
    """
    Create a foreign exchange rate dataset for converting currencies to USD.
    
    Parameters
    ----------
    exrt_dly : pd.DataFrame
        Compustat daily exchange rate data with columns:
        - fromcurd: source currency
        - tocurd: target currency
        - datadate: date of exchange rate
        - exratd: exchange rate
        
    Returns
    -------
    pd.DataFrame
        Daily exchange rates with columns:
        - curcdd: currency code
        - date: date of rate
        - fx: exchange rate to convert to USD
    """
    # Create USD base case (fx = 1 for all dates)
    start_date = pd.Timestamp('1950-01-01')
    usd_data = pd.DataFrame({
        'curcdd': ['USD'],
        'date': [start_date],
        'fx': [1.0]
    })
    
    # Process Compustat exchange rates
    # Filter for GBP as base currency
    gbp_base = exrt_dly[exrt_dly['fromcurd'] == 'GBP'].copy()
    
    # Get rates from GBP to other currencies
    rates_from_gbp = gbp_base[gbp_base['tocurd'] != 'USD'].copy()
    rates_from_gbp = rates_from_gbp.rename(columns={
        'tocurd': 'curcdd',
        'datadate': 'date'
    })
    
    # Get GBP to USD rate
    gbp_usd = gbp_base[gbp_base['tocurd'] == 'USD'].copy()
    gbp_usd = gbp_usd.rename(columns={'datadate': 'date'})
    
    # Calculate cross rates to USD
    fx_data = pd.merge(
        rates_from_gbp[['curcdd', 'date', 'exratd']],
        gbp_usd[['date', 'exratd']],
        on='date',
        suffixes=('_ccy', '_usd')
    )
    
    # Calculate the exchange rate to USD
    fx_data['fx'] = fx_data['exratd_usd'] / fx_data['exratd_ccy']
    fx_data = fx_data[['curcdd', 'date', 'fx']]
    
    # Combine with USD base case
    fx_all = pd.concat([fx_data, usd_data], ignore_index=True)
    
    # Sort by currency and date
    fx_all = fx_all.sort_values(['curcdd', 'date'], ascending=[True, False])
    
    # Fill in missing dates for each currency
    def fill_dates(group):
        min_date = group['date'].min()
        max_date = group['date'].max()
        full_dates = pd.date_range(min_date, max_date, freq='D')
        return group.set_index('date').reindex(full_dates).reset_index().rename(columns={'index': 'date'})
    
    fx_filled = fx_all.groupby('curcdd', group_keys=False).apply(fill_dates)
    
    # Forward fill missing values within each currency
    fx_filled['fx'] = fx_filled.groupby('curcdd')['fx'].fillna(method='ffill')
    
    # Remove duplicates and sort
    fx_final = fx_filled.drop_duplicates(subset=['curcdd', 'date']).sort_values(['curcdd', 'date'])    
    return fx_final


class AccountingCharacteristics:
    def __init__(self):
        # Define accounting characteristics groups
        self.acc_chars = {
            # Accounting Based Size Measures
            'size_measures': [
                'assets', 'sales', 'book_equity', 'net_income', 'enterprise_value'
            ],
            
            # 1yr Growth
            'growth_1yr': [
                'at_gr1', 'ca_gr1', 'nca_gr1', 'lt_gr1', 'cl_gr1', 'ncl_gr1',
                'be_gr1', 'pstk_gr1', 'debt_gr1', 'sale_gr1', 'cogs_gr1',
                'sga_gr1', 'opex_gr1'
            ],
            
            # 3yr Growth
            'growth_3yr': [
                'at_gr3', 'ca_gr3', 'nca_gr3', 'lt_gr3', 'cl_gr3', 'ncl_gr3',
                'be_gr3', 'pstk_gr3', 'debt_gr3', 'sale_gr3', 'cogs_gr3',
                'sga_gr3', 'opex_gr3'
            ],
            
            # Helper variables for calculations
            'helper_vars': [
                'data_available', 'count', 'curcd', 'gvkey', 'datadate'
            ]
        }

        self.growth_vars = [
            'at_x', 'ca_x', 'nca_x',          # Assets - Aggregated
            'lt', 'cl_x', 'ncl_x',            # Liabilities - Aggregated
            'be_x', 'pstk_x', 'debt_x',       # Financing Book Values
            'sale_x', 'cogs', 'xsga', 'opex_x', # Sales and Operating Costs
            'capx', 'invt'
        ]
        
        self.ch_asset_vars = [
            'che', 'invt', 'rect', 'ppegt', 'ivao', 'ivst', 'intan',  # Assets
            'dlc', 'ap', 'txp', 'dltt', 'txditc',                     # Liabilities
            'coa_x', 'col_x', 'cowc_x', 'ncoa_x', 'ncol_x', 'nncoa_x', # Operating Assets/Liabilities
            'oa_x', 'ol_x',                                           # Operating Assets/Liabilities
            'fna_x', 'fnl_x', 'nfna_x',                              # Financial Assets/Liabilities
            'gp_x', 'ebitda_x', 'ebit_x', 'ope_x', 'ni_x', 'nix_x', 'dp', # Income Statement
            'fincf_x', 'ocf_x', 'fcf_x', 'nwc_x',                    # Aggregated Cash Flow
            'eqnetis_x', 'dltnetis_x', 'dstnetis_x', 'dbnetis_x', 'netis_x', # Financing Cash Flow
            'eqnpo_x', 'txt', 'eqbb_x', 'eqis_x', 'div_x', 'eqpo_x', 'capx', 'be_x'
        ]

    def main(self):
        annual_data, quarterly_data = self.standardized_accounting_data(
            
        )
        annual_data = self.create_acc_chars(
            annual_data, lag_to_public=4, max_data_lag=18, 
        )
        quarterly_data = self.create_acc_chars(
            quarterly_data, lag_to_public=4, max_data_lag=18, suffix='_q',
        )
        dataset = self.combine_ann_qtr_chars(annual_data, quarterly_data, q_suffix='_q')

    def add_helper_vars(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add helper variables to standardized Compustat accounting data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe with accounting data containing Compustat variables
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added helper variables
        """
        # Create a copy to avoid modifying the input
        df = data.copy()
        
        # First ensure that the gap between two dates is always one month
        date_range = df.groupby(['gvkey', 'curcd']).agg(
            {'datadate': ['min', 'max']}
        ).reset_index()
        date_range.columns = ['gvkey', 'curcd', 'start_date', 'end_date']
        
        # Create monthly dates for each company
        all_dates = []
        for _, row in date_range.iterrows():
            dates = pd.date_range(
                start=row['start_date'], end=row['end_date'], freq='M',
            )
            temp_df = pd.DataFrame({
                'gvkey': row['gvkey'],
                'curcd': row['curcd'],
                'datadate': dates
            })
            all_dates.append(temp_df)
        
        full_dates = pd.concat(all_dates, ignore_index=True)
        
        # Merge with original data
        df = full_dates.merge(
            data,  how='left', on=['gvkey', 'curcd', 'datadate'],
        )
        
        # Add data_available indicator
        df['data_available'] = df['gvkey'].notna()
        
        # Add counter for each company-currency group
        df['count'] = df.groupby(['gvkey', 'curcd']).cumcount() + 1
        
        # Require certain variables to be positive
        positive_vars = {'at', 'sale', 'revt', 'dv', 'che'}
        for var in positive_vars.intersection(df.columns):
            df.loc[df[var] < 0, var] = np.nan
                
        # Income Statement helpers
        df['sale_x'] = df['sale'].fillna(df['revt'])
        df['gp_x'] = df['gp'].fillna(df['sale_x'] - df['cogs'])
        df['opex_x'] = df['xopr'].fillna(df['cogs'] + df['xsga'])
        df['ebitda_x'] = coalesce(
            df['ebitda'], 
            df['oibdp'],
            df['sale_x'] - df['opex_x'],
            df['gp_x'] - df['xsga'],
        )
        df['ebit_x'] = coalesce(
            df['ebit'],
            df['oiadp'],
            df['ebitda_x'] - df['dp'],
        )
        
        # More income statement calculations
        df['op_x'] = df['ebitda_x'] + df['xrd'].fillna(0)
        df['ope_x'] = df['ebitda_x'] - df['xint']
        df['pi_x'] = df['pi'].fillna(
            df['ebit_x'] - df['xint'] + df['spi'].fillna(0) + df['nopi'].fillna(0)
        )
        df['xido_x'] = df['xido'].fillna(
            df['xi'] + df['do'].fillna(0)
        )
        df['ni_x'] = coalesce(
            df['ib'],
            df['ni'] - df['xido_x'],
            df['pi_x'] - df['txt'] - df['mii'].fillna(0),
        )
        df['nix_x'] = coalesce(
            df['ni'],
            df['ni_x'] + df['xido_x'].fillna(0),
            df['ni_x'] + df['xi'] + df['do'],
        )
        df['fi_x'] = df['nix_x'] + df['xint']
        df['div_x'] = df['dvt'].fillna(df['dv'])
        
        # Cash Flow Statement helpers
        df['eqbb_x'] = df[['prstkc', 'purtshr']].sum(axis=1)
        df['eqis_x'] = df['sstk']
        df['eqnetis_x'] = df.apply(lambda x: sum_sas(x['eqis_x'], -x['eqbb_x']), axis=1)
        df['eqpo_x'] = df['div_x'] + df['eqbb_x']
        df['eqnpo_x'] = df['div_x'] - df['eqnetis_x']
        # Calculate the 12-month change in long-term debt (dltt)
        df['dif12_dltt'] = df.groupby(['gvkey', 'curcd'])['dltt'].diff(periods=12)
        # Calculate the 12-month change in short-term debt (dlc)
        df['dif12_dlc'] = df.groupby(['gvkey', 'curcd'])['dlc'].diff(periods=12)
        # Net Long Term Debt issuance
        # coalesce(sum(dltis,-dltr), ltdch, dif12(dltt))
        df['-dltr'] = -df['dltr']
        df['dltnetis_x'] = coalesce(
            df[['dltis', '-dltr']].sum(axis=1),
            df['ltdch'],
            df['dif12_dltt'],
        )
        df = df.drop('-dltr', axis=1)
        # Set to missing if all components are missing and count <= 12
        mask = (df['dltis'].isna() &  df['dltr'].isna() &  df['ltdch'].isna() & 
                (df['count'] <= 12))
        df.loc[mask, 'dltnetis_x'] = np.nan
        # Short-term debt issuance
        # coalesce(dlcch, dif12(dlc))
        df['dstnetis_x'] = df['dlcch'].fillna(df['dif12_dlc'])
        # Set to missing if dlcch is missing and count <= 12
        mask = df['dlcch'].isna() & (df['count'] <= 12)
        df.loc[mask, 'dstnetis_x'] = np.nan
        # Total debt issuance
        df['dbnetis_x'] = df[['dstnetis_x', 'dltnetis_x']].sum(axis=1)
        # Net total issuance (equity + debt)
        df['netis_x'] = df['eqnetis_x'] + df['dbnetis_x']
        # Financing cash flow
        df['fincf_x'] = df['fincf'].fillna(
            df['netis_x'] - df['dv'] + df['fiao'].fillna(0) + df['txbcof'].fillna(0)
        )

        # Balance Sheet helpers
        df['debt_x'] = df[['dltt', 'dlc']].sum(axis=1)
        df['pstk_x'] = coalesce(df['pstkrv'], df['pstkl'], df['pstk'])
        df['seq_x'] = coalesce(
            df['seq'],
            df['ceq'] + df['pstk_x'].fillna(0),
            df['at'] - df['lt'],
        )
        
        # More balance sheet calculations
        df['at_x'] = df['at'].fillna(
            df['seq_x'] + df['dltt'] + df['lct'].fillna(0) + df['lo'].fillna(0) + df['txditc'].fillna(0)
        )
        
        df['ca_x'] = df['act'].fillna(
            df['rect'] + df['invt'] + df['che'] + df['aco']
        )
        
        # Additional calculations
        df['nca_x'] = df['at_x'] - df['ca_x']
        df['cl_x'] = df['lct'].fillna(
            df['ap'] + df['dlc'] + df['txp'] + df['lco']
        )
        df['ncl_x'] = df['lt'] - df['cl_x']
        # Net debt and book equity components
        df['netdebt_x'] = df['debt_x'] - df['che'].fillna(0)
        df['txditc_x'] = df['txditc'].fillna(df[['txdb', 'itcb']].sum(axis=1))
        df['be_x'] = df['seq_x'] + df['txditc_x'].fillna(0) - df['pstk_x'].fillna(0)
        
        # Book enterprise value
        df['bev_x'] = coalesce(
            df['icapt'] + df['dlc'].fillna(0) - df['che'].fillna(0),
            df['netdebt_x'] + df['seq_x'] + df['mib'].fillna(0),
        )
                
        # Operating assets and liabilities decomposition
        df['coa_x'] = df['ca_x'] - df['che']  # Operating (non-cash) current assets
        df['col_x'] = df['cl_x'] - df['dlc'].fillna(0)  # Operating current liabilities
        df['cowc_x'] = df['coa_x'] - df['col_x']  # Current operating working capital
        
        # Non-current components
        df['ncoa_x'] = df['at_x'] - df['ca_x'] - df['ivao'].fillna(0)  # Non-current operating assets
        df['ncol_x'] = df['lt'] - df['cl_x'] - df['dltt']  # Non-current operating liabilities
        df['nncoa_x'] = df['ncoa_x'] - df['ncol_x']  # Net non-current operating assets
        
        # Financial assets and liabilities
        df['fna_x'] = df['ivst'].fillna(0) + df['ivao'].fillna(0)  # Financial assets
        df['fnl_x'] = df['debt_x'] + df['pstk_x'].fillna(0)  # Financial liabilities
        df['nfna_x'] = df['fna_x'] - df['fnl_x']  # Net financial assets
        
        # Operating assets and liabilities
        df['oa_x'] = df['coa_x'] + df['ncoa_x']  # Operating assets
        df['ol_x'] = df['col_x'] + df['ncol_x']  # Operating liabilities
        df['noa_x'] = df['oa_x'] - df['ol_x']  # Net operating assets
        
        # Long-term net operating assets (HXZ A.3.5)
        df['lnoa_x'] = df['ppent'] + df['intan'] + df['ao'] - df['lo'] + df['dp']
        
        # Liquidity measures
        df['caliq_x'] = (df['ca_x'] - df['invt']).fillna(df['che'] + df['rect'])
        df['nwc_x'] = df['ca_x'] - df['cl_x']
        df['ppeinv_x'] = df['ppegt'] + df['invt']
        
        # Asset liquidity (Ortiz-Molina and Phillips, 2014)
        df['aliq_x'] = df['che'] +  0.75 * df['coa_x'] + \
                        0.5 * (df['at_x'] - df['ca_x'] - df['intan'].fillna(0))
        
        # Set negative book equity and enterprise value to missing
        df.loc[df['be_x'] <= 0, 'be_x'] = np.nan
        df.loc[df['bev_x'] <= 0, 'bev_x'] = np.nan
        
        # Calculate accruals components
        # First calculate 12-month changes
        for var in ['cowc_x', 'nncoa_x', 'nfna_x']:
            df[f'dif12_{var}'] = df.groupby(['gvkey', 'curcd'])[var].diff(periods=12)
        
        # Operating accruals
        df['oacc_x'] = (df['ni_x'] - df['oancf']).fillna(df[f'dif12_cowc_x'] + df[f'dif12_nncoa_x'])        
        # Total accruals
        df['tacc_x'] = df['oacc_x'] + df['dif12_nfna_x']
        
        # Set accruals to missing if count <= 12
        df.loc[df['count'] <= 12, ['oacc_x', 'tacc_x']] = np.nan
        
        # Operating and free cash flow
        df['ocf_x'] = coalesce(
            df['oancf'],
            df['ni_x'] - df['oacc_x'],
            df['ni_x'] + df['dp'] - df['wcapt'].fillna(0),
        )
        # Free cash flow
        df['fcf_x'] = df['ocf_x'] - df['capx']
        # Cash-based operating profitability (Gerakos et al, 2016)
        df['cop_x'] = df['ebitda_x'] + df['xrd'].fillna(0) - df['oacc_x']
        
        # Drop the count column as it was temporary
        df = df.drop('count', axis=1)
        
        return df

    def standardized_accounting_data(
        self,
        comp_funda: pd.DataFrame,
        comp_fundq: pd.DataFrame,
        convert_to_usd: bool = True,
        me_data: Optional[pd.DataFrame] = None,
        include_helpers_vars: bool = True,
        start_date: str = '1950-01-01'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize Compustat accounting data across frequency and geography.
        
        Parameters:
        -----------
        comp_funda : pd.DataFrame
            Annual fundamental data from Compustat
        comp_fundq : pd.DataFrame
            Quarterly fundamental data from Compustat
        coverage : str
            One of 'na' (North America), 'global', or 'world'
        convert_to_usd : bool
            Whether to convert all currency values to USD
        me_data : pd.DataFrame
            Market equity data with columns [gvkey, eom, me_company]
        include_helpers_vars : bool
            Whether to include helper variables
        start_date : str
            Start date for data in 'YYYY-MM-DD' format
            
        Returns:
        --------
        tuple[pd.DataFrame, pd.DataFrame]
            Standardized annual and quarterly accounting data
        """
        # Define variable groups
        var_groups = {
            'inc': [  # Income statement
                'sale', 'revt', 'gp', 'ebitda', 'oibdp', 'ebit', 'oiadp', 
                'pi', 'ib', 'ni', 'mii', 'cogs', 'xsga', 'xopr', 'xrd',
                'xad', 'xlr', 'dp', 'xi', 'do', 'xido', 'xint', 'spi',
                'nopi', 'txt', 'dvt'
            ],
            'cf': [  # Cash flow
                # Operating
                'oancf', 'ibc', 'dpc', 'xidoc', 'capx', 'wcapt',
                # Financing
                'fincf', 'fiao', 'txbcof', 'ltdch', 'dltis', 'dltr',
                'dlcch', 'purtshr', 'prstkc', 'sstk', 'dv', 'dvc'
            ],
            'bs': [  # Balance sheet
                # Assets
                'at', 'act', 'aco', 'che', 'invt', 'rect', 'ivao', 'ivst',
                'ppent', 'ppegt', 'intan', 'ao', 'gdwl', 're',
                # Liabilities
                'lt', 'lct', 'dltt', 'dlc', 'txditc', 'txdb', 'itcb',
                'txp', 'ap', 'lco', 'lo', 'seq', 'ceq', 'pstkrv',
                'pstkl', 'pstk', 'mib', 'icapt'
            ],
            'other': ['emp']  # Other variables
        }
        
        def process_fundamental_data(df: pd.DataFrame) -> pd.DataFrame:
            """Process either annual or quarterly fundamental data."""
            # Convert date and ensure sorted
            df = df.copy()
            df['datadate'] = pd.to_datetime(df['datadate'])
            df = df.sort_values(['gvkey', 'datadate'])
                            
            na_mask = (
                (df['indfmt'] == 'INDL') &
                (df['datafmt'] == 'STD') &
                (df['popsrc'] == 'D') &
                (df['consol'] == 'C')
            )
            df_na = df[na_mask].copy()
            df_na['source'] = 'NA'                
            return df_na
        
        # Process annual and quarterly data
        annual_data = process_fundamental_data(comp_funda)
        quarterly_data = process_fundamental_data(comp_fundq)
        
        # Convert to USD if requested
        if convert_to_usd:
            #TODO            
            fx_rates = compustat_fx()
            
            # Convert annual data
            annual_data = annual_data.merge(
                fx_rates,
                left_on=['datadate', 'curcd'],
                right_on=['date', 'curcdd'],
                how='left'
            )
            
            for var in var_groups['inc'] + var_groups['cf'] + var_groups['bs']:
                if var in annual_data.columns:
                    annual_data[var] = annual_data[var] * annual_data['fx']
            
            annual_data['curcd'] = 'USD'
            annual_data = annual_data.drop(['fx', 'date', 'curcdd'], axis=1)
            
            # Convert quarterly data similarly
            quarterly_data = quarterly_data.merge(
                fx_rates,
                left_on=['datadate', 'curcdq'],
                right_on=['date', 'curcdd'],
                how='left'
            )
            
            q_vars = [col for col in quarterly_data.columns 
                    if col.endswith('q') or col.endswith('y')]
            for var in q_vars:
                quarterly_data[var] = quarterly_data[var] * quarterly_data['fx']
            
            quarterly_data['curcdq'] = 'USD'
            quarterly_data = quarterly_data.drop(['fx', 'date', 'curcdd'], axis=1)
        
        # Process quarterly data to be comparable with annual
        def calculate_ttm(group: pd.DataFrame, var: str) -> pd.Series:
            """Calculate trailing twelve months sum."""
            return (group[var] + 
                    group[var].shift(1) + 
                    group[var].shift(2) + 
                    group[var].shift(3))
        
        # Process quarterly variables
        quarterly_data = quarterly_data.sort_values(['gvkey', 'fyr', 'fyearq', 'fqtr'])
        
        # Create quarterly specific variables
        quarterly_data['ni_qtr'] = quarterly_data['ibq']
        quarterly_data['sale_qtr'] = quarterly_data['saleq']
        quarterly_data['ocf_qtr'] = quarterly_data['oancfq'].fillna(
            quarterly_data['ibq'] + quarterly_data['dpq'] - quarterly_data['wcaptq'].fillna(0)
        )
        annual_data['ni_qtr'], annual_data['sale_qtr'], annual_data['ocf_qtr'] = [np.nan] * 3
        
        # Calculate TTM for relevant variables
        yrl_vars = {
            'cogsq', 'xsgaq', 'xintq', 'dpq', 'txtq', 'xrdq', 'dvq', 'spiq', 'saleq', 
            'revtq', 'xoprq', 'oibdpq', 'oiadpq', 'ibq', 'niq', 'xidoq', 'nopiq', 'miiq', 
            'piq', 'xiq', 'xidocq', 'capxq', 'oancfq', 'ibcq', 'dpcq', 'wcaptq',
            'prstkcq', 'sstkq', 'purtshrq', 'dsq', 'dltrq', 'ltdchq', 'dlcchq',
            'fincfq', 'fiaoq', 'txbcofq', 'dvtq'
        }
        
        for var in yrl_vars.intersection(quarterly_data.columns):
            ttm_name = var[:-1]
            quarterly_data[ttm_name] = calculate_ttm(quarterly_data, var)
            quarterly_data[ttm_name] = np.where(
                (quarterly_data['gvkey'] != quarterly_data['gvkey'].shift(3)) | 
                (quarterly_data['fyr'] != quarterly_data['fyr'].shift(3)) | 
                (quarterly_data['curcdq'] != quarterly_data['curcdq'].shift(3)) | 
                (calculate_ttm(quarterly_data, 'fqtr') != 10),
                np.nan,
                quarterly_data[ttm_name]
            )
            ytd_col = f"{ttm_name}y"
            if ytd_col in quarterly_data.columns:  # Check if YTD column exists
                quarterly_data[ttm_name] = np.where(
                    quarterly_data[ttm_name].isna() & (quarterly_data['fqtr'] == 4),
                    quarterly_data[ytd_col],
                    quarterly_data[ttm_name],
                )
            quarterly_data = quarterly_data.drop(columns=[var, ytd_col], errors='ignore')
        
        bs_vars = {
            'seqq', 'ceqq', 'pstkq', 'icaptq', 'mibq', 'gdwlq', 'req',
            'atq', 'actq', 'invtq', 'rectq', 'ppegtq', 'ppentq', 'aoq', 'acoq', 'intanq',
            'cheq', 'ivaoq', 'ivstq', 'ltq', 'lctq', 'dlttq', 'dlcq', 'txpq', 'apq', 'lcoq', 
            'loq', 'txditcq', 'txdbq'
        }

        # Rename columns by removing trailing 'q'
        rename_dict = {var: var[:-1] for var in bs_vars.intersection(quarterly_data.columns)}
        # Rename curcdq explicitly
        rename_dict['curcdq'] = 'curcd'
        # Apply renaming
        quarterly_data.rename(columns=rename_dict, inplace=True)
        
        # Add market equity data
        if me_data is not None:
            self.merge_me(annual_data, quarterly_data, me_data)

        # Add helper variables if requested
        if include_helpers_vars:
            annual_data = self.add_helper_vars(annual_data)
            quarterly_data = self.add_helper_vars(quarterly_data)
        
        # Final cleanup and sorting
        annual_data = annual_data.sort_values(['gvkey', 'datadate'])\
            .drop_duplicates(['gvkey', 'datadate'])
        quarterly_data = quarterly_data.sort_values(['gvkey', 'datadate'])\
            .drop_duplicates(['gvkey', 'datadate'])
        
        return annual_data, quarterly_data

    def merge_me(self, annual_data, quarterly_data, me_data):
        me_data = me_data[
            (me_data['primary_sec'] == 1) &
            me_data['me_company'].notna() &
            (me_data['common'] == 1) &
            (me_data['obs_main'] == 1)
        ].copy()
            
        me_data = me_data.groupby(['gvkey', 'eom'])['me_company']\
            .max().reset_index()
        me_data = me_data.rename(columns={'me_company': 'me_fiscal'})
            
        annual_data = annual_data.merge(
            me_data,
            left_on=['gvkey', 'datadate'],
            right_on=['gvkey', 'eom'],
            how='left'
        )
            
        quarterly_data = quarterly_data.merge(
            me_data,
            left_on=['gvkey', 'datadate'],
            right_on=['gvkey', 'eom'],
            how='left'
        )
        return annual_data, quarterly_data


    def create_acc_chars(self, 
                        data: pd.DataFrame,
                        me_data: pd.DataFrame,
                        lag_to_public: int,
                        max_data_lag: int,
                        keep_vars: List[str],
                        suffix: Optional[str] = None) -> pd.DataFrame:
        """
        Create accounting characteristics from standardized Compustat data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe with standardized accounting data
        me_data : pd.DataFrame
            Market equity data
        lag_to_public : int
            Number of months to lag for public availability
        max_data_lag : int
            Maximum number of months to lag data
        keep_vars : List[str]
            List of variables to keep in final output
        suffix : str, optional
            Suffix to add to variable names
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with accounting characteristics
        """
        # Sort and add counter
        df = df.sort_values(['gvkey', 'curcd', 'datadate'])
        df['count'] = df.groupby(['gvkey', 'curcd']).cumcount() + 1
        
        # Basic size measures
        df['assets'] = df['at_x']
        df['sales'] = df['sale_x']
        df['book_equity'] = df['be_x']
        df['net_income'] = df['ni_x']
        
        # Calculate growth characteristics
        for var in self.growth_vars:
            # 1-year growth
            df[f'{var}_gr1'] = df.groupby(['gvkey', 'curcd'])[var].pct_change(periods=12)
            # 3-year growth
            df[f'{var}_gr3'] = df.groupby(['gvkey', 'curcd'])[var].pct_change(periods=36)
        
        # Calculate changes scaled by assets
        for var in self.ch_asset_vars:
            # 1-year change scaled by assets
            df[f'{var}_gr1a'] = (df.groupby(['gvkey', 'curcd'])[var].diff(periods=12) / 
                                df['at_x'])
            mask = (df['count'] <= 12) | (df['at_x'] <= 0)
            df.loc[mask, f'{var}_gr1a'] = np.nan
            
            # 3-year change scaled by assets
            df[f'{var}_gr3a'] = (df.groupby(['gvkey', 'curcd'])[var].diff(periods=36) / 
                                df['at_x'])
            mask = (df['count'] <= 36) | (df['at_x'] <= 0)
            df.loc[mask, f'{var}_gr3a'] = np.nan
        
        # Investment measures
        df['capx_at'] = df['capx'] / df['at_x']
        df['rd_at'] = df['xrd'] / df['at_x']
        
        # Non-recurring items
        df['spi_at'] = df['spi'] / df['at_x']
        df['xido_at'] = df['xido_x'] / df['at_x']
        df['nri_at'] = (df['spi'] + df['xido_x']) / df['at_x']
        
        # Profitability ratios
        # Profit margins
        df['gp_sale'] = df['gp_x'] / df['sale_x']
        df['ebitda_sale'] = df['ebitda_x'] / df['sale_x']
        df['ebit_sale'] = df['ebit_x'] / df['sale_x']
        df['pi_sale'] = df['pi_x'] / df['sale_x']
        df['ni_sale'] = df['ni_x'] / df['sale_x']
        df['nix_sale'] = df['ni'] / df['sale_x']
        df['ocf_sale'] = df['ocf_x'] / df['sale_x']
        df['fcf_sale'] = df['fcf_x'] / df['sale_x']
        
        # Return on assets
        df['gp_at'] = df['gp_x'] / df['at_x']
        df['ebitda_at'] = df['ebitda_x'] / df['at_x']
        df['ebit_at'] = df['ebit_x'] / df['at_x']
        df['fi_at'] = df['fi_x'] / df['at_x']
        df['cop_at'] = df['cop_x'] / df['at_x']
        df['ni_at'] = df['ni_x'] / df['at_x']
        
        # TODO: Add more ratio calculations...
        
        # Expand by public availability
        df['start_date'] = df['datadate'] + pd.DateOffset(months=lag_to_public)
        df['next_start_date'] = df.groupby('gvkey')['start_date'].shift(-1)
        df['end_date'] = pd.DataFrame({
            'from_next_start': df['next_start_date'] - pd.DateOffset(months=1),
            'from_datadate': df['datadate'] + pd.DateOffset(months=max_data_lag)
        }).min(axis=1)        
        # Expand dates
        expanded_df = self._expand_dates(df)
        
        # Convert to USD if needed
        expanded_df = self._convert_to_usd(expanded_df)
        
        # Calculate market-based ratios
        final_df = self._calculate_market_ratios(expanded_df, me_data)
        
        # Keep only selected columns and add suffix if specified
        result = final_df[['source', 'gvkey', 'public_date', 'datadate'] + keep_vars]
        if suffix:
            result = result.rename(columns={col: f"{col}{suffix}" for col in keep_vars})
            result = result.rename(columns={'datadate': f"datadate{suffix}"})
        
        return result.drop_duplicates(['gvkey', 'public_date']).sort_values(['gvkey', 'public_date'])

    def _expand_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand dates for each company"""
        # Create date ranges for each row
        df = df.dropna(subset=['start_date', 'end_date'])
        
        # Generate all date combinations efficiently
        dates_df = pd.DataFrame({
            'gvkey': df['gvkey'].repeat(df.apply(lambda x: len(pd.date_range(x['start_date'], x['end_date'], freq='M')), axis=1)),
            'idx': df.index.repeat(df.apply(lambda x: len(pd.date_range(x['start_date'], x['end_date'], freq='M')), axis=1))
        })
        
        dates_df['public_date'] = pd.concat([
            pd.Series(dates) for dates in 
            df.apply(lambda x: pd.date_range(x['start_date'], x['end_date'], freq='M'), axis=1)
        ]).reset_index(drop=True)
        
        # Merge back with original data
        expanded_df = dates_df.merge(df, left_on=['gvkey', 'idx'], right_index=True)
        expanded_df = expanded_df.drop('idx', axis=1)
        return expanded_df

    def _convert_to_usd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert values to USD using FX rates"""
        # Implement your FX conversion logic here
        return df

    def _calculate_market_ratios(self, df: pd.DataFrame, me_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-based ratios"""
        # Merge with market equity data
        df = df.merge(
            me_data[['gvkey', 'eom', 'me_company']],
            left_on=['gvkey', 'public_date'],
            right_on=['gvkey', 'eom'],
            how='left'
        )
        
        # Calculate enterprise value and market asset value
        df['mev'] = df['me_company'] + df['netdebt_x'] * df['fx']
        df['mat'] = df['at_x'] * df['fx'] - df['be_x'] * df['fx'] + df['me_company']
        
        # Set to nan if invalid
        df.loc[df['mev'] <= 0, 'mev'] = np.nan
        df.loc[df['me_company'] <= 0, 'me_company'] = np.nan
        df.loc[df['mat'] <= 0, 'mat'] = np.nan
        
        return df
    
    def combine_ann_qtr_chars(
        self,
        ann_data: pd.DataFrame,
        qtr_data: pd.DataFrame,
        char_vars: List[str],
        q_suffix: str
    ) -> pd.DataFrame:
        """
        Combine characteristics from annual and quarterly data, preferring more recent quarterly data.
        
        Parameters:
        -----------
        ann_data : pd.DataFrame
            Annual accounting data
        qtr_data : pd.DataFrame
            Quarterly accounting data
        char_vars : List[str]
            List of characteristic variables to process
        q_suffix : str
            Suffix used for quarterly variables
            
        Returns:
        --------
        pd.DataFrame
            Combined data with most recent values
        """
        # Merge annual and quarterly data
        combined_df = pd.merge(
            ann_data,
            qtr_data,
            on=['gvkey', 'public_date'],
            how='left',
            suffixes=('', q_suffix)
        )
        
        # For each characteristic variable, use quarterly value if it's more recent
        for var in char_vars:
            qtr_var = f"{var}{q_suffix}"
            date_var = 'datadate'
            date_var_q = f"datadate{q_suffix}"
            
            # Create mask for when to use quarterly data:
            # 1. When annual data is missing OR
            # 2. When quarterly data exists and is more recent
            mask = (
                combined_df[var].isna() | 
                (~combined_df[qtr_var].isna() & 
                (combined_df[date_var_q] > combined_df[date_var]))
            )
            # Update values based on mask
            combined_df.loc[mask, var] = combined_df.loc[mask, qtr_var]
            # Drop quarterly version of the variable
            combined_df = combined_df.drop(columns=[qtr_var])
        
        # Drop date columns as we can no longer be sure which items accounting dates refer to
        combined_df = combined_df.drop(columns=['datadate', f"datadate{q_suffix}"])
        
        # Remove duplicates and sort
        result = combined_df.drop_duplicates(subset=['gvkey', 'public_date'])\
            .sort_values(['gvkey', 'public_date'])
        
        return result
    
