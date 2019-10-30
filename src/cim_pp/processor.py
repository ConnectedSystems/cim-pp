import pandas as pd
import warnings


def collate_columns(data, column, reset_index=True):
    """Collate specified column from different DataFrames

    Parameters
    ----------
    * data : dict, of pd.DataFrames organized as {ZoneID: {ScenarioID: result_dataframe}}
    * column : str, name of column to collate into single dataframe

    Returns
    ----------
    * dict, of pd.DataFrame of column data collated for each zone
    """
    collated = {}
    for zone, res in list(data.items()):
        # need pd.Series to allow columns of different lengths
        zone_d = {scen: pd.Series(res[scen].loc[:, column].values) for scen in res}
        collated[zone] = pd.DataFrame(zone_d)
        if reset_index:
            collated[zone].reset_index(drop=True, inplace=True)
    # End for

    return collated
# End collate_columns()


def generate_ts_indicators(data, warmup_years=3, years_offset=3):
    """
    Generate the normalized indicators for a time series.

    Parameters
    ----------
    * data : pd.DataFrame, dataframe to extract data from
    * warmup_years : int, number of years that represent warmup period
    * years_offset : int, number of years to offset by

    Returns
    ----------
    * tuple[List], time index and values
    """
    index = []
    values = []

    offset_dt = pd.DateOffset(years=years_offset - 1)

    # Initial datetimes
    past_date = data.index[0] + pd.DateOffset(years=warmup_years)
    curr_date = past_date + offset_dt
    start_indicat = data[past_date:curr_date].sum()
    while(curr_date < data.index[-1]):

        index.append(curr_date)
        indicat = data[past_date:curr_date].sum() / start_indicat if start_indicat > 0.0 else 0.0
        values.append(indicat)

        curr_date = curr_date + pd.DateOffset(years=1)
        past_date = curr_date - offset_dt
    # End while

    return index, values
# End generate_ts_indicators()


def calc_perc_change(series):
    """Calculate percent change from first value in a series

    Parameters
    ----------
    * series : pd.Series, array of values

    Returns
    ----------
    * pd.Series, of percentage change from first value in series.
    """
    msg = """
    This currently calculates relative change from first series.
    We'd want this to display relative change from historic average!
    """
    warnings.warn(msg, FutureWarning)
    first_val = series[0]
    denom = (first_val * (first_val / abs(first_val)))
    return series.apply(lambda x: ((x - first_val) / denom) * 100.0)
# End calc_perc_change()


def determine_constant_factors(df, verbose=True):
    """Determine the column index positions of constant factors.

    Parameters
    ----------
    * df : DataFrame, of parameter bounds

    Returns
    ----------
    * list : of 0-based indices indicating position of constant factors.
    """
    num_vars = len(df.columns)
    constant_idx = [idx for idx in range(num_vars)
                    if np.all(df.iloc[:, idx].value_counts() == df.iloc[:, idx].count())]

    if verbose:
        num_vars, num_consts = len(df.columns), len(constant_idx)
        print(("Number of Constant Parameters: {} / {} | {} params vary".format(num_consts, num_vars, abs(num_consts - num_vars))))

    return constant_idx
# End determine_constant_factors()


def strip_constants(df, indices):
    """Remove columns from DF that are constant input factors.

    Parameters
    ----------
    * df : DataFrame, of input parameters used in model run(s).
    * indices : list, of constant input factor index positions.

    Returns
    ----------
    * copy of `df` modified to exclude constant factors
    """
    df = df.copy()

    const_col_names = df.iloc[:, indices].columns
    df = df.loc[:, ~df.columns.isin(const_col_names)]

    return df
# End strip_constants()


def identify_climate_scenario_run(scen_info, target_scen):
    """Identify the first matching run for a given climate scenario.

    Returns
    ----------
    * str or None, run id
    """
    for scen in scen_info:
        if target_scen in scen_info[scen]['climate_scenario']:
            return scen
    # End for

    return None
# End identify_climate_scenario_run()


def identify_scenario_climate(scen_info, target_run):
    """Given a run id, return its climate scenario.
    """
    try:
        c_scen = scen_info[target_run]['climate_scenario']
    except KeyError:
        s_id = "_".join(target_run.split('_')[0:4])
        match = {k: v for k, v in scen_info.items() if k.startswith(s_id)}
        c_scen = match[list(match.keys())[0]]['climate_scenario']
    # End try

    return c_scen
# End identify_scenario_climate()


def sort_climate_order(df, scenario_info):
    """Sort the climate scenarios based on their conceptual names
    "worst", "maximum", "best"

    Parameters
    ----------
    * df : DataFrame, with climate scenario names in columns.
    * scenario_info : dict, of scenario information organized by scenario run id.

    Returns
    ----------
    * copy of `df` modified to exclude constant factors

    """
    SORT_ORDER = {
        "historic": 0,
        "worst_case_rcp45_2016": 1,
        "worst_case_rcp45_2036": 2,
        "worst_case_rcp85_2016": 3,
        "worst_case_rcp85_2036": 4,
        "maximum_consensus_rcp45_2016": 5,
        "maximum_consensus_rcp45_2036": 6,
        "maximum_consensus_rcp85_2016": 7,
        "maximum_consensus_rcp85_2036": 8,
        "best_case_rcp45_2016": 9,
        "best_case_rcp45_2036": 10,
        "best_case_rcp85_2016": 11,
        "best_case_rcp85_2036": 12,
    }

    cols = [identify_scenario_climate(scenario_info, run_id)
            for run_id in df.columns]

    if len(set(cols)) == 1:
        # All entries are for the same climate scenario
        return df

    col_map = {run_id: identify_scenario_climate(scenario_info, run_id)
               for run_id in df.columns}

    df = df.rename(index=str, columns=col_map)

    cols.sort(key=lambda val: SORT_ORDER['_'.join(
        val.split('_')[0:4]).split('-')[0]])
    df = df.loc[:, cols]
    return df
# End sort_climate_order()
