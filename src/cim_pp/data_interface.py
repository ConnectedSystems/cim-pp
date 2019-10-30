import os
import re
import io as StringIO
import tarfile
import json

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import pandas as pd


zone_tag = re.compile('Zone_([0-9]+)')
ML_to_mm = 100.0
mm_to_ML = 100.0
MILLION = 1000000.0

def get_gz(tar):
    """Get list of files in tarfile ending with 'gz'"""
    batch_results = [tar.extractfile(f) for f in tar.getmembers()
                     if f.name.endswith('gz')]
    return batch_results


def get_input_dict(tar):
    """Load scenario data if available
    
    Returns
    -------
    dict or None
    """
    scen_inputs = [tar.extractfile(f) for f in tar.getmembers()
                   if f.name.endswith('_input_dict.pkl')]
    
    num_ins = len(scen_inputs)
    if num_ins > 1:
        raise ValueError("Cannot load more than one master scenario dict!\n{}".format(scen_inputs))
    elif num_ins == 0:
        return None
    
    scenarios = pickle.load(scen_inputs[0], encoding='bytes')
    scenarios = {str(k): v for k, v in scenarios[0].items()}

    return scenarios
# End get_input_dict()


def filter_condition(haystack, needle, ignore=None):
    if isinstance(needle, str):
        needle = [needle]

    if ignore:
        if isinstance(ignore, str):
            ignore = [ignore]

        for ig in ignore:
            if ig in haystack:
                return False
    # End if

    for n in needle:
        if n in haystack:
            return True
    # End for

    return False
# End filter_condition()


def load_from_tarfile(batch_results, filter_cond, extract_func, timeseries=False):
    """
    Load result sets from a tar file.

    Parameters
    ----------
    * batch_results : list, of gz files
    * filter_cond : list-like 2 elements, first element indicates things to include, second element items to exclude
    * extract_func : function, data extraction method to use
    * timeseries : bool, extract time series data as well. Defaults to False (indicators only).

    Returns
    -------
    * tuple[dict, dict] -> dictionary of results and scenario information
    """
    res_dict = {}

    for gz in batch_results:
        set_name = os.path.splitext(os.path.basename(gz.name))[0]
        with tarfile.open(fileobj=gz) as res_gz:
            result_set = [res_gz.extractfile(f) for f in res_gz 
                          if filter_condition(f.name, filter_cond[0], filter_cond[1])]
            res_dict = extract_func(result_set, set_name, res_dict, timeseries)
    # End for

    return res_dict
# End load_from_tarfile()

def load_input_json(batch_results):
    """Loads inputs used for a run stored in JSON format.
    
    Returns
    -------
    * dict, elements will be empty string if JSON file not found.
    """
    scenario_info = {}
    for gz in batch_results:
        set_name = os.path.splitext(os.path.basename(gz.name))[0]
        with tarfile.open(fileobj=gz) as res_gz:
            inputs = [res_gz.extractfile(f) for f in res_gz
                      if 'inputs_used.json' in f.name]

            if len(inputs) > 0:
                inputs = inputs[0]
                scenario_info[set_name] = json.loads(inputs.read())
            else:
                scenario_info[set_name] = ""
    # End for

    return scenario_info
# End load_input_json()


# @asyncio.coroutine
def read_result_csv(result_file, warmup_period=3, end_step=None, **kwargs):
    """Read in data from CSV file with common pandas settings.

    Parameters
    ----------
    * result_file : file obj or str, of data file to read in
    * warmup_period : int, number of time steps to skip

    Returns
    ---------
    * Pandas DataFrame
    """
    tmp_df = pd.read_csv(result_file, **kwargs)
    return tmp_df.loc[warmup_period:end_step, :].reset_index(drop=True)
# End read_result_csv()


def extract_farm_zone_data(batch, set_name, batch_dict, timeseries=True):
    # constrain to shortest expected time frame
    # as historic and future scenarios differ in length
    # Need to make these comparable for analysis.
    end_exclusive_step = 31

    for res in batch:
        zone_id = zone_tag.search(res.name).group()
        tmp = batch_dict.get(zone_id, {})
        batch_dict[zone_id] = tmp

        batch_dict[zone_id][set_name] = read_result_csv(res, parse_dates=True, end_step=end_exclusive_step)
        tmp = batch_dict[zone_id][set_name]
        tmp['Total Profit ($M)'] = (tmp['Total Field Profit ($/Ha)'] * tmp['Total Field Area (Ha)']) / MILLION

        tmp['$/ML'] = ((tmp['Total Profit ($M)'] * MILLION) /
                        ((tmp["Irrigation (mm)"] / mm_to_ML) * tmp["Irrigated Area (Ha)"]))
    # End for

    return batch_dict
# End extract_farm_zone_data()


def extract_farm_gw_data(batch, set_name, batch_dict, timeseries=False):
    end_exclusive_step = 31
    for res in batch:
        tmp_df = read_result_csv(res, end_step=end_exclusive_step)
        for zone_id in tmp_df:
            r_zone_id = zone_id.replace(' ', '_')
            tmp_dict = batch_dict.get(r_zone_id, {})
            batch_dict[r_zone_id] = tmp_dict

            batch_dict[r_zone_id][set_name] = pd.Series(tmp_df[zone_id].values)
        # End for
    # End for

    for zone_id in batch_dict:
        batch_dict[zone_id] = pd.DataFrame(batch_dict[zone_id])

    return batch_dict
# End extract_farm_gw_data()


def extract_policy_gw_data(batch, set_name, batch_dict, timeseries):
    end_exclusive_step = 31
    for res in batch:
        tmp_df = read_result_csv(res, end_step=end_exclusive_step)
        for col in tmp_df:
            tmp_dict = batch_dict.get(col, {})
            batch_dict[col] = tmp_dict

            batch_dict[col][set_name] = pd.Series(tmp_df[col].values)
        # End for
    # End for

    for col in batch_dict:
        batch_dict[col] = pd.DataFrame(batch_dict[col])

    return batch_dict
# End extract_policy_gw_data()


def extract_sw_data(batch, set_name, batch_dict, timeseries):
    end_exclusive_step = 31
    for res in batch:
        tmp_df = read_result_csv(res, end_step=end_exclusive_step)
        for col in tmp_df:
            tmp_dict = batch_dict.get(set_name, {})
            batch_dict[set_name] = tmp_dict

            batch_dict[set_name][col] = pd.Series(tmp_df[col].values)
        # End for

        for col in batch_dict:
            batch_dict[set_name] = pd.DataFrame(batch_dict[set_name])

    return batch_dict
# End extract_sw_data()


def extract_farm_data(batch, set_name, batch_dict, timeseries):
    """Extract farm data metrics.
    """
    farm_indicators = {
        'Avg. Annual Profit ($M)': 0.0,
        # 'Avg. SW Irrigation Usage (mm)': 0.0,
        # 'Avg. GW Irrigation Usage (mm)': 0.0,
        'Avg. $/ML': 0.0,
        'Avg. GPWUI': 0.0,
        'Avg. IWUI': 0.0,
        'Income Volatility': 0.0,
        'GW Used (ML)': 0.0
    }

    # Farm
    farm_ts = {}

    _profit_tmp = {}
    _dollar_ML = {}
    _gpwui = {}
    _iwui = {}
    _ML = {}
    _costs = {}
    _irrigated_area = {}
    _dryland_area = {}
    _total_area = {}
    _sw_amount = {}
    _gw_amount = {}
    _rainfall = {}
    _ML_used = {}

    _irrigated_crop_yield = {}
    _dryland_crop_yield = {}
    _total_crop_yield = {}
    _avg_crop_yield_per_ha = {}

    # # Ecology
    # _platypus = {}
    # _fish = {}
    # _riverredgums = {}
    
    # Calculate indices for all identified files
    end_exclusive_step = 31
    for res in batch:
        zone_id = zone_tag.search(res.name).group()

        # Get total profit for each zone
        # Constrain to shortest expected time frame
        # as historic and future scenarios differ in length
        # Need to make these comparable for analysis.
        farm_res = read_result_csv(res, parse_dates=True, end_step=end_exclusive_step)

        field_area = farm_res['Total Field Area (Ha)']
        _total_area[zone_id] = field_area
        _profit_tmp[zone_id] = (farm_res['Total Field Profit ($/Ha)'] * field_area).values

        irrigated_area = farm_res['Irrigated Area (Ha)']
        _irrigated_area[zone_id] = irrigated_area
        dryland_area = farm_res['Dryland Area (Ha)']
        _dryland_area[zone_id] = dryland_area

        gross_yield = farm_res['P+I Crop Yield (t/Ha)']
        _irrigated_crop_yield[zone_id] = gross_yield
        dryland_yield = farm_res['P Crop Yield (t/Ha)']
        _dryland_crop_yield[zone_id] = dryland_yield
        _avg_crop_yield_per_ha[zone_id] = farm_res['Avg. Crop Yield (t/Ha)']

        _total_crop_yield[zone_id] = (gross_yield * irrigated_area) + (dryland_yield * dryland_area)

        _costs[zone_id] = farm_res['Costs ($/Ha)'] * field_area
        _sw_amount[zone_id] = farm_res['Surface Water (mm)']
        _gw_amount[zone_id] = farm_res['Groundwater (mm)']
        _rainfall[zone_id] = farm_res['Rainfall (mm)']

        # mm of water / 100.0 = ML/Ha
        # ML/Ha * Field Area = Total ML
        # total profit / Total ML = $/ML
        ML_per_ha = (farm_res['Irrigation (mm)'] / 100.0)
        _ML[zone_id] = ML_per_ha

        ML_used = ML_per_ha * irrigated_area

        _ML_used[zone_id] = ML_used
        # ML_used = ML_used.replace(0.0, np.nan)

        # _dollar_ML[zone_id] = (_profit_tmp[zone_id] / ML_used.values)
        _gpwui[zone_id] = farm_res['GPWUI']
        _iwui[zone_id] = farm_res['IWUI']
    # End for

    # Calculate average annual profit
    collated = pd.DataFrame(_profit_tmp)
    annual_sum = collated.sum(axis=1)
    farm_indicators['Avg. Annual Profit ($M)'] = annual_sum.mean() / MILLION
    farm_indicators['Avg. Annual Profit ($)'] = annual_sum.mean()

    farm_ts['Total Annual Profit ($)'] = annual_sum

    farm_ts['Irrigated Area (Ha)'] = pd.DataFrame(_irrigated_area).sum(axis=1)
    farm_ts['Dryland Area (Ha)'] = pd.DataFrame(_dryland_area).sum(axis=1)
    farm_ts['Total Field Area (Ha)'] = pd.DataFrame(_total_area).sum(axis=1)

    total_area = farm_ts['Total Field Area (Ha)'].mean()

    farm_indicators['Avg. Irrigated Area (Ha)'] = farm_ts['Irrigated Area (Ha)'].mean() / total_area
    farm_indicators['Avg. Dryland Area (Ha)'] = farm_ts['Dryland Area (Ha)'].mean(
    ) / total_area

    farm_ts['Costs ($)'] = pd.DataFrame(_costs).sum(axis=1)

    sw_df = pd.DataFrame(_sw_amount)
    gw_df = pd.DataFrame(_gw_amount)

    farm_ts['SW Used (ML/Ha)'] = sw_df.mean(axis=1) / mm_to_ML  # Converting mm to ML
    farm_ts['GW Used (ML/Ha)'] = gw_df.mean(axis=1) / mm_to_ML

    farm_indicators['GW Used (ML)'] = gw_df.sum(axis=1).sum() / mm_to_ML
    farm_indicators['SW Used (ML)'] = sw_df.sum(axis=1).sum() / mm_to_ML

    farm_indicators['Total Rainfall (mm)'] = pd.DataFrame(_rainfall).sum(axis=1).sum()

    irrig_crop_yield = pd.DataFrame(_irrigated_crop_yield)
    dry_crop_yield = pd.DataFrame(_dryland_crop_yield)
    farm_ts['Irrigated Yield (t/Ha)'] = irrig_crop_yield.mean(axis=1)
    farm_ts['Dryland Yield (t/Ha)'] = dry_crop_yield.mean(axis=1)

    # Catchment average yield
    irrig_crop_yield = farm_ts['Irrigated Yield (t/Ha)'] * \
                        farm_ts['Irrigated Area (Ha)']
    dryland_crop_yield = farm_ts['Dryland Yield (t/Ha)'] * \
                            farm_ts['Dryland Area (Ha)']

    # farm_ts['Avg. Yield (t/Ha)'] = pd.DataFrame(_avg_crop_yield_per_ha).mean(axis=1)
    farm_ts['Avg. Yield (t/Ha)'] = (irrig_crop_yield +
                                    dryland_crop_yield) / farm_ts['Total Field Area (Ha)']
    farm_ts['Total Yield (t)'] = pd.DataFrame(_total_crop_yield).sum(axis=1)

    # collated = pd.DataFrame(_dollar_ML)
    collated_ML_used = pd.DataFrame(_ML_used)    
    total_ML_used = collated_ML_used.sum()  # sum for each Zone

    farm_indicators['Avg. ML'] = total_ML_used.mean()

    avg_profit_ML = (annual_sum / total_ML_used.sum()).mean()
    farm_indicators['Avg. $/ML'] = avg_profit_ML
    farm_ts['Avg. $/ML'] = avg_profit_ML

    collated_ML = pd.DataFrame(_ML)
    avg = collated_ML.mean(axis=1)
    farm_indicators['Avg. ML/Ha'] = avg.mean()
    farm_ts['Avg. ML/Ha'] = avg

    collated = pd.DataFrame(_gpwui)
    avg = collated.mean(axis=1)
    farm_indicators['Avg. GPWUI'] = avg.mean()
    farm_ts['Avg. GPWUI'] = avg

    collated = pd.DataFrame(_iwui)
    avg = collated.mean(axis=1)
    farm_indicators['Avg. IWUI'] = avg.mean()
    farm_ts['Avg. IWUI'] = avg

    farm_indicators['Income Volatility'] = (annual_sum / 1000000.0).std() / farm_indicators['Avg. Annual Profit ($M)']

    try:
        batch_dict['farm']

        if timeseries:
            batch_dict['farm_timeseries']
    except KeyError:
        batch_dict['farm'] = {}

        if timeseries:
            batch_dict['farm_timeseries'] = {}

    batch_dict['farm'][set_name] = farm_indicators

    if timeseries:
        batch_dict['farm_timeseries'][set_name] = farm_ts

    return batch_dict
# End extract_farm_data()


def extract_ecology_data(batch, set_name, batch_dict, timeseries):
    """Read and mung ecology model results into dict.
    """
    r_indices = {}
    related_to = ['platypus', 'fish', 'RRG', 'rec']

    # constrain to shortest expected time frame
    # as historic and future scenarios differ in length
    # Need to make these comparable for analysis.
    end_exclusive_step = 31  # years

    for res in batch:
        r_name = res.name.split('/')[-1].replace('.csv', '')
        if 'gw_ecology' in r_name:
            # Skip this file as we don't need it - info contained in other indices
            continue

        try:
            species_data = [rt for rt in related_to if rt in r_name][0]
        except IndexError as e:
            print(r_name)
            raise IndexError(e)


        # Get ecological indicators
        if 'rec' in r_name:
            rec_start = int(365*3)+1
            rec_end = 10951
            eco_res = read_result_csv(res, parse_dates=True, 
                                      warmup_period=rec_start, 
                                      end_step=rec_end)
        else:
            eco_res = read_result_csv(res, parse_dates=True, 
                                      end_step=end_exclusive_step)

        eco_res.columns = [c.replace('water_suitability', 'RRG')
                           for c in eco_res.columns]

        target_cols = [col for col in eco_res.columns if any(
            [col.startswith(rt) and col.endswith('index') for rt in related_to])]

        try:
            old = r_indices[species_data]
            new = eco_res.loc[:, target_cols]
            new.columns = ["{}".format(r_name) for c in new.columns]
            r_indices[species_data] = pd.concat([old, new], axis=1)
        except KeyError as e:
            r_indices[species_data] = eco_res.loc[:, target_cols]
            cols = r_indices[species_data].columns
            r_indices[species_data].columns = ["{}".format(r_name) for c in cols]
        # End try
    # End for

    for rind in list(r_indices.values()):
        rind['Average'] = rind.mean(axis=1)

    batch_dict['ecology_indices'] = batch_dict.get('ecology_indices', {})
    if timeseries:
        batch_dict['ecology_timeseries'] = batch_dict.get('ecology_timeseries', {})
        batch_dict['ecology_timeseries'][set_name] = r_indices

    batch_dict['ecology_indices'][set_name] = {}
    eco_index = batch_dict['ecology_indices'][set_name]
    eco_index['Ecology Index'] = 0.0

    for rt, data in list(r_indices.items()):

        if rt == 'rec':
            eco_index['Recreation Index'] = data.Average.mean()
            continue
        
        if rt == 'RRG':
            # RRG seems to range to 100
            data.Average = data.Average / 100.0

        eco_index['Ecology Index'] += data.Average.mean()
    # End for
    
    # Divide by number of non-recreational indices to get average of averages
    eco_index['Ecology Index'] /= (len(list(r_indices.keys())) - 1)

    return batch_dict
# End extract_ecology_data()


def calc_gw_trigger_indicator(thresholds, bid_name, pgw):
    total_score = 0.0
    samples = float(len(pgw.index))
    tmp = pgw[bid_name]
    threshold_order = sorted(thresholds.keys())
    for thres in threshold_order:
        score = thresholds[thres]

        breached = (thres - tmp) >= 0.0
        num_times_trig_reached = tmp[breached].count()

        num_years_below = (samples - num_times_trig_reached)
        total_score += num_years_below * score

        if num_years_below == samples:
            break

        tmp = tmp[~breached]

    # End for

    return total_score
# End calc_gw_trigger_indicator()


def calc_gw_movement_indicator(pgw, bid_name, thresholds):
    """Calculate movement average of groundwater table relative 
    to lowest trigger threshold (in mAHD). Negative values indicate lowest 
    trigger level was exceeded.
    """
    first = pgw.head(1)[bid_name].iloc[0]
    last_level = pgw.tail(1)[bid_name].iloc[0]
    lowest_threshold = min(thresholds.keys())

    return (last_level - lowest_threshold) / (first - lowest_threshold)
# End calc_gw_movement_indicator()


def calc_trigger_indicator(scenario, set_name, batch_dict, timeseries):
    """IN PROGRESS
    """
    # Bad practice I know, but trying to keep things contained now.
    critical_levels = {
        '62589': {
            'current': {
                131.8: 1.0,
                115.8: 0.75,
                113.8: 0.5,
                110.8: 0.0,
            },
            'proposed': {
                131.8: 1.0,
                115.8: 1.0,
                113.8: 1.0,
                110.8: 0.0,
            }
            
        },
        '79324': {
            'current': {
                97.1: 1.0,
                82.1: 0.75,
                79.1: 0.5,
                76.1: 0.4,
                73.1: 0.0,
            },
            'proposed': {
                97.1: 1.0,
                82.1: 1.0,
                79.1: 1.0,
                76.1: 0.4,
                73.1: 0.0,
            }
        }
    }

    # constrain to shortest expected time frame
    # as historic and future scenarios differ in length
    # Need to make these comparable for analysis.
    start_step = int(365.25*3)+1
    end_exclusive_step = 10951
    for res in scenario:

        # Get level data at trigger bores
        # Data should be in mAHD
        pgw = read_result_csv(res, warmup_period=start_step, parse_dates=True, end_step=end_exclusive_step)

        pgw.columns = ['Bore {}'.format(c) 
                       for c in pgw.columns]

        c_res = {}
        p_res = {}
        m_res = {}
        scen_tf = float(len(pgw.index))
        num_bores = len(pgw.columns)
        for bid, trig_level in list(critical_levels.items()):
            bid_name = 'Bore {}'.format(bid)

            c_score = calc_gw_trigger_indicator(trig_level['current'], 
                                                    bid_name,
                                                    pgw)
            p_score = calc_gw_trigger_indicator(trig_level['proposed'], 
                                                    bid_name,
                                                    pgw)
            
            m_score = calc_gw_movement_indicator(
                pgw, bid_name, trig_level['current'])

            c_res[bid_name] = c_score / scen_tf
            p_res[bid_name] = p_score / scen_tf

            m_res[bid_name] = m_score

        # End for

        if timeseries:
            batch_dict['gw_policy_levels_timeseries'] = batch_dict.get(
                'gw_policy_levels_timeseries', {})
            batch_dict['gw_policy_levels_timeseries'][set_name] = pgw
        
        c_rules = 'gw_trigger_current_rules'
        batch_dict[c_rules] = batch_dict.get(c_rules, {})
        batch_dict[c_rules][set_name] = sum(c_res.values()) / num_bores

        p_rules = 'gw_trigger_proposed_rules'
        batch_dict[p_rules] = batch_dict.get(p_rules, {})
        batch_dict[p_rules][set_name] = sum(p_res.values()) / num_bores

        # If any reference bore level is below the lowest permitted level
        # then halve the score
        m_vals = list(m_res.values())
        if any(m <= 0.0 for m in m_vals):
            m_vals = [ms if ms <= 0.0 else ms / 2.0 for ms in m_vals]

        m_score = sum(m_vals)

        n_dist = 'gw_trigger_norm_dist'
        batch_dict[n_dist] = batch_dict.get(n_dist, {})
        batch_dict[n_dist][set_name] = m_score / num_bores
    # End for

    return batch_dict
# End calc_trigger_indicator()


def cache_me(func):
    def wrap(*args, **kwargs):
        tmp_dir = '../tmp/'
        arg_hash = hash(frozenset(args))
        d_hash = hash(frozenset(list(kwargs.items())))
        cache_name = func.__name__ + '_' + str(arg_hash) + '_' + str(d_hash)

        cache_pth = os.path.join(tmp_dir, cache_name)
        if os.path.exists(cache_pth):
            with open(cache_pth, 'rb') as cfn:
                res = pickle.load(cfn)
        else:
            res = func(*args, **kwargs)
            with open(cache_pth, 'wb') as cfn:
                pickle.dump(res, cfn)

        return res
    # End wrap()

    return wrap
# End cache_wrapper()


def filter_filelist(fp, conds):
    """Returns set of desired file objects from a tarfile pointer.

    Parameters
    ----------
    * fp : tarfile pointer
    * conds : list-like with 2 elements, filename element to include, and exclude.

    Returns
    ----------
    * subset of file list
    """
    file_list = [fp.extractfile(f) for f in fp
                 if filter_condition(f.name, conds[0], conds[1])]
    return file_list
# End filter_filelist


@cache_me
def catchment_indicators(batch_fn, timeseries=False):
    """Load in results from given tar file.

    Parameters
    ----------
    * fn : str, filename of tar file.
    * timeseries : bool, if true will store time series data as well.

    Returns
    ----------
    * tuple[dict, dict], of indicators and scenario run info
    """
    with tarfile.open(batch_fn) as tar:
        scenario_info = get_input_dict(tar)

        batch_dict = {}
        batch_results = get_gz(tar)

        for gz in batch_results:
            set_name = os.path.splitext(os.path.basename(gz.name))[0]
            with tarfile.open(fileobj=gz) as res_gz:

                farm_files = filter_filelist(res_gz, ('Zone', 'ts_results'))
                batch_dict = extract_farm_data(farm_files,
                                             set_name, 
                                             batch_dict, 
                                             timeseries)

                eco_files = filter_filelist(res_gz, ('ecology', ''))
                batch_dict = extract_ecology_data(eco_files,
                                                  set_name, 
                                                  batch_dict, 
                                                  timeseries)

                pgw_files = filter_filelist(res_gz, ('gw_policy', ''))
                batch_dict = calc_trigger_indicator(pgw_files,
                                                    set_name, 
                                                    batch_dict, 
                                                    timeseries)

    results = batch_dict

    return results, scenario_info
# End catchment_indicators()


def farm_zone_results(fn):
    """Load in farm results for each zone from given tar file.

    Parameters
    ----------
    * fn : str, filename of tar file.

    Returns
    ----------
    * dict, of farm results
    """
    with tarfile.open(fn) as tar:
        batch_results = get_gz(tar)
        farm_results = load_from_tarfile(batch_results,
                                         ('Zone', 'ts_results'),
                                         extract_farm_zone_data,
                                         timeseries=True)
        
        # Get Farm Zone GW levels as well
    # End with

    return farm_results
# End farm_zone_results()


def get_farm_gw_results(fn):
    """Load in groundwater results from given tar file.

    Parameters
    ----------
    * fn : str, filename of tar file.
    """
    with tarfile.open(fn) as tar:
        batch_results = get_gz(tar)
        gw_results = load_from_tarfile(batch_results,
                                       ('gw_farm', ''),
                                       extract_farm_gw_data)
    # End with

    return gw_results
# End get_farm_gw_results()


def get_policy_gw_results(fn):
    """Load in groundwater results from given tar file.

    Parameters
    ----------
    * fn : str, filename of tar file.
    """
    with tarfile.open(fn) as tar:
        batch_results = get_gz(tar)
        pol_gw_results = load_from_tarfile(batch_results,
                                           ('policy_gw_int', ''),
                                           extract_policy_gw_data)
    # End with

    return pol_gw_results
# End get_policy_gw_results()


def get_sw_results(fn, timeseries=False):
    with tarfile.open(fn) as tar:
        batch_results = get_gz(tar)
        sw_results = load_from_tarfile(batch_results,
                                       ('int_sw_data', ''),
                                       extract_sw_data,
                                       timeseries)

    return sw_results
# End get_sw_results()


def extract_tar_data(archive_fn):
    """
    Parameters
    ----------
    * archive_fn: str, filename of archive to extract data from
    """
    if tarfile.is_tarfile(archive_fn):
        data_archive = tarfile.open(archive_fn)
    else:
        raise RuntimeError("Invalid data archive file! {}".format(archive_fn))
    # End if

    file_names_data = list(zip(data_archive.getnames(), data_archive.getmembers()))
    data_collection = {}
    for out_fn, out_datafile in file_names_data:
        # should we read in based on file basename, or should we use it as a matching ID?
        ext = os.path.splitext(out_fn)[1]
        if ext not in ('.csv', '.dat'):
            continue
        fn = os.path.basename(out_fn)
        fn_data = data_archive.extractfile(out_datafile)
        if not fn_data:
            continue

        out_data = StringIO.StringIO(fn_data.read())
        data_collection[fn] = pd.read_csv(out_data)
    # End for

    data_archive.close()

    return data_collection
# End extract_tar_data()


def get_master_input(tar):
    """Retrieve as a dataframe the master scenario parameter listing."""
    master_input = [i for i in tar.getmembers() if i.name.endswith('.csv')][0]    
    fdata = tar.extractfile(master_input)
    batch_inputs = pd.read_csv(fdata, index_col=0)
    # batch_inputs = batch_inputs.set_index('scenario_run_id', drop=True)

    return batch_inputs
# End get_master_input()


def get_ema_design(tar):
    """Retrieve the EMA workbench sampling design data."""
    ema_design = [i for i in tar.getmembers() 
                  if i.name.endswith('_ema_designs.pkl')][0]
    fdata = tar.extractfile(ema_design)
    designs = pickle.load(fdata)
    return designs
# End get_ema_design()


def get_param_bounds(tar):
    """Extract parameter bound data file from tarfile.

    Parameters
    ----------
    * tar : file pointer, to tarfile

    Returns
    ----------
    * DataFrame, of input bounds
    """
    bound_fn = [i for i in tar.getmembers() if i.name.endswith('_param_bounds.dat')][0]
    fdata = tar.extractfile(bound_fn)
    input_bounds = pd.read_csv(fdata, index_col=0)
    return input_bounds
# End get_param_bounds()

