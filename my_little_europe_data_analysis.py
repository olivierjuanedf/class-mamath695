from long_term_uc.common.long_term_uc_io import OUTPUT_DATA_ANALYSIS_FOLDER
from long_term_uc.utils.basic_utils import get_period_str
from long_term_uc.include.dataset import Dataset
from long_term_uc.include.dataset_analyzer import ANALYSIS_TYPES
from long_term_uc.include.uc_timeseries import UCTimeseries
from long_term_uc.utils.read import read_and_check_data_analysis_params, read_and_check_uc_run_params


data_analyses = read_and_check_data_analysis_params()

usage_params, eraa_data_descr, uc_run_params = read_and_check_uc_run_params()

uc_period_msg = get_period_str(period_start=uc_run_params.uc_period_start, 
                               period_end=uc_run_params.uc_period_end)

# loop over the different cases to be analysed
for elt_analysis in data_analyses:
    print(elt_analysis)
    # set UC run params to the ones corresponding to this analysis
    uc_run_params.set_countries(countries=[elt_analysis.country])
    uc_run_params.set_target_year(year=elt_analysis.year)
    uc_run_params.set_climatic_year(climatic_year=elt_analysis.climatic_year)
    # Attention check at each time if stress test based on the set year
    uc_run_params.set_is_stress_test(avail_cy_stress_test=eraa_data_descr.available_climatic_years_stress_test)
    # And if coherent climatic year, i.e. in list of available data
    uc_run_params.coherence_check_ty_and_cy(eraa_data_descr=eraa_data_descr, stop_if_error=True)

    print(f"Read needed ERAA ({eraa_data_descr.eraa_edition}) data for period {uc_period_msg}")
    # initialize dataset object
    eraa_dataset = Dataset(source=f"eraa_{eraa_data_descr.eraa_edition}", 
                        agg_prod_types_with_cf_data=eraa_data_descr.agg_prod_types_with_cf_data, 
                        is_stress_test=uc_run_params.is_stress_test)

    if elt_analysis.data_subtype is not None:
        subdt_selec = [elt_analysis.data_subtype]
    else:
        subdt_selec = None
    eraa_dataset.get_countries_data(uc_run_params=uc_run_params,
                                    aggreg_prod_types_def=eraa_data_descr.aggreg_prod_types_def,
                                    datatypes_selec=elt_analysis.data_type, subdt_selec=subdt_selec)
    # create Unit Commitment Timeseries object from data read
    uc_timeseries = UCTimeseries(name=f"{elt_analysis.data_type}-{elt_analysis.country}", 
                                 data_type=elt_analysis.get_full_datatype())
    # And apply calc./plot
    if elt_analysis.analysis_type == ANALYSIS_TYPES.plot:
        uc_timeseries.plot(output_dir=OUTPUT_DATA_ANALYSIS_FOLDER)
    elif elt_analysis.analysis_type == ANALYSIS_TYPES.plot_duration_curve:
        uc_timeseries.plot_duration_curve(output_dir=OUTPUT_DATA_ANALYSIS_FOLDER)