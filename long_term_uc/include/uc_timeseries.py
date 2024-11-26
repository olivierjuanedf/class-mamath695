import os
from dataclasses import dataclass
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd

from long_term_uc.utils.plot import simple_plot


@dataclass
class UCTimeseries:
    name: str = None
    values: np.ndarray = None
    unit: str = None
    dates: List[datetime] = None

    def from_df_col(self, df: pd.DataFrame, col_name: str, unit: str = None):
        self.name = col_name
        self.values = np.array(df[col_name])
        if unit is not None:
            self.unit = unit
    
    def plot_duration_curve(self, output_dir: str, as_a_percentage: bool = False) -> np.ndarray:
        # sort values in descending order
        vals_desc_order = np.sort(self.values)[::-1]
        # this calculation is done assuming uniform time-slot duration
        duration_curve = np.arange(1, len(vals_desc_order) + 1)
        if as_a_percentage is True:
            duration_curve = np.cumsum(duration_curve) / len(duration_curve)
            xlabel = "Duration (%)"
        else:
            xlabel = "Duration (nber of time-slots - hours)"
        name_label = self.name.capitalize()
        fig_file = os.path.join(output_dir, f"{name_label}_duration_curve.png")
        simple_plot(x=duration_curve, y=vals_desc_order, fig_file=fig_file,
                    title=f"{name_label} duration curve", xlabel=xlabel, 
                    ylabel=name_label)
        

def list_of_uc_timeseries_to_df(uc_timeseries: List[UCTimeseries]) -> pd.DataFrame:        
    uc_ts_dict = {uc_ts.name: uc_ts.values for uc_ts in uc_timeseries}
    # add dates, if available
    if uc_timeseries[0].dates is not None:
        uc_ts_dict["date"] = uc_timeseries[0].dates
    return pd.DataFrame(uc_ts_dict)
