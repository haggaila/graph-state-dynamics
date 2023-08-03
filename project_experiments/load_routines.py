# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import csv
import os
from datetime import datetime, timedelta
import time
from typing import List, Union, Optional, Dict, Any

from qiskit import QiskitError
import pandas as pd
from qiskit_experiments.framework import ExperimentData, AnalysisResultData
from uncertainties import UFloat


def query_db(
    s_csv_file: str,
    s_filter_query: str,
    sort_by: Optional[Any] = None,
    ascending: Union[bool, List[bool]] = True,
    na_position="last",
    parse_dates=None,
):
    """Find the simulations according to the criteria query and metadata dictionaries in a list.

    Args:
            s_csv_file: The db (.csv) file.
            s_filter_query: A string with the desired query.
            sort_by: If not None, defines sorting using the method sort_values() of the data frame.
            ascending: If sort_by is not None, defines an option for the sort
            na_position: If sort_by is not None, defines an option for the sort
            parse_dates: Date fields.

    Returns:
            A list with the relevant simulation dicts.
    """

    df = pd.read_csv(s_csv_file, parse_dates=parse_dates)
    df_2 = df.query(s_filter_query)
    if sort_by is not None:
        df_3 = df_2.sort_values(sort_by, ascending=ascending, na_position=na_position)
    else:
        df_3 = df_2
    return df_3


def unpack_analysis_result(
    result: Union[AnalysisResultData, Dict],
    backend_name: str,
    completion_time: datetime,
) -> (Union[dict, None], str):
    """Turn an AnalysisResult into a dictionary for the monitoring dataframe.

    Keys below must match the monitoring dataframe keys.
    Args:
        result: The database result entry.
        backend_name: The name of the backend.
        completion_time: The job's completion time.

    Returns:
        A dictionary that describes one data line to be added to the monitoring variables
            dataframe. If None is returned, this data line should be skipped.
    """
    val = result.value
    name = result.name
    experiment_id = result.experiment_id
    device_components = result.device_components

    data_dict = {}
    if isinstance(val, UFloat):
        unit = getattr(val, "tag", None)
        if unit is None:
            extra = getattr(result, "extra", None)
            if unit in extra:
                unit = extra["unit"]

        chi_sq = result.chisq
        verified = result.verified
        data_dict.update(
            {
                "value": val.nominal_value,
                "std_dev": val.std_dev,
                "unit": unit,
                "chi_sq": chi_sq,
                "verified": verified,
            }
        )
    elif isinstance(val, dict):
        data_dict.update(
            {"value": val["value"], "std_dev": val["std_dev"], "unit": val["tag"]}
        )
    elif isinstance(val, float):
        data_dict.update({"value": val})
    else:
        # We do not currently handle other types (such as lists), and returning None
        # indicates that this data line should be skipped.
        return None, ""

    data_dict.update(
        {
            "variable_name": name,
            "backend": backend_name,
            "end_date": completion_time,
            "device_components": ",".join(map(str, device_components)),
            "experiment_id": experiment_id,
        }
    )
    return data_dict, name
