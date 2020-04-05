#!/usr/bin/env python3

import src
from src.read.paths import safe_makedir
import pandas as pd
import numpy as np
import pygsheets
import os
from datetime import date
from pathlib2 import Path
from texttable import Texttable


PROJECT_PATH = Path("__file__").resolve().parents[0]  # ! Ego = notebook
DATA_PATH = PROJECT_PATH / "data"
FIELD_PATH = DATA_PATH / "fieldwork" / str(date.today().year)


##########################################################################
# Download google spreadsheets with nestbox info and create lists and maps
##########################################################################


# Auth key (keep private)
gc = pygsheets.authorize(service_file=str(
    PROJECT_PATH / "private" / "client_secret.json"))


# Download Summary sheet from Google Drive and convert to pandas dataframe
#! This needs to be changed every year
summary = gc.open_by_key('1DWcjGS8j2axWgbNxA7RnYqN_0oEb80J_O1bSALVMzxE')[
    0].get_as_df(has_header=True)


# Add coordinates of nestboxes
postmp = FIELD_PATH / "nestbox_position.xls"

nestbox_coords = (pd.read_excel(postmp)
                  .filter(['x', 'y', 'Nestbox']))

nestbox_coords['Nestbox'] = nestbox_coords['Nestbox'].str.upper()

merged = pd.merge(summary, nestbox_coords, on=['Nestbox'])


# Save dataframe of new nestboxes
# with 'date added' column and filename = 'date saved'
filename = FIELD_PATH / (str(
    str('summary')
    + '_'
    + str(pd.Timestamp('today', tz="UTC").strftime('%Y%m%d_%H%M'))
    + '.pkl'))

which_greati = (merged
                .filter(['Species', 'Nestbox', 'Owner', 'x', 'y'])
                .query('Species == "test" or Species == "G"')
                )

# Define undefined (:D) error type and stop execution if there are
# no great tit nestboxes. Otherwise save a pickle file.


class Error(Exception):
    pass


if len(which_greati) == 0:
    raise Error("There are no GRETI nestboxes")
else:
    which_greati['Added'] = (str(pd.Timestamp('today', tz="UTC")
                                 .strftime('%Y-%m-%d_%H:%M')))
    which_greati.to_pickle(str(filename))


# Read spreadsheets and compare last two (if there are two)

spreadsheets = list(FIELD_PATH.glob('*.pkl'))


def dataframe_diff(df1, df2, colnames):
    """Pluck rows in df2 that are not in df1, based on colnames.

    Args:
        df1 (Pandas DataFrame): first dataframe
        df2 (Pandas DataFrame): second dataframe with new rows
        colnames (str): columns to look at e.g.:['col1', 'col2']

    Returns:
        [Pandas DataFrame]: New dataframe cointaining 
        rows thar are in df2 but not in df1
    """
    diff_df = (df1.merge(df2,
                         on=colnames,
                         indicator=True,
                         how='outer')
               .query('_merge != "both"')
               .query('_merge != "left_only"')
               .drop(['Owner_x', 'x_x',
                      'y_x', 'Added_x', '_merge'], 1)
               )
    diff_df = diff_df.rename(
        columns={col: col.split('_')[0] for col in diff_df.columns})
    return diff_df


if len(spreadsheets) > 1:  # there are more than 1 files
    dat = [pd.read_pickle(s) for s in spreadsheets[-2:]]
    new = dataframe_diff(dat[0],
                         dat[1],
                         colnames=['Species', 'Nestbox'])
    if len(new) >= 10:  # and they have at least 10 rows

        new.to_csv(FIELD_PATH / str(
            'NEW-'
            + str(pd.Timestamp('today', tz="UTC").strftime('%Y%m%d_%H%M'))
            + '.csv'
        ),
            index=False
        )
    else:  # if not, remove .pkl
        os.remove(spreadsheets[-1])

elif len(spreadsheets) == 1:
    new = pd.read_pickle(str(spreadsheets)[12:-3])

    if len(new) >= 10:

        new.to_csv(FIELD_PATH / str(
            'NEW-'
            + str(pd.Timestamp('today', tz="UTC").strftime('%Y%m%d_%H%M'))
            + '.csv'
        ),
            index=False
        )
    else:
        os.remove(spreadsheets[-1])


else:
    print("No .pkl files in this directory")
