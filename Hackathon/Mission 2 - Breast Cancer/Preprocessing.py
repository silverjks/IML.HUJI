from typing import List

import pandas as pd
from numpy import asarray
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from pandas import DataFrame


def load_data(path):
    """
    :param path: path of data
    :return: Dataframe from data
    """
    df = pd.read_csv(path, parse_dates=['אבחנה-Diagnosis date', 'אבחנה-Surgery date1', 'אבחנה-Surgery date2',
                                        'אבחנה-Surgery date3'], dayfirst=True, infer_datetime_format=True,
                     low_memory=False)
    df.fillna('Null', inplace=True)
    return df


def preprocess_dummies(df: DataFrame) -> DataFrame:
    """

    Parameters
    ----------
    df

    Returns
    -------

    """
    """
    STEP 1: simple dummies
    """
    form_name_dummies = pd.get_dummies(df[" Form Name"])
    hospital_dummies = pd.get_dummies(df[" Hospital"])
    doctor_dummies = pd.get_dummies(df["User Name"])
    histological_dummies = pd.get_dummies(df["אבחנה-Histological diagnosis"])
    margin_type_dummies = pd.get_dummies(df["אבחנה-Margin Type"])
    side_dummies = pd.get_dummies(df["אבחנה-Side"])  # todo perform imputation to fill for missing values

    """
    STEP 2: complex dummies
    """
    # for the surgery name, we would like to fill the instances where no surgery was performed with 'None'
    surgery_name_dummies1 = pd.get_dummies(df["אבחנה-Surgery name1"])  # 23 + null, todo perform imputation
    surgery_name_dummies2 = pd.get_dummies(df["אבחנה-Surgery name2"])  # 17 + null, todo perform imputation
    surgery_name_dummies3 = pd.get_dummies(df["אבחנה-Surgery name3"])  # 4 + null, todo perform imputation

    # todo add michael's code

    before_or_after_dummies = pd.get_dummies(df["surgery before or after-Actual activity"])
    # 10 + null, todo perform imputation

    id_dummies = pd.get_dummies(df["id-hushed_internalpatientid"])

    # add dummies and remove redundant features
    df.drop([" Form Name", " Hospital", "User Name", "אבחנה-Histological diagnosis", "אבחנה-Margin Type",
             "אבחנה-Side", "אבחנה-Surgery name1", "אבחנה-Surgery name2", "אבחנה-Surgery name3",
             "surgery before or after-Actual activity", "id-hushed_internalpatientid"], inplace=True, axis=1)
    df = pd.concat([df, form_name_dummies, hospital_dummies, doctor_dummies, histological_dummies,
                    margin_type_dummies, side_dummies, surgery_name_dummies1, surgery_name_dummies2,
                    surgery_name_dummies3, before_or_after_dummies, id_dummies], axis=1)

    return df


def preprocess_quantifiable(column: DataFrame, ordered_options_list: List) -> DataFrame:
    """

    Parameters
    ----------
    column
    ordered_options_list
    column_name

    Returns
    -------

    """

    num_options = len(ordered_options_list)
    new_values = np.arange(0, num_options + 1)

    transform_dict = {'Not yet Established': 'Null', '#NAME?': 'Null'}
    for i in range(num_options):
        transform_dict[ordered_options_list[i]] = new_values[i]

    return column.replace(transform_dict)


def preprocess(filename: str) -> DataFrame:
    """

    Parameters
    ----------
    filename

    Returns
    -------

    """

    df = load_data(filename)

    """
    STEP 1: Get dummies for non-quantifiable features that cannot be ordered
    """
    df = preprocess_dummies(df)

    # replace values of stages to something quantifiable
    ordered_options_list = ['c - Clinical', 'p - Pathological', 'r - Reccurent']
    column_name = "אבחנה-Basic stage"
    df[column_name] = preprocess_quantifiable(df[column_name], ordered_options_list)

    # replace values of degrees to something quantifiable
    # todo deal with null!!!! also, should GX be 1 or joker?
    grade_list = ['G1 - Well Differentiated', 'G2 - Modereately well differentiated',
                  'G3 - Poorly differentiated', 'G4 - Undifferentiated', 'GX - Grade cannot be assessed']

    num_grades = 5
    grade_nums = np.arange(0, num_grades + 1)

    grade_dict = {}
    for i in range(num_grades):
        grade_dict[grade_list[i]] = grade_nums[i]

    df["אבחנה-Histopatological degree"] = df["אבחנה-Histopatological degree"].replace(grade_dict)

    # replace values of Lymphovascular invasion to something quantifiable
    lymph_dict = {'(+)': 1, '(-)': 0, '+': 1, '-': 0, 'MICROPAPILLARY VARIANT': 1, 'NO': 0, 'No': 0,
                  'extensive': 1, 'neg': 0, 'no': 0, 'none': 0, 'not': 0, 'pos': 1, 'yes': 1}

    df["אבחנה-Ivi -Lymphovascular invasion"] = df["אבחנה-Ivi -Lymphovascular invasion"].replace(lymph_dict)

    # replace values of metastases mark to something quantifiable
    mark_list = ['M0', 'M1', 'M1a', 'M1b', 'MX']
    # todo imputation
    num_marks = 5
    marks_nums = np.arange(0, num_marks + 1)

    mark_dict = {'Not yet Established': 'Null'}
    for i in range(num_marks):
        mark_dict[mark_list[i]] = marks_nums[i]

    df["אבחנה-M -metastases mark (TNM)"] = df["אבחנה-M -metastases mark (TNM)"].replace(mark_dict)

    # replace values of lymph nodes mark to something quantifiable
    lymph_mark_list = ['N0', 'N1', 'N1a', 'N1b', 'N1c', 'N1d', 'N2', 'N2a', 'N2b', 'N2c', 'N3', 'N3a',
                       'N3c', 'N3c', 'N4', 'NX']

    num_lymph_nodes_mark = 18
    lymph_nodes_mark_num = np.arange(0, num_lymph_nodes_mark + 1)

    lymph_node_mark_dict = {'Not yet Established': 'Null', '#NAME?': 'Null'}
    # todo: seperately handle ITC, N1mic, NX
    for i in range(num_stages):
        lymph_node_mark_dict[lymph_mark_list[i]] = lymph_nodes_mark_num[i]

    df["אבחנה-N -lymph nodes mark (TNM)"] = df["אבחנה-N -lymph nodes mark (TNM)"].replace(lymph_node_mark_dict)
    # replace values of cancer stages to something quantifiable
    stages_list = ['Stage0', 'Stage0a', 'Stage0is', 'Stage1', 'Stage1a', 'Stage1b', 'Stage1c',
                   'Stage2', 'Stage2a', 'Stage2b', 'Stage3', 'Stage3a', 'Stage3b', 'Stage3c',
                   'Stage4', ]

    num_stages = 15
    stages_num = np.arange(0, num_stages + 1)

    stages_dict = {'Not yet Established': 'Null'}
    # todo: handle la, imputation

    for i in range(num_stages):
        stages_dict[stages_list[i]] = stages_num[i]

    df["אבחנה-Stage"] = df["אבחנה-Stage"].replace(stage_dict)

    # replace values of tumor mark stages to something quantifiable
    mark_list = ['T0', 'T1', 'T1a', 'T1b', 'T1c', 'T1mic', 'T2', 'T2a', 'T2b', 'T3', 'T3b', 'T3c',
                 'T3d', 'T4', 'T4a', 'T4b', 'T4c', 'T4d']
    # todo handle mf,tis, tx
    num_marks = 18
    marks_nums = np.arange(0, num_lymph_nodes_mark + 1)

    mark_dict = {'Not yet Established': 'Null'}
    # todo: seperately handle ITC, N1mic, NX
    for i in range(num_marks):
        mark_dict[mark_list[i]] = marks_nums[i]

    df["אבחנה-T -Tumor mark (TNM)"] = df["אבחנהאבחנה-T -Tumor mark (TNM)"].replace(mark_dict)

    # merge אבחנה-Positive nodes and אבחנה-Nodes exam todo: after imputation
    # for quantifiable features, set valid range
    df = df.loc[(df["אבחנה-Age"] > 14) & (df["אבחנה-Age"] < 120)]

    ### TODO ADI'S TEST
    dummy = pd.get_dummies(df["surgery before or after-Actual activity"])  # 4 values
    c = df["אבחנה-N -lymph nodes mark (TNM)"]
    c = c.dropna()
    ### TODO ADI'S TEST
    return df


if __name__ == '__main__':
    """
    get argument from command line (the csv file) and return the preprocessed data?
    """
    path = "train.feats.csv"
    preprocess(path)
