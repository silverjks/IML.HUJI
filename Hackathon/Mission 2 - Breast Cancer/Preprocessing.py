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
    df = pd.read_csv(path, parse_dates=['אבחנה-Diagnosis date'])
    return df


def bad_rows(path):
    """
    print the 10 smallest/biggest values of any quantifiable features to search for outliers
    Parameters
    ----------
    path

    Returns
    -------

    """
    data = load_data(path)
    for col in data.select_dtypes(include='number').columns:
        print(f'Smallest Values at {col}')
        x = data.nsmallest(5, [col])
        print(x[col], "\n")


def preprocess(path: str) -> DataFrame:
    """

    Parameters
    ----------
    path

    Returns
    -------

    """
    # get dummies for unquantifiable features
    df = load_data(path)
    form_name_dummies = pd.get_dummies(df[" Form Name"])  # 9 values
    hospital_dummies = pd.get_dummies(df[" Hospital"])  # 4 values
    doctor_dummies = pd.get_dummies(df["User Name"])  # 148 values
    histological_dummies = pd.get_dummies(df["אבחנה-Histological diagnosis"])
    margin_type_dummies = pd.get_dummies(df["אבחנה-Margin Type"])
    # todo find out is values can be merged- is none = no cancer?
    # add dummies and remove redundant features
    df.drop([" Form Name", " Hospital", "User Name", "אבחנה-Histological diagnosis", "אבחנה-Margin Type"], inplace=True, axis=1)
    df = pd.concat([df, form_name_dummies, hospital_dummies, doctor_dummies, histological_dummies,
                    margin_type_dummies], axis=1)


    ### TODO ADI'S TEST
    dummy = pd.get_dummies(df["אבחנה-Side"])  # 4 values
    c = df["אבחנה-N -lymph nodes mark (TNM)"]
    c = c.dropna()
    ### TODO ADI'S TEST

    # replace values of stages to something quantifiable
    stage_list = ['c - Clinical', 'p - Pathological', 'r - Reccurent']

    num_stages = 3
    stage_nums = np.arange(0, num_stages + 1)

    stage_dict = {}
    for i in range(num_stages):
        stage_dict[stage_list[i]] = stage_nums[i]

    df["אבחנה-Basic stage"] = df["אבחנה-Basic stage"].replace(stage_dict)

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

    # replace values of stages to something quantifiable
    lymph_nodes_mark_list = ['N0', 'N1', 'N1a', 'N1b', 'N1c', 'N1d', 'N2', 'N2a', 'N2b', 'N2c', 'N3', 'N3a',
                             'N3c', 'N3c', 'N4', 'NX']

    num_lymph_nodes_mark = 18
    lymph_nodes_mark_num = np.arange(0, num_lymph_nodes_mark + 1)

    lymph_node_mark_dict = {'Not yet Established': 'Null', '#NAME?': 'Null'}
    # todo: seperately handle ITC, N1mic, NX
    for i in range(num_stages):
        lymph_node_mark_dict[lymph_nodes_mark_list[i]] = lymph_nodes_mark_num[i]

    df["אבחנה-N -lymph nodes mark (TNM)"] = df["אבחנה-N -lymph nodes mark (TNM)"].replace(lymph_node_mark_dict)

    # merge אבחנה-Positive nodes and אבחנה-Nodes exam todo: after imputation
    # for quantifiable features, set valid range
    df = df.loc[(df["אבחנה-Age"] > 14) & (df["אבחנה-Age"] < 120)]



if __name__ == '__main__':
    """
    get argument from command line (the csv file) and return the preprocessed data?
    """
    path = "train.feats.csv"
    preprocess(path)
