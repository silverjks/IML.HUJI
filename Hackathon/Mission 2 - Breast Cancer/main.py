from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import re
from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier


def load_data(path):
    """
    :param path: path of data
    :return: Dataframe from data
    """
    df = pd.read_csv(path)
    df.dropna()
    return df


def preprocess_ki67(data: pd.DataFrame):
    data["אבחנה-KI67 protein"] = data["אבחנה-KI67 protein"].fillna("Null")

    # replace roman numerals
    for roman, num in [("IV", "4"), ("III", "3"), ("II", "2")]:
        data["אבחנה-KI67 protein"] = data["אבחנה-KI67 protein"].str.replace(roman, num)
    data["אבחנה-KI67 protein"] = data["אבחנה-KI67 protein"].str.lower()
    data["אבחנה-KI67 protein"] = data["אבחנה-KI67 protein"].str.replace("score i", "score 1")

    # replace entries without numbers with unknown
    def no_nums(string: str) -> bool:
        return not bool(re.search(r'\d', string))

    no_nums = np.vectorize(no_nums)
    no_nums_idx = data.index[no_nums(data["אבחנה-KI67 protein"])]
    months = "jan|fab|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    month_idx = data.index[data["אבחנה-KI67 protein"].str.contains(months, regex=True)]
    data.loc[no_nums_idx, "אבחנה-KI67 protein"] = "unknown"
    data.loc[month_idx, "אבחנה-KI67 protein"] = "unknown"

    # replace scores
    for score, percent in [(1, "5%"), (2, "9%"), (3, "30%"), (4, "50%")]:
        # contains % but doesn't contain score
        idx = data.index[
            (~data["אבחנה-KI67 protein"].str.contains("%")) & (data["אבחנה-KI67 protein"].str.contains(str(score)))]
        data.loc[idx, "אבחנה-KI67 protein"] = percent

    # score 1: 0-5%
    # score 2: 6-9%
    # score 3: 20-30%
    # score 4: > 30%

    # extract numbers
    def extract_nums(string: str):
        nums = np.array(re.findall(r"\d+", string)).astype(int)
        if len(nums) > 0:
            nums = nums.astype(int)
            return np.mean(nums)

        return -1

    extract_nums = np.vectorize(extract_nums)
    data["אבחנה-KI67 protein"] = extract_nums(data["אבחנה-KI67 protein"])

    bounds_idx = data.index[(data["אבחנה-KI67 protein"] < 0) | (data["אבחנה-KI67 protein"] > 100)]
    data.loc[bounds_idx, "אבחנה-KI67 protein"] = np.NaN

    return data


def preprocess_nodes_exam_and_positive(data: pd.DataFrame):
    good_idx = data.index[(~data["אבחנה-Nodes exam"].isna()) & (~data["אבחנה-Positive nodes"].isna())]
    bad_idx = data.index[(data["אבחנה-Nodes exam"].isna()) | (data["אבחנה-Positive nodes"].isna())]
    new_col = pd.Series(data=np.zeros(len(data)), name="Positive Nodes Ratio")
    new_col[bad_idx] = np.NaN
    new_col[good_idx] = data["אבחנה-Positive nodes"][good_idx] / data["אבחנה-Nodes exam"][good_idx]
    data["Positive Nodes Ratio"] = new_col
    data.drop(["אבחנה-Nodes exam", "אבחנה-Positive nodes"], axis=1, inplace=True)
    return data


def HER2_test_preproc(df, col_name):
    """
    Extracts data about the HER2 tests, makes 2 columns, of IHC and of FISH tests.
    Param: Data Frame
    Param: Column name (str)
    Return: Data Frame of 2 columns of IHC and FISH test results.
    """
    data = {'IHC': df[col_name], 'FISH': df[col_name]}
    df = pd.DataFrame(data)

    class_fish_col(df, 'FISH')
    class_ihc_col(df, 'IHC')

    return df


def class_ihc_col(df, col_name):
    """
    Classifies the IHC column, using the FISH classification or indicative values.
    Helper function for 'HER2_test_preproc' func.
    """

    ihc_pos_vals = ['pos', '+', 'jhuch', 'חיובי', '3']
    ihc_neutral_vals = ['2', '+2', '2+']
    ihc_neg_vals = ['neg', '-', 'akhkh', 'שלילי', '0', '1']

    for i, val in enumerate(df[col_name]):

        if df[col_name][i] == 1:  # Classification using the FISH column
            df[col_name][i] = 1
        elif df[col_name][i] == 0:
            df[col_name][i] = 0

        else:
            if type(val) is not str:  # If there is no FISH value, and only a number is presented
                if val <= 1:
                    df[col_name][i] = 0
                elif val > 2:
                    df[col_name][i] = 1
                elif val == 2:
                    df[col_name][i] = 0.5
                else:
                    df[col_name][i] = np.NaN

            elif type(val) is str:
                val = val.lower()  # Classification using indicative strings
                for neut_str in ihc_neutral_vals:
                    if neut_str in val:
                        df[col_name][i] = 0.5
                        break
                else:
                    for pos_str in ihc_pos_vals:  # Checks if there is an indicative value for positive test
                        if pos_str in val:
                            df[col_name][i] = 1
                            break
                    else:
                        for neg_str in ihc_neg_vals:  # Checks if there is an indicative value for positive test
                            if neg_str in val:
                                df[col_name][i] = 0
                                break
                        else:
                            df[col_name][i] = np.NaN


def class_fish_col(df, col_name):
    """
    Classifies the of HER2 FISH test using indicative strings.
    Helper function for 'HER2_test_preproc' func.
    """
    fish_pos_vals = ['pos', 'fish +', 'fish amplified', 'fish(+)', 'fish-pos',
                     'חיובי', 'fish+', 'psitive']
    fish_neg_vals = ['neg', 'fish -', 'fish(-)', 'fish (-)', 'fish-', '(-) by fish', 'שלילי', 'negative- fish',
                     'fish non amplified', 'nonamplified', 'fish no amplified', 'fish - amplified', 'non amplified',
                     'not amplified', ]

    for i, val in enumerate(df[col_name]):

        if type(val) is not str:  # In case the string 'FISH' does not appear
            df[col_name][i] = np.NaN
        elif "ish" not in val.lower():
            df[col_name][i] = np.NaN

        elif type(val) == str:
            val = val.lower()
            for pos_str in fish_pos_vals:  # Checks if there is an indicative value for positive test
                if pos_str in val:
                    df[col_name][i] = 1
                    break
            else:
                for neg_str in fish_neg_vals:  # Checks if there is an indicative value for negative test
                    if neg_str in val:
                        df[col_name][i] = 0
                        break
                else:
                    df[col_name][i] = np.NaN


def sort_percent_first(vals_arr):
    """
    Helper function for 'er_pr_col_prepos'
    """

    for i, val in enumerate(vals_arr):
        if len(vals_arr) > 1:
            if '%' in val:
                temp = vals_arr[0]
                vals_arr[0] = vals_arr[i]
                vals_arr[i] = temp
                if '2-3' in val:
                    vals_arr[0] = '2%'
                return vals_arr

            if '3' in val:
                vals_arr[i] = '3'
            if '2' in val:
                vals_arr[i] = '2'
            if '1' in val:
                vals_arr[i] = '1'

    return vals_arr


def er_pr_col_prepos(df, col_name):
    """
    Pre-Precessing of the er/pr tests column. Gives a grade of 0-3 according to the intensity of the cancer
    Param: df - Data Frame
    Param: col_name - The name of the wanted column (str)
    """
    if 'er' in col_name:
        name = 'ER'
    else:
        name = 'PR'

    data = {name: df[col_name]}
    new_df = pd.DataFrame(data)

    pos_vals = ['strong', 'weak', 'int', 'po', '+']
    neg_vals = ['neg', '-']
    for i, val in enumerate(new_df[name]):

        if val == np.NaN:
            continue

        if type(val) is str:
            vals_arr = val.split(sep=' ')
            vals_arr = sort_percent_first(vals_arr)
            for char in vals_arr:
                if '%' in char:  # If a percentage of cells appears
                    try:
                        num = int(re.sub(r'[^\d]', '', char))
                    except:
                        continue
                    if num > 33:
                        new_df[name][i] = 3
                    elif 11 < num <= 33:
                        new_df[name][i] = 2
                    elif 1 < num <= 11:
                        new_df[name][i] = 1
                    else:
                        new_df[name][i] = 0
                    break
                elif char.isdigit():  # If only a digit appears
                    num = int(char)
                    if num < 1:
                        new_df[name][i] = 0
                    elif num == 1:
                        new_df[name][i] = 1
                    elif num == 2:
                        new_df[name][i] = 2
                    elif num >= 3:
                        new_df[name][i] = 3
                    break
                else:
                    char = char.lower()  # Classifying according to indicative negative strings
                    for neg_str in neg_vals:
                        if neg_str in char:
                            new_df[name][i] = 0
                            break
                    else:
                        for pos_str in pos_vals:  # Classifying according to indicative positive strings
                            if pos_str in char:
                                if 'strong' in pos_str:
                                    new_df[name][i] = 3
                                elif 'int' in pos_str:
                                    new_df[name][i] = 2
                                else:
                                    new_df[name][i] = 1
                                break

                        if type(new_df[name][i]) is int:
                            break
                        else:
                            new_df[name][i] = np.NaN
    return new_df


def preprocess_dummies(df: DataFrame) -> DataFrame:
    """

    Parameters
    ----------
    df

    Returns
    -------

    """
    form_name_dummies = pd.get_dummies(df[" Form Name"])
    hospital_dummies = pd.get_dummies(df[" Hospital"])
    doctor_dummies = pd.get_dummies(df["User Name"])
    histological_dummies = pd.get_dummies(df["אבחנה-Histological diagnosis"])
    margin_type_dummies = pd.get_dummies(df["אבחנה-Margin Type"])
    side_dummies = pd.get_dummies(df["אבחנה-Side"])  # todo perform imputation to fill for missing values
    surgery_name_dummies1 = pd.get_dummies(df["אבחנה-Surgery name1"])  # 23 + null, todo perform imputation
    surgery_name_dummies2 = pd.get_dummies(df["אבחנה-Surgery name2"])  # 17 + null, todo perform imputation
    surgery_name_dummies3 = pd.get_dummies(df["אבחנה-Surgery name3"])  # 4 + null, todo perform imputation
    before_or_after_dummies = pd.get_dummies(df["surgery before or after-Actual activity"])
    # 10 + null, todo perform imputation

    # add dummies and remove redundant features
    df.drop([" Form Name", " Hospital", "User Name", "אבחנה-Histological diagnosis", "אבחנה-Margin Type",
             "אבחנה-Side", "אבחנה-Surgery name1", "אבחנה-Surgery name2", "אבחנה-Surgery name3",
             "surgery before or after-Actual activity", "id-hushed_internalpatientid", "אבחנה-Diagnosis date",
             "אבחנה-Surgery date1", "אבחנה-Surgery date2", "אבחנה-Surgery date3",
             "surgery before or after-Activity date"], inplace=True, axis=1)
    df = pd.concat([df, form_name_dummies, hospital_dummies, doctor_dummies, histological_dummies,
                    margin_type_dummies, side_dummies, surgery_name_dummies1, surgery_name_dummies2,
                    surgery_name_dummies3, before_or_after_dummies], axis=1)
    return df


def get_replacement_dict(ordered_options_list: List) -> Dict:
    """

    Parameters
    ----------
    ordered_options_list

    Returns
    -------

    """

    num_options = len(ordered_options_list)
    values = np.arange(0, num_options + 1)

    transform_dict = {'Not yet Established': np.NaN, '#NAME?': np.NaN}
    for i in range(num_options):
        transform_dict[ordered_options_list[i]] = values[i]

    return transform_dict


def preprocess(path: str):
    """
    Parameters
    ----------
    path
    Returns
    -------
    """
    # get dummies for unquantifiable features
    df = load_data(path)
    df = preprocess_dummies(df)
    df = preprocess_ki67(df)  # handle ki67
    df = preprocess_nodes_exam_and_positive(df)
    er_col = er_pr_col_prepos(df, "אבחנה-er")
    pr_col = er_pr_col_prepos(df, "אבחנה-pr")
    fish_cols = HER2_test_preproc(df, "אבחנה-Her2")
    df = pd.concat([df, fish_cols, pr_col, er_col], axis=1)
    df.drop(["אבחנה-Her2", "אבחנה-er", "אבחנה-pr"], inplace=True, axis=1)

    # replace values of Basic stage to something quantifiable
    ordered_options_list = ['c - Clinical', 'p - Pathological', 'r - Reccurent']
    df["אבחנה-Basic stage"] = df["אבחנה-Basic stage"].replace(get_replacement_dict(ordered_options_list))
    # todo perform imputation

    # replace values of Histopatological degree to something quantifiable
    # todo should GX be 1 or joker?
    ordered_options_list = ['G1 - Well Differentiated', 'G2 - Modereately well differentiated',
                            'G3 - Poorly differentiated', 'G4 - Undifferentiated', 'GX - Grade cannot be assessed']
    df["אבחנה-Histopatological degree"] = df["אבחנה-Histopatological degree"].replace(
        get_replacement_dict(ordered_options_list))
    # todo perform imputation

    # replace values of Lymphovascular invasion to something quantifiable
    ordered_options_list = {'(+)': 1, '(-)': 0, '+': 1, '-': 0, 'MICROPAPILLARY VARIANT': 1, 'NO': 0, 'No': 0,
                            'extensive': 1, 'neg': 0, 'no': 0, 'none': 0, 'not': 0, 'pos': 1, 'yes': 1}
    df["אבחנה-Ivi -Lymphovascular invasion"] = df["אבחנה-Ivi -Lymphovascular invasion"].replace(ordered_options_list)
    # todo perform imputation

    # replace values of -Lymphatic penetration to something quantifiable
    ordered_options_list = ['L0 - No Evidence of invasion', 'L1 - Evidence of invasion of superficial Lym.',
                            'LI - Evidence of invasion', 'L2 - Evidence of invasion of depp Lym.']
    df["אבחנה-Lymphatic penetration"] = df["אבחנה-Lymphatic penetration"].replace(get_replacement_dict(
        ordered_options_list))
    # todo imputation

    # replace values of metastases mark to something quantifiable
    # todo place MX
    ordered_options_list = ['M0', 'M1', 'M1a', 'M1b', 'MX']
    df["אבחנה-M -metastases mark (TNM)"] = df["אבחנה-M -metastases mark (TNM)"].replace(
        get_replacement_dict(ordered_options_list))

    # replace values of lymph nodes mark  to something quantifiable
    # todo place NX
    ordered_options_list = ['ITC', 'N0', 'N1', 'N1mic', 'N1a', 'N1b', 'N1c', 'N1d', 'N1mic', 'N2', 'N2a', 'N2b', 'N2c',
                            'N3', 'N3a',
                            'N3c', 'N4', 'NX']
    df["אבחנה-N -lymph nodes mark (TNM)"] = df["אבחנה-N -lymph nodes mark (TNM)"].replace(
        get_replacement_dict(ordered_options_list))

    # # replace values of cancer stages to something quantifiable
    # todo deal with LA
    ordered_options_list = ['Stage0', 'Stage0a', 'Stage0is', 'Stage1', 'Stage1a', 'Stage1b', 'Stage1c',
                            'Stage2', 'Stage2a', 'Stage2b', 'Stage3', 'Stage3a', 'Stage3b', 'Stage3c',
                            'Stage4', 'LA']

    df["אבחנה-Stage"] = df["אבחנה-Stage"].replace(get_replacement_dict(ordered_options_list))
    # todo imputation

    # # replace values of tumor mark stages to something quantifiable
    ordered_options_list = ['T0', 'Tis', 'T1', 'T1a', 'T1b', 'T1c', 'T1mic', 'T2', 'T2a', 'T2b', 'T3', 'T3b', 'T3c',
                            'T3d', 'T4', 'T4a', 'T4b', 'T4c', 'T4d', 'MF', 'Tx']
    # todo handle mf, tx

    df["אבחנה-T -Tumor mark (TNM)"] = df["אבחנה-T -Tumor mark (TNM)"].replace(get_replacement_dict(
        ordered_options_list))

    # merge אבחנה-Positive nodes and אבחנה-Nodes exam todo: after imputation

    for col in df.columns:
        df[col] = df[col].replace(['Null', np.inf, -np.inf], np.NaN)

    # for quantifiable features, set valid range
    df["אבחנה-Surgery sum"] = pd.to_numeric(df["אבחנה-Surgery sum"], downcast="float")
    df["אבחנה-Tumor depth"] = pd.to_numeric(df["אבחנה-Tumor depth"], downcast="float")
    df["אבחנה-Tumor width"] = pd.to_numeric(df["אבחנה-Tumor width"], downcast="float")
    df["אבחנה-Age"] = pd.to_numeric(df["אבחנה-Age"], downcast="float")

    df.to_csv("testing", sep='\t')
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(df.values)
    df = imputer.transform(df.values)

    # df = df.loc[(df["אבחנה-Surgery sum"] >= 0) & (df["אבחנה-Surgery sum"] < 2)]
    # df = df.loc[(df["אבחנה-Tumor depth"] >= 0) & (df["אבחנה-Tumor depth"] < 10)]
    # df = df.loc[(df["אבחנה-Tumor width"] >= 0) & (df["אבחנה-Tumor width"] < 10)]
    # df = df.loc[(df["אבחנה-Age"] > 14) & (df["אבחנה-Age"] < 120)]

    return df


def extract_labels_of_sample(val):
    set_of_labels = set()
    x = val.split(",")
    for label in x:
        if val == "[]":
            set_of_labels.add("None")
            continue

        label = label.replace("[", "")
        label = label.replace("]", "")
        label = label.replace(" ", "")
        label = label.replace("'", "")
        set_of_labels.add(label)
    return set_of_labels


def extract_all_labels(y):
    unique_vals = pd.unique(y["אבחנה-Location of distal metastases"])
    all_labels = set()
    for val in unique_vals:
        all_labels |= extract_labels_of_sample(val)

    return all_labels


def process_labels(y):
    unique_labels = extract_all_labels(y)

    vector_labels = pd.DataFrame(0, columns=unique_labels, index=range(len(y)))
    for i, val in enumerate(y["אבחנה-Location of distal metastases"]):
        labels = extract_labels_of_sample(val)
        for label in labels:
            vector_labels.loc[i, label] = 1

    return vector_labels


if __name__ == '__main__':
    path = "train.feats.csv"

    y_path = "train.labels.0.csv"
    df = preprocess(path)
    y = process_labels(load_data(y_path))
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.25)
    clf = MultiOutputClassifier(svm.SVC()).fit(x_train, y_train)

    clf.score(x_test, y_test)
    print()
