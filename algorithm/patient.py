
# from matplotlib import pyplot as plt
import pandas as pd

from algorithm.lesion import CT
from algorithm.feature import add_prefix_to_dict, get_features_diff
from algorithm.config import *

class Patient:
    def __init__(self, pid: int):
        self.pid = pid
        self.ct_id_1 = str(pid) + '-1'
        self.ct_id_2 = str(pid) + '-2'

        # ct objects
        self.ct_1 = None
        self.ct_2 = None

        # feature related
        self.patient_feature_dict = {}

    def load_data(self):
        self.ct_1 = CT(self.ct_id_1)
        self.ct_2 = CT(self.ct_id_2)

    def extract_features(self):
        # before/after treatment features and their diff
        features_before = self.ct_1.extract_features()
        features_after = self.ct_2.extract_features()

        # sanity check of lesion label_ids are the same
        #check_lesion_count(self)     ### disable lesion count check

        # sanity check of label-layer relationship
        #check_id_layers_relationship(self)     ### disable mask layer relationship check

        features_diff = get_features_diff(features_before, features_after)

        features_before = add_prefix_to_dict(features_before, 'before')
        features_after = add_prefix_to_dict(features_after, 'after')
        features_diff = add_prefix_to_dict(features_diff, 'diff')

        # patient_feature_dict = {**features_before, **features_after, **features_diff}
        patient_feature_dict = {**features_after, **features_diff}
        self.patient_feature_dict = patient_feature_dict

        return patient_feature_dict


def check_lesion_count(patient: Patient):
    if patient.ct_1.lesion_id_list != patient.ct_2.lesion_id_list:
        error_data = {
            'pid': patient.pid,
            'ct1_lesion': str(patient.ct_1.lesion_id_list),
            'ct2_lesion': str(patient.ct_2.lesion_id_list)
        }
        raise ValueError({
            'message': ERROR_LESION_COUNT,
            'data': error_data
        })


def get_df_lesion_layers(ct: CT, ct_id: int):
    lesion_layers = []
    for lesion in ct.lesion_list:
        lesion_layers.append(lesion.lesion_layer)

    col_name = 'ct' + str(ct_id) + '_layer'
    df_lesion_layers = pd.DataFrame({'lesion_id': ct.lesion_id_list, col_name: lesion_layers})

    return df_lesion_layers


def check_id_layers_relationship(patient: Patient):
    df_lesion1_layers = get_df_lesion_layers(patient.ct_1, 1)
    df_lesion2_layers = get_df_lesion_layers(patient.ct_2, 2)

    lesion1_sorted_id = list(df_lesion1_layers.sort_values('ct1_layer')['lesion_id'])
    lesion2_sorted_id = list(df_lesion2_layers.sort_values('ct2_layer')['lesion_id'])

    if lesion1_sorted_id != lesion2_sorted_id:
        error_data = df_lesion1_layers.merge(df_lesion2_layers)
        error_data['pid'] = patient.pid
        error_data = error_data[['pid', 'lesion_id', 'ct1_layer', 'ct2_layer']].to_dict('records')
        raise ValueError({
            'message': ERROR_LESION_RELATIONSHIP,
            'data': error_data
        })
