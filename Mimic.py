import os
import pandas as pd
import dask.dataframe as dd
from datetime import datetime
import sys
from google.colab import drive

drive.mount('/content/drive')

sys.path.append('..')
from utils.util import generic_utils

class MIMIC:
    def __init__(self, dir_path) -> None:
        self._dir_path = os.path.abspath(dir_path)
        self._mimic_structure = [
            'admissions', 'patients', 'transfers', 'd_hcpcs', 'd_icd_diagnoses', 
            'd_icd_procedures', 'd_labitems', 'diagnoses_icd', 'drgcodes', 'emar', 
            'emar_detail', 'hcpcsevents', 'labevents', 'microbiologyevents', 'pharmacy', 
            'poe', 'poe_detail', 'prescriptions', 'procedures_icd', 'services', 'chartevents', 
            'd_items', 'datetimeevents', 'icustays', 'inputevents', 'outputevents', 
            'procedureevents'
        ]

        try:
            assert os.path.exists(self._dir_path)
            print(f"MIMIC dataset found at path : {self._dir_path}")
        except AssertionError:
            print(f"Dataset directory is empty. Exiting system.")
            sys.exit(0)

    def read_data(self, file_path):
        if os.path.exists(file_path):
            df = dd.read_parquet(file_path) 
            return df
        else:
            print(f"Given file path {file_path} doesn't exist.")
            
    def join_path(self, file_name):
        return os.path.join(self._dir_path, f'{file_name}.parquet')

    def read_admissions(self):
        return self.read_data(self.join_path('admissions'))

    def read_patients(self):
        return self.read_data(self.join_path('patients'))

    def read_transfers(self):
        return self.read_data(self.join_path('transfers'))

    def read_d_hcpcs(self):
        return self.read_data(self.join_path('d_hcpcs'))

    def read_d_icd_diagnoses(self):
        return self.read_data(self.join_path('d_icd_diagnoses'))

    def read_d_icd_procedures(self):
        return self.read_data(self.join_path('d_icd_procedures'))

    def read_d_labitems(self):
        return self.read_data(self.join_path('d_labitems'))

    def read_diagnoses_icd(self):
        return self.read_data(self.join_path('diagnoses_icd'))

    def read_drgcodes(self):
        return self.read_data(self.join_path('drgcodes'))

    def read_emar(self):
        return self.read_data(self.join_path('emar'))

    def read_emar_detail(self):
        return self.read_data(self.join_path('emar_detail'))

    def read_hcpcsevents(self):
        return self.read_data(self.join_path('hcpcsevents'))

    def read_labevents(self):
        return self.read_data(self.join_path('labevents'))

    def read_microbiologyevents(self):
        return self.read_data(self.join_path('microbiologyevents'))

    def read_pharmacy(self):
        return self.read_data(self.join_path('pharmacy'))

    def read_poe(self):
        return self.read_data(self.join_path('poe'))

    def read_poe_detail(self):
        return self.read_data(self.join_path('poe_detail'))

    def read_prescriptions(self):
        return self.read_data(self.join_path('prescriptions'))

    def read_procedures_icd(self):
        return self.read_data(self.join_path('procedures_icd'))

    def read_services(self):
        return self.read_data(self.join_path('services'))

    def read_chartevents(self):
        return self.read_data(self.join_path('chartevents'))

    def read_d_items(self):
        return self.read_data(self.join_path('d_items'))

    def read_datetimeevents(self):
        return self.read_data(self.join_path('datetimeevents'))

    def read_icustays(self):
        return self.read_data(self.join_path('icustays'))

    def read_inputevents(self):
        return self.read_data(self.join_path('inputevents'))

    def read_outputevents(self):
        return self.read_data(self.join_path('outputevents'))

    def read_procedureevents(self):
        return self.read_data(self.join_path('procedureevents'))

class MIMICManipulations:
    def __init__(self, dir_path):
        self._mimic_instance = MIMIC(dir_path)
        self._util_instance = generic_utils()

    def filter_admission(self, df):
        df_admission = df.drop(['hadm_id', 'edregtime', 'edouttime', 'deathtime'], axis=1)
        df_admission['admittime'] = dd.to_datetime(df_admission['admittime'])
        df_admission['dischtime'] = dd.to_datetime(df_admission['dischtime'])
        df_admission['los_admission'] = (df_admission['dischtime'] - df_admission['admittime']).dt.total_seconds()/86400
        df_admission = df_admission.drop(['admittime', 'dischtime'], axis=1)
        df_admission = df_admission[df_admission['los_admission'] > 0]
        df_admission = self._util_instance.remove_duplicates_and_re_index(df_admission, 'subject_id')
        return df_admission

    def filter_patients(self, df):
        df_patient = df.drop(['anchor_year', 'anchor_year_group', 'dod'], axis=1)
        df_patient = self._util_instance.remove_duplicates_and_re_index(df_patient, 'subject_id')
        return df_patient

    def filter_transfers(self, df):
        df_transfer = df.drop(['hadm_id', 'transfer_id', 'intime', 'outtime'], axis=1)
        df_transfer = self._util_instance.remove_duplicates_and_re_index(df_transfer, 'subject_id')
        return df_transfer

    def merge_core_tables(self):
        df_admission = self.filter_admission(self._mimic_instance.read_admissions())
        df_patient = self.filter_patients(self._mimic_instance.read_patients())
        df_transfers = self.filter_transfers(self._mimic_instance.read_transfers())
        df_core_merge = df_admission.merge(df_patient, how='left', on=['subject_id'])\
                                    .merge(df_transfers, how='outer', on=['subject_id', 'hadm_id'])
        return df_core_merge

    def filter_diagnoses_icd(self, df):
        df_diagnoses_icd = df.drop(['hadm_id', 'seq_num', 'icd_version'], axis=1)
        df_diagnoses_icd = df_diagnoses_icd.rename(columns = {'icd_code':'diagnosis_icd_code'})
        df_diagnoses_icd = self._util_instance.remove_duplicates_and_re_index(df_diagnoses_icd, 'subject_id')
        return df_diagnoses_icd

    def filter_procedures_icd(self, df):
        df_procedures_icd = df.drop(['hadm_id', 'seq_num', 'chartdate', 'icd_version'], axis=1)
        df_procedures_icd = df_procedures_icd.rename(columns = {'icd_code':'procedures_icd_code'})
        df_procedures_icd = self._util_instance.remove_duplicates_and_re_index(df_procedures_icd, 'subject_id')
        return df_procedures_icd

    def filter_lab_events(self, df):
        df_lab_events = df.drop(['labevent_id', 'specimen_id', 'itemid', 'charttime', 'storetime', 'valueuom', 
                                    'ref_range_lower', 'ref_range_upper', 'comments', 'valuenum', 'hadm_id'], axis=1)
        df_lab_events = self._util_instance.remove_duplicates_and_re_index(df_lab_events, 'subject_id')                            
        return df_lab_events

    def merge_lab_events_d_labitems(self):
        df_lab_events = self.filter_lab_events(self._mimic_instance.read_labevents())
        df_d_lab_items = self._mimic_instance.read_d_labitems()
        df_lab_events_merge = df_lab_events.merge(df_d_lab_items, how='left', on=['itemid'])
        return df_lab_events_merge

    def filter_drgcodes(self, df):
        df_drgcodes = df.drop(['hadm_id', 'description', 'drg_severity', 'drg_mortality'], axis=1)
        df_drgcodes = self._util_instance.remove_duplicates_and_re_index(df_drgcodes, 'subject_id')
        return df_drgcodes

    def filter_emar(self, df):
        df_emar = df[['subject_id', 'medication', 'event_txt']]
        df_emar = self._util_instance.remove_duplicates_and_re_index(df_emar, 'subject_id')
        return df_emar

    def filter_poe(self, df):
        df_poe = df[['subject_id', 'order_type', 'transaction_type']]
        df_poe = self._util_instance.remove_duplicates_and_re_index(df_poe, 'subject_id')
        return df_poe

    def merge_emar_poe(self):
        df_emar = self.filter_emar(self._mimic_instance.read_emar())
        df_poe = self.filter_poe(self._mimic_instance.read_poe())
        df_merge_emar_poe = df_emar.merge(df_poe, how='outer', on=['poe_id', 'subject_id', 'hadm_id'])
        df_merge_emar_poe.drop(['emar_id', 'poe_id'], axis=1)
        return df_merge_emar_poe

    def filter_microbiologyevents(self, df):
        df_microbiologyevents = df[['subject_id', 'org_name', 'test_name', 'quantity', 'ab_name']]
        df_microbiologyevents = self._util_instance.remove_duplicates_and_re_index(df_microbiologyevents, 'subject_id')
        return df_microbiologyevents

    def filter_prescriptions(self, df):
        df_prescriptions = df[['subject_id', 'drug', 'route']]
        df_prescriptions = self._util_instance.remove_duplicates_and_re_index(df_prescriptions, 'subject_id')
        return df_prescriptions

    def filter_service(self, df):
        df_service = df[['subject_id', 'curr_service']]
        df_service = self._util_instance.remove_duplicates_and_re_index(df_service, 'subject_id')
        return df_service

    def merge_core_hosp_tables(self):
        df_core_merged = self.merge_tables()
        df_hosp_lab_events_merge = self.merge_lab_events_d_labitems()
        df_hosp_emar_poe_merge = self.merge_emar_poe()
        df_core_diagnoses_merge = df_core_merged.merge(self.filter_diagnoses_icd(self._mimic_instance.read_diagnoses_icd()), how='outer', on=['subject_id', 'hadm_id'])\
                                                .merge(self.filter_procedures_icd(self._mimic_instance.read_procedures_icd()), how='outer', on=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
        df_core_hosp_merge = df_core_diagnoses_merge.merge(self.filter_drgcodes(self._mimic_instance.read_drgcodes()), how='outer', on=['subject_id', 'hadm_id'])
        df_core_hosp_merge = df_core_hosp_merge.merge(df_hosp_lab_events_merge, how='outer', on=['subject_id', 'hadm_id'])
        df_core_hosp_merge = df_core_hosp_merge.merge(df_hosp_emar_poe_merge, how='outer', on=['subject_id', 'hadm_id'])\
                                               .merge(self.filter_prescriptions(self._mimic_instance.read_prescriptions()), how='outer', on=['subject_id', 'hadm_id'])\
                                               .merge(self.filter_microbiologyevents(self._mimic_instance.read_microbiologyevents()), how='outer', on=['subject_id', 'hadm_id'])\
                                               .merge(self.filter_service(self._mimic_instance.read_services()), how='outer', on=['subject_id', 'hadm_id'])
        return df_core_hosp_merge
    
    def filter_icustays(self, df):
        df_icustays = df.drop(['hadm_id', 'stay_id', 'intime', 'outtime'], axis=1)
        df_icustays = self._util_instance.remove_duplicates_and_re_index(df_icustays, 'subject_id')
        return df_icustays

    def filter_chartevents(self, df):
        df_chartevents = df.drop(['hadm_id', 'stay_id', 'charttime', 'storetime', 'value', 'valuenum', 'valueuom', 'warning'], axis=1)
        df_chartevents = self._util_instance.remove_duplicates_and_re_index(df_chartevents, 'subject_id')
        return df_chartevents

    def merge_core_hosp_icu_tables(self):
        df_core_hosp_merged = self.merge_core_hosp_tables()
        df_core_hosp_icu_merged = df_core_hosp_merged.merge(self.filter_icustays(self._mimic_instance.read_icustays()), how='outer', on=['subject_id', 'hadm_id', ])\
                                                     .merge(self.filter_chartevents(self._mimic_instance.read_chartevents()), how='outer', on=['subject_id', 'hadm_id', 'stay_id'])
        df_core_hosp_icu_merged.drop(['stay_id'], axis=1)
        return df_core_hosp_icu_merged