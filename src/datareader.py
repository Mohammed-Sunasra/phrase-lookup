import pandas as pd
from config.constants import *

class DataReader:

    def __init__(self, train_file_path, test_file_path, dictionary_path):
        self.train_data = pd.read_csv(train_file_path)
        self.test_data = pd.read_csv(test_file_path)
        self.meddra_dict = pd.read_csv(dictionary_path)
        #self.unique_ids = self.data[MED_ID].unique()
        self.data, self.pt_to_int, self.int_to_pt = self._clean_data()
        

    def _clean_data(self):
        self.train_data = self._drop_null_values(self.train_data)
        self.test_data = self._drop_null_values(self.test_data)
        
        self.train_data[MED_ID] = self.train_data[MED_ID].astype(int)
        self.test_data[MED_ID] = self.test_data[MED_ID].astype(int)
        
        self.meddra_dict.rename(columns={'ID':'id', 'Term':'term','Primary SOC':'primary_soc'}, inplace=True)
        
        self.train_data = self.train_data[self.train_data[MED_ID].isin(self.meddra_dict.id)]
        self.test_data = self.test_data[self.test_data[MED_ID].isin(self.meddra_dict.id)]
        data = pd.concat([self.train_data, self.test_data], axis=0)
        data = data[data[MED_ID].isin(self.meddra_dict.id)]
        data = data[~data[OUTPUT_COL_NAME].isnull()]
        unique_pts = data[OUTPUT_COL_NAME].unique()
        #unique_ids = data[MED_ID].unique()
        int_to_pt = {idx:pt_term for idx, pt_term in enumerate(unique_pts)}
        pt_to_int = {pt_term:idx for idx, pt_term in enumerate(unique_pts)}
        #data = data[~data[OUTPUT_COL_NAME].isnull()]
        #self.int_to_med = {idx:med_id for idx, med_id in enumerate(unique_ids)}
        #self.med_to_int = {med_id:idx for idx, med_id in enumerate(unique_ids)}
        data[OUTPUT_COL_NAME] = data[OUTPUT_COL_NAME].apply(lambda x: pt_to_int[x])
        #data[MED_ID] = data[MED_ID].apply(lambda x: self.med_to_int[x])
        #data = data[~data[OUTPUT_COL_NAME].isnull()]
        return data, pt_to_int, int_to_pt

    def _drop_null_values(self, data):
        data = data.dropna(axis=0, how='all')
        return data

    def save(self, filepath):
        self.data.to_csv(str(filepath))