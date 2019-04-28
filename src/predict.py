from keras.optimizers import RMSprop
from datareader import DataReader
from tokenizer import Tokenize
from model import LSTMModel
from config import constants as const
from config import path
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
import pickle

data_reader = DataReader(path.train_path, path.test_path, path.med_path)

no_of_classes = len(data_reader.int_to_pt)
data_tokenizer = Tokenize(data_reader, max_words=const.MAX_WORDS, max_len=const.MAX_LENGTH, no_of_classes=no_of_classes)

# with open(str(path.TOKENIZER), 'rb') as pkl_file:
#     data_tokenizer = pickle.load(pkl_file)

lstm_model = LSTMModel(data_tokenizer, model_json_path=path.MODEL_JSON_PATH, 
                        model_weights_path=path.MODEL_BEST_WEIGHTS_PATH, eval=True)

X = data_tokenizer.prepare_text(["URINARY SYMPTOMS FOLLOWING OFF LABEL USE OF METHOTREXATE FOR MORPHEA PROFUNDA"])
pt_id, pt_term = lstm_model.predict(X)
print(pt_id)
print(pt_term)
