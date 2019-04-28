from keras.optimizers import RMSprop
from datareader import DataReader
from tokenizer import Tokenize
from model import LSTMModel
from config import constants as const
from config import path
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy

data_reader = DataReader(path.train_path, path.test_path, path.med_path)
data_reader.save(path.DATA_OUTPUT_CSV)

data_tokenizer = Tokenize(data_reader, max_words=const.MAX_WORDS, max_len=const.MAX_LENGTH)
data_tokenizer.save_tokenizer(path.TOKENIZER)

lstm_model = LSTMModel(data_tokenizer, loss='categorical_crossentropy',
                       optimizer='adam', metrics=[categorical_accuracy, top_k_categorical_accuracy],
                       batch_size=const.BATCH_SIZE, epochs=const.EPOCHS)

lstm_model.fit()

lstm_model.save(path.MODEL_JSON_PATH, path.MODEL_WEIGHTS_PATH)

