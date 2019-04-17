from keras.optimizers import RMSprop
from datareader import DataReader
from tokenizer import Tokenize
from model import LSTMModel

from config import constants as const
from config import path

data_reader = DataReader(path.train_path, path.test_path, path.med_path)

#data_bunch = DataReader(path.ICSR_YES, path.ICSR_NO)
data_reader.save(path.DATA_OUTPUT_CSV)

reader = Reader(data_reader, max_words=const.MAX_WORDS, max_len=const.MAX_LENGTH)

reader.save_tokenizer(path.TOKENIZER)

lstm_model = LSTMModel(reader, loss='categorical_crossentropy',
                       optimizer='adam', metrics=['categorical_accuracy'],
                       batch_size=const.BATCH_SIZE, epochs=const.EPOCHS)
lstm_model.fit()

lstm_model.save(path.MODEL_JSON, path.MODEL_WEIGHTS)

