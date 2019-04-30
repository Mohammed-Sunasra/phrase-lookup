from pathlib import Path

version = 'v1'
data_path = Path("data")
output_path = Path("output_files")
model_path = Path("model_files")
glove_dir = Path("word_embeddings/glove.6B")

train_path = data_path / 'train.csv'
test_path = data_path /'test.csv'
med_path = data_path / 'meddra.csv'

TOKENIZER = output_path / f'tokenizer_{version}.pkl'
DATA_OUTPUT_CSV = output_path / f'data_dict_{version}.pkl'
MODEL_WEIGHTS_PATH = model_path / f'model_lstm_{version}.h5'
MODEL_BEST_WEIGHTS_PATH = model_path / f'model_lstm_best_weights_{version}.h5'
MODEL_JSON_PATH = model_path / f'model_lstm_{version}.json'


