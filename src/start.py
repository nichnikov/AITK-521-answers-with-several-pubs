import os
import json
import pandas as pd
from src.texts_processing import TextsTokenizer
from src.config import (logger,
                        PROJECT_ROOT_DIR)
from src.classifiers import FastAnswerClassifier # FastAnswerClassifier
from sentence_transformers import SentenceTransformer
from src.data_types import Parameters
from transformers import T5Tokenizer, T5ForConditionalGeneration


with open(os.path.join(PROJECT_ROOT_DIR, "data", "config.json"), "r") as jf:
    config_dict = json.load(jf)

parameters = Parameters.parse_obj(config_dict)

stopwords = []
if parameters.stopwords_files:
    for filename in parameters.stopwords_files:
        root = os.path.join(PROJECT_ROOT_DIR, "data", filename)
        stopwords_df = pd.read_csv(root, sep="\t")
        stopwords += list(stopwords_df["stopwords"])

t5_tokenizer = T5Tokenizer.from_pretrained('ai-forever/ruT5-large')
t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), "models", 'models_bss')).to("cuda")

sbert_model = SentenceTransformer(os.path.join(os.getcwd(), "models", "all_sys_paraphrase.transformers"))
mystem_path = os.path.join(os.getcwd(), "models", "mystem")
tokenizer = TextsTokenizer(mystem_path)
tokenizer.add_stopwords(stopwords)
classifier = FastAnswerClassifier(tokenizer, parameters, sbert_model, t5_model, t5_tokenizer)
logger.info("service started...")

# model = SentenceTransformer(os.path.join(PROJECT_ROOT_DIR, "models", "old_paraphrase.transformers"))
# model = SentenceTransformer(os.path.join(PROJECT_ROOT_DIR, "models", "bss_paraphrase.transformers"))

'''
model = SentenceTransformer(os.path.join(PROJECT_ROOT_DIR, "models", "all_sys_paraphrase.transformers"))
# model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
mystem_path = os.path.join(PROJECT_ROOT_DIR, "models", "mystem")
tokenizer = TextsTokenizer(mystem_path)
tokenizer.add_stopwords(stopwords)
classifier = FastAnswerClassifier(tokenizer, parameters, model)
logger.info("service started...")'''