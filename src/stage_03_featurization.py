import argparse
import sklearn
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories,get_df
from src.utils.featurize import save_matrix
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


STAGE = "stage 03 featurization" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
     # Load artifacts directory
    artifacts = config["artifacts"]
    prepared_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"],artifacts["PREPARED_DATA"])

    # Train and Test data paths
    train_data_path = os.path.join(prepared_data_dir_path,artifacts["TRAIN_DATA"])
    test_data_path = os.path.join(prepared_data_dir_path,artifacts["TEST_DATA"])

    # Featurized data directory path
    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"],artifacts["FEATURIZED_DATA"])
    create_directories([featurized_data_dir_path])
    # Train and Test  Featurizrd data paths
    featurized_train_data_path = os.path.join(prepared_data_dir_path,artifacts["FEATURIZED_TRAIN_DATA"])
    featurized_test_data_path = os.path.join(prepared_data_dir_path,artifacts["FEATURIZED_TEST_DATA"])
    #print(train_data_path)
    df_train = get_df(train_data_path)
    #print(df_train)
    # Extract txt from the data frame
    train_words = np.array(df_train.text.str.lower().values.astype("U"))  ## >>U1000
    # Loading params 
    max_features = params["featurize"]["max_features"]
    ngrams = params["featurize"]["ngrams"]
    # Loading Bag of words
    bag_words = CountVectorizer(
        stop_words = "english",max_features = max_features,ngram_range = (1,ngrams)
    )
    bag_words.fit(train_words)
    train_words_binary_matrix = bag_words.transform(train_words)
    # Tfidf Transform
    tfidf = TfidfTransformer(smooth_idf=False)
    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)
    save_matrix(df_train, train_words_tfidf_matrix, featurized_train_data_path)
    # Save matrix()
    # Getting on test data for bow and TDF
    df_test = get_df(test_data_path)
    test_words = np.array(df_test.text.str.lower().values.astype("U")) ## << U1000

    test_words_binary_matrix = bag_words.transform(test_words)

    test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)
    save_matrix(df_test, test_words_tfidf_matrix, featurized_test_data_path)
    # Save matrix for test()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e