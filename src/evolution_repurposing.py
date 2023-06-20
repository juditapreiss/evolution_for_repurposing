import os
import re
import ast
import csv
import random
import sklearn
import logging
import argparse
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping
from dateutil import relativedelta
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Builds and tests a deep learning model for the prediction of drug repurposing.")
parser.add_argument('-r', '--repurposing', metavar='FILEPATH', dest='treatment_dates_file', required=True, help='Full path to treatment dates file')
parser.add_argument('-e', '--embedding', metavar='FILEPATH', dest='embedding_file', required=True, help='Full path to embedding file')
parser.add_argument('-s', '--size', metavar='integer', type=int, dest='history_size', default=70, help='Quantity of history to take into account')
parser.add_argument('-o', '--output', metavar='FILEPATH', dest='output_file', required=True, help='Full path to output file')

# Global variables
semrep_year_start = 2006
random_seed = 3
max_length = 10

embedding_vector_length = 50
num_folds = 5
num_epochs = 100
batch_size = 64


def check_date(date):
    result = re.match('^[0-9]+-(01|03|05|07|09|11)-[0-9]+$', date)
    if result:
        return True
    return False


def extract_date_range(end_date, history_size, all_dates):
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - relativedelta.relativedelta(months=history_size)
    start_dt = start_dt - relativedelta.relativedelta(day=1)
    dt_range = []
    for date in all_dates:
        dt = pd.to_datetime(date)
        if start_dt <= dt and dt <= end_dt:
            dt_range.append(date)
    return dt_range


def extract_beginning(y, m):
    if m == "12":
        y = str(int(y) + 1)
        m = "01"
    elif m in ["02","04","06","08","10"]:
        m = str(int(m) + 1).rjust(2, "0")
    beginning = "-".join([y, m])
    return beginning    


def remove_old_dates(dates):
    final_dates = []
    for date in dates:
        entries = date.split('-')
        y = entries[0]

        if int(y) < semrep_year_start:
            continue
        elif y.startswith("2022"):
            continue
        elif date.startswith("2021-12"):
            continue
        # Keep date if not old
        final_dates.append(date)
    return final_dates


def map_semrep_dates(dates, all_dates):
    final_dates = []
    # Dates represents all the dates when a change was detected
    for date in dates:
        entries = date.split('-')
        y = entries[0]

        if int(y) < semrep_year_start:
            continue
        elif y.startswith("2022"):
            continue
        elif date.startswith("2021-12"):
            continue

        m = entries[1].rjust(2,"0")
        beginning = extract_beginning(y, m)
        for d in all_dates:
            if d.startswith(beginning):
                final_dates.append(d)
                break

    return final_dates


def extract_examples(dataset_output):
    global max_length

    # Read in data from file: vectors are in increasing date order
    dataDF = pd.read_csv(dataset_output, header=None)

    # Name the label column and convert to integers
    columns = dataDF.columns
    label_column = len(columns) - 1
    dataDF = dataDF.rename(columns={label_column: 'label'})
    
    # Balance dataset
    positiveDF = dataDF[dataDF['label'] == 1]
    negativeDF = dataDF[dataDF['label'] == 0]

    if positiveDF.shape[0] > negativeDF.shape[0]:
        downsampleDF = resample(positiveDF, replace=False,
                                n_samples=negativeDF.shape[0],
                                random_state=random_seed)
        dataDF = pd.concat([downsampleDF,negativeDF]).sample(frac=1, axis=1).reset_index(drop=True)
    else:
        downsampleDF = resample(negativeDF, replace=False,
                                n_samples= positiveDF.shape[0],
                                random_state=random_seed)
        dataDF = pd.concat([positiveDF,downsampleDF]).sample(frac=1, axis=1).reset_index(drop=True)

    # Print out the value counts
    if max_length < label_column:
        # Drop non required columns
        drop_columns = []
        for i in range(len(columns) - 1 - max_length):
            drop_columns.append(i)
        dataDF = dataDF.drop(columns=drop_columns)
    else:
        max_length = len(columns) - 1

    # Convert columns to proper vectors
    for i in range(len(columns) - 1 - max_length, len(columns) - 1):
        dataDF[i] = dataDF.apply(lambda x: ast.literal_eval(x[i]), axis=1)

    return dataDF, columns
        

def produce_training_data(mergeDF, embeddingDF, all_dates, reduced_dates, history_size, dataset_output):
    csv_file = open(dataset_output, 'w')
    writer = csv.writer(csv_file, delimiter=',')

    # Generate data
    num_examples = 0
    for index, row in mergeDF.iterrows():
        negative_dates = reduced_dates.copy()
        num_positive_examples = 0
        for date in row['dates']:        
            if not date in all_dates:
                continue
            try:
                negative_dates.remove(date)
            except ValueError:
                pass
            date_range = extract_date_range(date, history_size, all_dates)

            # Extract relevant part of embedding
            tempDF = embeddingDF[(embeddingDF.treatment == row['treatment']) & (embeddingDF['date'].isin(date_range))]
            # This is a positive instance
            dataset = tempDF.vector.tolist()
            # Add the label
            dataset.append(1)
            assert len(dataset) == int(history_size / 2) + 2
            writer.writerow(dataset)
            num_positive_examples += 1
            num_examples += 1
            
        # Trying to keep the dataset roughly balanced but varied
        sample_dates = random.sample(negative_dates, min(num_positive_examples, len(negative_dates)))
        
        for date in sample_dates:
            date_range = extract_date_range(date, history_size, all_dates)
            # Extract relevant part of embedding
            tempDF = embeddingDF[(embeddingDF.treatment == row['treatment']) & (embeddingDF['date'].isin(date_range))]
            # This is a negative instance
            dataset = tempDF.vector.tolist()
            # Add the label
            dataset.append(0)
            assert len(dataset) == int(history_size / 2) + 2
            writer.writerow(dataset)
            num_examples += 1

    csv_file.close()
    

def create_model():

    # Best model
    layer1 = 32
    layer2 = 32
    dropout1 = 0.2
    dropout2 = 0.2
    pool_size = 2
    
    # Set up the model
    model = Sequential()

    # Add layers
    model.add(Conv1D(filters=layer1, input_shape=(max_length,embedding_vector_length), kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Bidirectional(LSTM(layer2, dropout=dropout1, recurrent_dropout=dropout2)))

    # Add final classification layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
      

def main(treatment_dates_file, embedding_file, history_size, dataset_output):

    ### Create dataset
    
    # Read treatments
    treatmentDF = pd.read_csv(treatment_dates_file)

    # Read embeddings
    embeddingDF = pd.read_csv(embedding_file, names=["date", "treatment", "vector"])

    # Drop embedding rows that don't correspond to breakpoints
    embeddingDF['umls'] = embeddingDF.apply(lambda x: check_date(x['date']), axis=1)
    embeddingDF = embeddingDF[embeddingDF['umls']]
    all_dates = embeddingDF.date.unique()
    
    # Reduce dates to generate negative examples
    reduced_dates = extract_date_range(all_dates[len(all_dates) - 1], history_size, all_dates)
    
    # Overlap treatments that have entries in both
    mergeDF = treatmentDF.merge(embeddingDF, how='inner', on='treatment').drop(columns=['date', 'vector', 'umls']).drop_duplicates()
    mergeDF['dates'] = mergeDF.apply(lambda x: ast.literal_eval(x['dates']), axis=1)

    mergeDF['dates'] = mergeDF.apply(lambda x: remove_old_dates(x['dates']), axis=1)

    assert mergeDF.treatment.nunique() == mergeDF.shape[0]

    mergeDF['dates'] = mergeDF.apply(lambda x: map_semrep_dates(x['dates'], all_dates), axis=1)
    # Drop any empty dates
    mergeDF = mergeDF[mergeDF['dates'].str.len() > 2]
    
    ### Train and evaluate model

    # Read data into DF
    dataDF, columns = extract_examples(dataset_output)
    
    # Set up stratified cross validation
    matrix = []
    labels = []

    for index, row in dataDF.iterrows():
        matrix_row = []
        for j in range(max_length):
            matrix_row.append(row[(j + len(columns) - 1 - max_length)])
        matrix.append(matrix_row)
        labels.append(row['label'])

    # Set up cross validation
    skf = StratifiedKFold(n_splits=num_folds, random_state=random_seed, shuffle=True)

    matrixn = np.array([np.array(i) for i in matrix])
    labelsn = np.array(labels)
    
    for train_index, test_index in skf.split(matrixn, labelsn):
        x, x_test = matrixn[train_index], matrixn[test_index]
        y, y_test = labelsn[train_index], labelsn[test_index]

        # Separate training into training and validation
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=y, random_state=random_seed)

        # Construct model
        model = create_model()

        # Summarize model
        logger.debug(model.summary())        
        
        # Early stopping to avoid overfitting: stop training when
        # validation loss has not improved after 10 epochs (patience = 10).
        early_stop = EarlyStopping(monitor = 'val_loss', patience = 10)
        
        # Fit model
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=batch_size, use_multiprocessing=True, callbacks = [early_stop])

        # Evaluate model
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Accuracy (max_length: %d): %.2f%%" % (max_length, scores[1]*100))

        # Confusion matrix
        y_pred = model.predict(x_test)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, np.rint(y_pred))
        logger.debug(confusion_matrix)
        classification_report = sklearn.metrics.classification_report(y_test, np.rint(y_pred))
        print(classification_report)

    

if __name__ == '__main__':
    # Parse input arguments
    args = parser.parse_args()
    assert os.path.exists(args.treatment_dates_file)
    assert os.path.exists(args.embedding_file)
    
    main(args.treatment_dates_file, args.embedding_file, args.history_size, args.output_file)
