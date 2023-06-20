import os
import csv
import pickle
import logging
import argparse
import pandas as pd

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Reduces the evolvemb pkl dynamic embeddings to windows of embeddings.")
parser.add_argument('-p', '--pkl', metavar='FILEPATH', dest='pkl_file', required=True, help='Full path to pkl file')
parser.add_argument('-u', '--umls', metavar='DIR', dest='umls_dir', required=True, help='Full path to UMLS directory containing RRF files')
parser.add_argument('-o', '--output', metavar='FILEPATH', dest='output_file', required=True, help='Full path to output file')

# Gobal variables
drug_sources = ["VANDF", "DRUGBANK", "RXNORM"]



def extract_treatment_names(mrconso_file):
    # Read MRCONSO file
    names = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", \
             "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", \
             "CVF", "EXTRA"]
    usecols = ["CUI", "LAT", "SAB", "CODE", "STR"]
    mrconsoDF = pd.read_csv(mrconso_file, delimiter='|', names=names, usecols=usecols)
    mrconsoDF = mrconsoDF[mrconsoDF.LAT == 'ENG'].drop(columns=['LAT'])
    mrconsoDF['STR'] = mrconsoDF['STR'].str.lower()

    # Discard any lines where STR is too short
    mrconsoDF = mrconsoDF[mrconsoDF['STR'].str.len() > 4]

    # Discard any lines where STR isn't a single word
    mrconsoDF = mrconsoDF[mrconsoDF.STR.apply(lambda x: len(x.split())== 1)]

    # Discard any lines where source (SAB) isn't one of the drug sources
    mrconsoDF = mrconsoDF[mrconsoDF['SAB'].isin(drug_sources)]

    treatment_names_list = mrconsoDF.STR.unique()
    treatment_names_dict = dict.fromkeys(treatment_names_list)

    return treatment_names_dict
    

def main(snapshot_pkl, umls_dir, csv_file):

    # Find treatment names
    mrconso_file = os.path.join(umls_dir, "MRCONSO.RRF")
    assert os.path.exists(mrconso_file)
    treatment_names_dict = extract_treatment_names(mrconso_file)
    
    # Load embeddings
    snapshot_emb = pickle.load(open(snapshot_pkl, "rb"))
    snapshots = sorted(snapshot_emb)

    csv_fp = open(csv_file, "w")
    writer = csv.writer(csv_fp)
    
    for s in range(len(snapshots)):
        for t in snapshot_emb[snapshots[s]].input_model.index2token:
            if t in treatment_names_dict:
                writer.writerow([snapshots[s], t, list(snapshot_emb[snapshots[s]][t])])

    csv_fp.close()
    

if __name__ == '__main__':
    # Parse input arguments
    args = parser.parse_args()
    assert os.path.exists(args.pkl_file)
    assert os.path.exists(args.umls_dir)
    assert os.path.isdir(args.umls_dir)

    output_file = args.output_file
    if not output_file.endswith(".csv"):
        output_file += ".csv"
    
    main(args.pkl_file, args.umls_dir, output_file)
