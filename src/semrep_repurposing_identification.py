import os
import re
import logging
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Produces a CSV file of drugs alongside dates when they are deemed to have been repurposed based on information provided by semrep processed pubmed.")
parser.add_argument('-p', '--predication', metavar='FILEPATH', dest='predication_file', required=True, help='Full path to PREDICTION.csv.gz')
parser.add_argument('-c', '--citation', metavar='FILEPATH', dest='citation_file', required=True, help='Full path to CITATION.csv.gz')
parser.add_argument('-u', '--umls', metavar='DIR', dest='umls_dir', required=True, help='Full path to UMLS directory containing RRF files')
parser.add_argument('-o', '--output', metavar='FILEPATH', dest='output_file', required=True, help='Path to output file (csv will be added if missing)')

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables
treatment_semantic_types = ["T200", "T121", "T103"]
disease_semantic_types = ["T047"]
wanted_predicates = ['TREATS', 'AFFECTS', 'PREVENTS']
citation_names = ['PMID', 'ISSN', 'DP', 'EDAT', 'PYEAR']
citation_usecols = ['PMID', 'EDAT']
predication_names = ['PREDICATION_ID', 'SENTENCE_ID', 'PMID', 'PREDICATE', 'SUBJECT_CUI', 'SUBJECT_NAME', 'SUBJECT_SEMTYPE', 'SUBJECT_NOVELTY', 'OBJECT_CUI', 'OBJECT_NAME', 'OBJECT_SEMTYPE', 'OBJECT_NOVELTY', 'empty1', 'empty2', 'empty3']
predication_usecols = ['PMID', 'PREDICATE', 'SUBJECT_CUI', 'OBJECT_CUI']

def read_mrsty(mrsty_file):
    mrstyDF = pd.read_csv(mrsty_file, delimiter='|', names=['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF', 'Empty'], usecols=['CUI', 'TUI'])
    treatmentDF = mrstyDF[mrstyDF['TUI'].isin(treatment_semantic_types)]
    treatment_list = treatmentDF.CUI.unique()
    diseaseDF = mrstyDF[mrstyDF['TUI'].isin(disease_semantic_types)]
    disease_list = diseaseDF.CUI.unique()
    return treatment_list, disease_list
    

def reduce_predications(predication_file, mrsty_file):

    # Extract UMLS recognised treatments and diseases
    treatment_list, disease_list = read_mrsty(mrsty_file)

    # Extract predications between treatment and disease
    chunksize = 10 ** 6
    treatsDF = pd.DataFrame()
    total = 0
    with pd.read_csv(predication_file, names=predication_names, usecols=predication_usecols, chunksize=chunksize, encoding_errors='replace') as reader:
        for chunk in reader:
            total += chunk.shape[0]
            if treatsDF.empty:
                logger.warn('All predicates: %s' % chunk.PREDICATE.unique())
            chunk = chunk[chunk['PREDICATE'].isin(wanted_predicates)]
            # Force these to be between treatments and diseases
            chunk = chunk[(chunk['SUBJECT_CUI'].isin(treatment_list)) & (chunk['OBJECT_CUI'].isin(disease_list))]
            if treatsDF.empty:
                treatsDF = chunk.drop(columns=['PREDICATE'])
            else:
                treatsDF = pd.concat([treatsDF, chunk.drop(columns=['PREDICATE'])]).drop_duplicates()

    return treatsDF


# Reduce description further where possible
def reduce_description(description):

    delivery_string = "aerosol|aerosol,top|alora patch|\(bov\) top spray|bucc tab|buccal tablet|cap|cap,sa|capsule|carb|\(climara\) patch|chewable tablet|chewing gum|chw tb|chew tab|conc inj|crm|cream|cream,top|disp tab|disintegrating oral tablet|disp tablet|drops|drug implant|ear drops|ec cap|emulsion|emulsion,top|enema|\(esclim\) patch|estraderm patch|extended release oral capsule|extended release oral tablet|eye drop|eye drops|fempatch patch|ec tab|gas|gel|gel,top|\(human\) soln,top|tab|tablet|implant|inhalation aerosol with adapter|inhalation powder|inhalation suspension|inj|inj cart|inj,soln|inj,susp|inj,susp,sa|in|inf|inj,conc|injection|\(iron\) inj|intrauterine system|irrigation|i-v inf|kit|kit inj|liquid|liquid,dent|lotion|mouthwash|nasal kit w/syr|nasal ointment|nasal product|nasal solution|nasal spray|nebulizer solution|oint,top|ointment|oph oint|oph soln|ophthalmic gel|ophthalmic ointment|ophthalmic solution|ophthalmic suspension|\@oral\@capsule|oral capsule|oral gel|oral lozenge|oral product|oral solution|oral soln|oral suspension|\@oral\@tablet|oral tablet|otic solution|otic suspension|paste|patch|pellet|pill|powder|prefilled syringe|pwdr|pwdr,top|rectal cream|rectal product|rectal suppository|ring|ring,vag|rx tab|sa cap|sa disintegrating tab|sa susp inj|sa tab|shampoo|soln|soln inj|soln,top|solution|spray|sublingual powder|supp|supp,vag|suppository|suppositories|susp|swab,top|syringe|syrng|syrnge|tab|tab,chewable|tab,ec|tab,effervsc|tab,sa|top aerosol|top gel|top oint|top pwdr|top soln|topical cream|topical gel|topical lotion|topical oil|topical ointment|topical product|topical solution|topical spray|transdermal patch|transdermal system|vaginal cream|vaginal insert|vaginal gel|vaginal product|vag supp|\(vivelle\) patch|vivelle patch"

    unit_string = "%|%/neo/polymx|gm|gm/amp|gm/pkt|gm/vi|gm/vil|mcg|mcg/inh|mcg/ml|mcg/vil|meq|meq/ml|mg|ml|ml/ml|g|mg/day|mg/ctm|mg/hctz|mg/kit|mg/ml|m/vi|mg/vi|mg/vil|mg/vial|iu|l|hr|hrs|miu|units|unt|unt/g|unt/gm|unt/vil|unit/vil|unt/ml|unt/vil|inj|inj,conc"
    
    result = re.match('(.*) [^ ]*[0-9\.]+ ?(' + unit_string + ') (' + delivery_string + ')', description)
    result2 = re.match('(.+)\@[0-9\.]+[ ]?(%|\@mg|mg)[ ]?(\@oral\@capsule|\@oral\@tablet)', description)
    if result:
        return result.group(1)
    elif result2:
        return result2.group(1)
    else:
        result = re.match('(.+) [0-9\.]+ ?(' + unit_string + ')[,]?', description)
        result2 = re.match('([^ ]+ hcl) [0-9\.]+ ?(' + unit_string + ')', description)
        if result:
            return result.group(1)
        elif result2:
            return result2.group(1)
        else:
            result = re.match('([^ ]+) (' + delivery_string + '|,)', description)
            result2 = re.match('[0-9\.]+[ ]?(mg|ml) ([^ ]+)', description)
            result3 = re.match('[0-9]+ ([^ ]+) (mg)', description)
            result4 = re.match('(oral form|gel) ([^ ]+)', description)
            result5 = re.match('[^ ]+ (' + delivery_string + ',) \[([^\]]+)\]', description)
            if result:
                return result.group(1)
            elif result2:
                return result2.group(2)
            elif result3:
                return result3.group(1)
            elif result4:
                return result4.group(2)
            elif result5:
                return result5.group(1)
            else:            
                if description.endswith(','):
                    return description[:-1]
                else:
                    return description


def read_mrconso(mrconso_file):
    names = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", \
             "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", \
             "CVF", "EXTRA"]
    usecols = ["CUI", "TS", "LAT", "ISPREF", "STR"]
    mrconsoDF = pd.read_csv(mrconso_file, delimiter='|', names=names, usecols=usecols)
    mrconsoDF = mrconsoDF[mrconsoDF.LAT == 'ENG'].drop(columns=['LAT'])
    mrconsoDF['STR'] = mrconsoDF['STR'].str.lower()
    mrconsoDF = mrconsoDF.drop_duplicates()

    # Drop rows corresponding to ispref N if there's a ispref Y
    isPrefDF = mrconsoDF[mrconsoDF['ISPREF'] == 'Y']
    have_pref = isPrefDF.CUI.unique()

    mrconsoDF = mrconsoDF[~((mrconsoDF['CUI'].isin(have_pref)) & (mrconsoDF['ISPREF'] == 'N'))]
    
    # Drop rows corresponding to non preferred TS if there's a preferred one
    # Term status:
    # P Preferred Name
    # S Synonym
    # p Suppressible preferred name
    # s Suppressible synonym
    isTsDF = mrconsoDF[mrconsoDF['TS'] == 'P']
    has_p_ts = isTsDF.CUI.unique()

    mrconsoDF = mrconsoDF[~((mrconsoDF['CUI'].isin(has_p_ts)) & (mrconsoDF['TS'] != 'P'))]
    mrconsoDF = mrconsoDF.drop(columns=['TS', 'ISPREF'])
        
    # Keep shortest description (?)
    mrconsoDF = (
        mrconsoDF.assign(count=(mrconsoDF["STR"].str.len()))
        .sort_values("count")
        .drop_duplicates(subset=["CUI"], keep="first")
    ).drop('count',axis=1)

    mrconsoDF['STR'] = mrconsoDF.apply(lambda x: reduce_description(x['STR']), axis=1)
    mrconsoDF = mrconsoDF.drop_duplicates()

    return mrconsoDF


def pad_date(date):
    result = re.match('^([0-9]+)-([0-9]+)-([0-9]+)$', date)
    if result:
        date = result.group(1) + "-" + result.group(2).rjust(2, "0") + "-" + result.group(3).rjust(2, "0")
    return date
    

def main(predication_file, citation_file, umls_dir, output_file):

    # Read in treatments and diseases from UMLS
    mrsty_file = os.path.join(umls_dir, "MRSTY.RRF")
    assert os.path.exists(mrsty_file)
    
    # Extract relevant information from predication file
    treatsDF = reduce_predications(predication_file, mrsty_file)
    
    # Extract relevant information from citation file
    chunksize = 10 ** 6
    datesDF = pd.DataFrame()
    total = 0
    with pd.read_csv(citation_file, names=citation_names, usecols=citation_usecols, chunksize=chunksize, encoding_errors='replace') as reader:
        for chunk in reader:
            total += chunk.shape[0]
            # Overlap with treatsDF
            tempDF = treatsDF.merge(chunk, how='inner', on='PMID')
            if not tempDF.empty:
                if datesDF.empty:
                    datesDF = tempDF
                else:
                    datesDF = pd.concat([datesDF, tempDF]).drop_duplicates()

    datesDF = datesDF.drop(columns=['PMID'])

    # If the same pair is repeated, we need to keep the first date and
    # drop the rest
    datesDF = datesDF.sort_values('EDAT')

    datesDF = datesDF.drop_duplicates(['SUBJECT_CUI', 'OBJECT_CUI'], keep='first')
    
    # Read treatment definitions from UMLS
    mrconso_file = os.path.join(umls_dir, "MRCONSO.RRF")
    assert os.path.exists(mrconso_file)
    mrconsoDF = read_mrconso(mrconso_file)

    # Merge mrconso information with dates
    cuiDF = datesDF.merge(mrconsoDF, how='inner', left_on='SUBJECT_CUI', right_on='CUI').drop(columns=['CUI']).rename(columns={'STR': 'SUBJECT_STR'})
    cuiDF = cuiDF.merge(mrconsoDF, how='inner', left_on='OBJECT_CUI', right_on='CUI').drop(columns=['CUI']).rename(columns={'STR': 'OBJECT_STR'})

    # Drop duplicates based on STRs as well (?)
    cuiDF = cuiDF.drop_duplicates(['SUBJECT_STR', 'OBJECT_STR'])

    # Rename columns
    cuiDF = cuiDF.rename(columns={'SUBJECT_CUI': 'CUI1', 'OBJECT_CUI': 'CUI2', 'SUBJECT_STR': 'treatment', 'EDAT': 'change_date'})

    # Pad dates
    cuiDF['change_date'] = cuiDF.apply(lambda x: pad_date(x['change_date']), axis=1)
    
    # Reduce changes to treatment names
    cuiDF = cuiDF[['treatment', 'change_date']].drop_duplicates()

    # Group by treatment
    groupsDF = cuiDF.groupby('treatment')['change_date'].apply(list).reset_index(name='dates')
    groupsDF.to_csv(output_file, index=False)

    
if __name__ == '__main__':
    # Parse input arguments
    args = parser.parse_args()
    assert os.path.exists(args.predication_file)
    assert os.path.exists(args.citation_file)
    assert args.predication_file.endswith(".csv.gz")
    assert args.citation_file.endswith(".csv.gz")
    assert os.path.exists(args.umls_dir)
    assert os.path.isdir(args.umls_dir)

    if args.output_file.endswith(".csv"):
        output_file = args.output_file
    else:
        output_file = args.output_file + ".csv"
    
    main(args.predication_file, args.citation_file, args.umls_dir, output_file)
    
