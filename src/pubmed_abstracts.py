import os
import re
import sys
import gzip
import nltk.data
import logging
import argparse
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

parser = argparse.ArgumentParser(description="Extracts abstracts from PubMed xml.gz files, producing a single file with abstracts ordered chronologically.")
parser.add_argument('-p', '--pubmed', metavar='DIR', dest='pubmed_dir', required=True, help='Full path to PubMed directory')
parser.add_argument('-o', '--output', metavar='FILEPATH', dest='output_file', required=True, help='Full path to output file location')
parser.add_argument('-t', '--temp', metavar='FILEPATH', dest='temp_file', default='/tmp/temp_pubmed.csv', help='Full path to temporary file location')

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variable setting
min_sent_length = 10


def extract_completed_date(chunk):
    if chunk == None:
        return None 
    year = chunk.find('Year').text
    month = chunk.find('Month').text.zfill(2)
    day = chunk.find('Day').text.zfill(2)
    return year + "-" + month + "-" + day


def extract_article_text(chunk):
    if chunk == None:
        return None
    sents = []
    # Extract title of article
    text = chunk.find('ArticleTitle').text
    # Remove any weird or consecutive whitespace
    if text != None:
        text = " ".join(text.split())
        # For foreign articles, this appears in brackets. Strip these.
        if len(text) > 2 and text[0] == '[' and text[len(text) - 2] == ']':
            text = text[1:]
            text = text[0:(len(text) - 2)] + text[len(text) - 1]
        # Sentence split title
        tokenized_text = tokenizer.tokenize(text)
        sents += tokenized_text

    # Go through the abstract
    abstract = chunk.find('Abstract')
    if abstract != None:
        for abstractText in abstract.findall('AbstractText'):
            if abstractText.text != None:
                at = abstractText.text
                # Remove any weird or consecutive whitespace
                at = at.replace('\t', ' ')
                at = " ".join(at.split())
                tokenized_text = tokenizer.tokenize(at)
                sents += tokenized_text
                    
    return sents


def main(pubmed_dir, output_file, unsorted_output):
    # List all baseline files in directory
    files = sorted(os.listdir(args.pubmed_dir))
    
    with gzip.open(unsorted_output, 'wt') as oh:
        for f in files:
            if not f.endswith(".xml.gz"):
                # Skip any other type of file
                continue
    
            path = os.path.join(args.pubmed_dir, f)

            # Parse gzip'ed XML
            handle = gzip.open(path, 'r')
            tree = ET.parse(handle)
            handle.close()
            root = tree.getroot()

            # Extract relevant information
            for pubmedArticle in root.findall('PubmedArticle'):
                medlineCitation = pubmedArticle.find('MedlineCitation')
                for article in medlineCitation.findall('Article'):
                    date = extract_completed_date(medlineCitation.find('DateCompleted'))
                    if date == None:
                        # Look for PubMedPubDate PubStatus="pubmed"
                        pubmedData = pubmedArticle.find('PubmedData')
                        history = pubmedData.find('History')
                        for pmpd in history.findall("PubMedPubDate"):
                            attribute = pmpd.get("PubStatus")
                            if attribute == 'pubmed':
                                date = extract_completed_date(pmpd)
                                break
                        if date == None:
                            # Look for DateRevised (instead of DateCompleted)
                            date = extract_completed_date(medlineCitation.find('DateRevised'))
                            if date == None:
                                # Double check that there's an article there
                                article = medlineCitation.find('Article')
                                if article != None:
                                    abstract = article.find('Abstract')
                                    if abstract != None:
                                        logger.warn("Couldn't find date in: %s (%s)" % (medlineCitation.find("PMID").text, f))
                    
                    if date != None and re.match("[0-9]{4}-[0-9]{2}-[0-9]{2}", date):
                        sents = extract_article_text(medlineCitation.find('Article'))
                        for sent in sents:
                            sent = sent.strip()
                            if not sent.isspace() and len(sent) > min_sent_length:
                                # Now separate out words in the sentences
                                word_tokens = word_tokenize(sent.lower())
                                try:
                                    oh.write('%s\t%s\n' % (date, " ".join(word_tokens)))
                                except:
                                    logger.error('Failed printing on: %s (%s) %s' % (sents, sent, word_tokens))
                                    sys.exit()
                    else:
                        logger.warn('Dropping %s (%s, %s)' % (medlineCitation.find("PMID").text, f, date))

    # Produce sorted output to output_file and remove any multiple tabs
    with open(output_file, "w") as oh:
        zcat = subprocess.Popen(("zcat", unsorted_output), stdout=subprocess.PIPE)
        sort = subprocess.Popen(("sort", "-k", str(1)), stdin=zcat.stdout, stdout=subprocess.PIPE)
        grep = subprocess.Popen(("grep", "-v", "$'" + "\\t.*\\t" + "'"), stdin=sort.stdout, stdout=subprocess.PIPE)
        gzipped = subprocess.Popen(("gzip"), stdin=grep.stdout, stdout=oh)
        gzipped.wait()
    logger.debug('Produced sorted, compressed output to: %s' % output_file)
    
    # Unlink temporary file
    os.unlink(unsorted_output)


if __name__ == '__main__':
    # Parse input arguments
    args = parser.parse_args()
    assert os.path.exists(args.pubmed_dir)
    assert os.path.exists(os.path.dirname(args.temp_file))
    # Check output needs to have gz extension added
    if args.output_file.endswith(".gz"):
        output_file = args.output_file
    else:
        output_file = args.output_file + ".gz"
    # Invoke main
    main(args.pubmed_dir, output_file, args.temp_file)
