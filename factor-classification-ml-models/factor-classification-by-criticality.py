# %%
# INTRO

# version: simulation-11.31 of 24/Jul/2024

# This script processes a PDF corpus into a data-frame suitable for SHAP value and causal analysis. 
# The script processes corpus' TXT evaluation files of the relevant types: ["PVR", "PCR", "PPER", "PPA", "PCRD", "PPE", "MTR", 
# "PE", "ICR", "ICRR", "PPAR"] without tables or figures, and then into individual TXT sentences. 
# The script also detects and does not process scanned image PDF or documents with 20% of more sentences in languages different 
# than English. 
# The script then performs sentiment analysis with four different algorithms, of which Stanza has proved the closest to human 
# expert polarity scores, on a separate 325 sentence trial. Stanza was then re-trained on a domain-specific data-set for enhanced performances.
# Script's Step # 8 prepares a data-frame with the count of corpus' positive polarity sentences and negative polarity sentences
# that are semantically similar to the factors that influence the performance of partnerships between multilateral agencies.
# SCRIPT PARAMETERS:
#   minimum percentage of English text in PDF documents in Step 2 (e.g. 80%)
#   minimum sentencen length in Step 3 (e.g. 15 words)
#   sentiment analysis model utilized in Step 5 (e.g. VADER, TextBlob, Stanza, Stanza-1, Stanza-Retrained, Stanza-Retrained-1, Flair)
#   neutral sentiment upper ceiling (e.g. 0.6) and lower threshold (e.g. -0.6) in Step 6 



# %%
# Step 1 PART A PDF to TXT corpus

# Skipping scanned image PDF, and removing tables and figures from the text PDF files. Screening and counting non-English language documents.

import os
import re
from pathlib import Path
import fitz  # PyMuPDF
from tqdm.notebook import tqdm
from langdetect import detect, LangDetectException
from joblib import Parallel, delayed


def clean_text(text):
    # Clean the text by removing tables, figures, and unnecessary characters
    text = re.sub(r'\b(Table|Figure)\s+\S+\b', '', text)
    text = re.sub(r'^[0-9\s\W]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{3,}', '', text)
    return text

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def save_pdf_as_txt(pdf_path, output_dir_full_text):
    doc = fitz.open(pdf_path)
    all_text = ""
    english_sentences = 0
    total_sentences = 0

    contains_text = False
    for page in doc:
        if page.get_text().strip():
            contains_text = True
            text = page.get_text()
            cleaned_text = clean_text(text)
            sentences = re.split(r'(?<=[.!?]) +', cleaned_text)
            for sentence in sentences:
                if sentence.strip():
                    total_sentences += 1
                    if is_english(sentence):
                        english_sentences += 1
                    all_text += sentence + "\n"

    if not contains_text:
        print(f"Skipping {os.path.basename(pdf_path)} as it appears to be a scanned image without text.")
        return False

    english_ratio = english_sentences / total_sentences if total_sentences > 0 else 0

    if english_ratio < 0.8 or not all_text.strip():
        print(f"Skipping {os.path.basename(pdf_path)} due to high non-English content in more than 20% of sentences.")
        return False

    full_text_file_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
    full_text_file_path = os.path.join(output_dir_full_text, full_text_file_name)
    with open(full_text_file_path, 'w', encoding='utf-8') as full_text_file:
        full_text_file.write(all_text)
    print(f"Full text saved for {os.path.basename(pdf_path)} to {output_dir_full_text}")
    return True



# %%
# Saving only TXT documents that are evaluations.

# Parallelized

import os
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

# Assuming the functions clean_text, is_english, and save_pdf_as_txt are already defined as in the previous example

def process_single_file(file_name, input_dir, output_dir_full_text, relevant_doc_types):
    input_pdf_path = os.path.join(input_dir, file_name)
    parts = file_name.split('_')  # Splitting file_name to check against relevant_doc_types
    
    if len(parts) >= 3 and parts[2] in relevant_doc_types:
        was_processed = save_pdf_as_txt(input_pdf_path, output_dir_full_text)
        return was_processed, file_name
    return False, file_name

def process_corpus(input_dir, output_dir_full_text, relevant_doc_types):
    # Expand the user path if it contains the ~ symbol
    if input_dir.startswith('~'):
        input_dir = os.path.expanduser(input_dir)
    if output_dir_full_text.startswith('~'):
        output_dir_full_text = os.path.expanduser(output_dir_full_text)
    
    Path(output_dir_full_text).mkdir(parents=True, exist_ok=True)
    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    total_docs = len(files_to_process)
    print(f"Total documents found: {total_docs}")

    # Use joblib to process files in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_single_file)(file_name, input_dir, output_dir_full_text, relevant_doc_types) 
        for file_name in tqdm(files_to_process, desc="Progress in saving TXT corpus:")
    )
    
    docs_processed = sum(1 for result, _ in results if result)
    docs_skipped_due_to_language = sum(1 for result, _ in results if not result)

    # Print summary
    print(f"\nSummary:")
    print(f"Total documents: {total_docs}")
    print(f"Documents processed and kept: {docs_processed}")
    print(f"Documents skipped due to language: {docs_skipped_due_to_language}")
    if total_docs > 0:  # Prevent division by zero
        percentage_kept = (docs_processed / total_docs) * 100
        percentage_skipped = (docs_skipped_due_to_language / total_docs) * 100
        print(f"Percentage of documents kept: {percentage_kept:.2f}%")
        print(f"Percentage of documents skipped: {percentage_skipped:.2f}%")

if __name__ == "__main__":
    input_dir = input("Enter the path to the directory containing the PDF documents: ")
    output_dir_full_text = input("Enter the path to the output directory to save the complete TXT version of each PDF file: ")
    relevant_doc_types = ["PVR", "PCR", "PPER", "PPA", "PCRD", "PPE", "MTR", "PE", "ICR", "ICRR", "PPAR"]
    
    process_corpus(input_dir, output_dir_full_text, relevant_doc_types)




# %%
# Step 2, from TXT to BoW sentences

# Package download, independent from the previous step
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from pathlib import Path

# Downloads necessary for NLTK processing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# %%
# Step 3 from TXT documents to clean sentences

# ATTENTION: Prepared Stop Words and Lemmatizer Outside the Parallelized Function may alter output compared to versions
# up to corpus-processing-11.22. In such a case, use Step 3 from corpus-processing-11.21.

# Parallelized
# Compared to the original 11.22 script, the parallelization required to avoid passing NLTK objects directly to the 
# parallelized function. Instead, these objects are prepared outside the parallelized function and only the necessary 
# data is passed.
# multiprocessing was introduced in version 11.29 instead of joblib because joblib caused global sentence counting (senNGlo) issues
# due to pickling problems that could not be solved otherwise. Trial to avoid lock, by means of internal initialization, didn't work.

# Added dynamic input of minimum sentence length in terms of lemmatized word number.
# Script skips documents without sentences longer than min_length


import os
from pathlib import Path
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool, Manager, cpu_count

# Function to prepare text by tokenizing, lemmatizing, and removing stop words
def prepare_text(text, stop_words, lemmatizer):
    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return " ".join(filtered_words)

# Function to process text from a file, tokenize into sentences, and save each sentence as a separate file
def process_text(args):
    file_path, output_dir, eliminate_short_sentences, min_length, stop_words, lemmatizer, sen_glo_counter, counter_lock = args
    file_name_parts = os.path.basename(file_path).split('_')
    org_acro, doc_type, pro_code = file_name_parts[:3]
    sen_doc_counter = 0  # Initialize document-specific sentence counter
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    sentences = sent_tokenize(text)
    valid_sentences = []
    
    for sentence in sentences:
        prepared_sentence = prepare_text(sentence, stop_words, lemmatizer)
        if eliminate_short_sentences and len(word_tokenize(prepared_sentence)) < min_length:
            continue  # Skip sentences shorter than the specified minimum length
        valid_sentences.append(prepared_sentence)
    
    if not valid_sentences:
        # If no valid sentences, skip this document
        return None
    
    for prepared_sentence in valid_sentences:
        # Ensure atomic update to the global counter
        with counter_lock:
            local_sen_glo_counter = sen_glo_counter.value
            sen_glo_counter.value += 1
        
        sentence_file_name = f"{org_acro}_{doc_type}_{pro_code}_{local_sen_glo_counter}_{sen_doc_counter}.txt"
        output_path = os.path.join(output_dir, sentence_file_name)
        
        with open(output_path, 'w', encoding='utf-8') as sentence_file:
            sentence_file.write(prepared_sentence)
        
        sen_doc_counter += 1
    
    return local_sen_glo_counter

def main():
    # Paths should be set dynamically for Jupyter Lab environment
    current_dir = os.getcwd()
    
    # Initialize stop words and lemmatizer outside the parallelized function
    stop_words = set(stopwords.words('english')) - {
        "but", "M&E", "should", "while", "whereas", "more", "against", "not", "cannot", "can not", 
        "will not", "won't", "would not", "wouldn't", "should not", "shouldn't", "could not", 
        "couldn't", "did not", "didn't", "does not", "doesn't", "do not", "don't", "have not", 
        "haven't", "has not", "hasn't", "had not", "hadn't", "were not", "weren't", "was not", 
        "wasn't", "are not", "aren't", "is not", "isn't", "fail to", "lack of", "without", 
        "absence of", "fall short"
    }
    lemmatizer = WordNetLemmatizer()
    
    # Ask if the user wants to re-use sentences from previous steps
    reuse = input("\nDo you want to re-use the sentences of the previous steps as input for the sentiment analysis? yes or no: \n").lower().strip() == "yes"
    if reuse:
        # If yes, get the path to the lemmatized TXT files from step 2
        input_dir = input("Enter the path where your lemmatized TXT files from step 2 are saved: ")
    else:
        # If no, get the path to the input sentences for sentiment analysis
        input_dir = input("\nPlease write the path to the input sentences for sentiment analysis.\n")
    
    # Get the output directory for saving the processed sentences
    output_dir = input("\nEnter the path for processed sentences saving:\n")

    # Expand user paths for input and output directories
    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.expanduser(output_dir)

    # Ensure input and output directories exist
    if not os.path.isdir(input_dir):
        print(f"Error: The directory {input_dir} does not exist.")
        return
    if not os.path.isdir(output_dir):
        print(f"Error: The directory {output_dir} does not exist.")
        return

    # Initialize global sentence counter and lock using Manager
    manager = Manager()
    sen_glo_counter = manager.Value('i', 0)
    counter_lock = manager.Lock()

    # Get the list of TXT files in the input directory
    files = sorted(os.listdir(input_dir))
    files = [f'{input_dir}/{file}' for file in files if file.endswith(".txt")]
    pbar = tqdm(total=len(files), desc="Processing Text Files")

    # Use all available CPUs for parallel processing
    n_jobs = cpu_count()

    # Get the minimum number of words for a sentence to be considered
    min_length = int(input("\nEnter the minimum number of words a sentence must have to be considered: "))

    # Prepare arguments for processing
    args = [(file, output_dir, True, min_length, stop_words, lemmatizer, sen_glo_counter, counter_lock) for file in files]

    # Perform text processing in parallel
    with Pool(n_jobs) as pool:
        for _ in pool.imap_unordered(process_text, args):
            pbar.update(1)

    pbar.close()
    print(f"Processing completed. Files are saved in {output_dir}.")

if __name__ == "__main__":
    main()


# %%
# Step 4: TXT sentence polarity scores

# ULTRA-LONG: 3 days on whole corpus

# The 10.04 script version introduced several improvements to speed execution up and improve monitoring:
# Stanza's one sole initialization;
# Stanza parallelization and threading, locking to avoid conflicts;
# Global progress bar.
# SPEED on i5 4PCU 9.7 Gb RAM: 3h 15' 47" for 8 projects with 2480 sentences (24.4 min/project, 4.7 seconds/sentence)
# SPEED on i7 4CPU 16. GB RAM: 1h 08' 55" for same input ( 8.6 min/project, 1.7 seconds/sentence)
# SPEED on server 16 CPU 197 Gb RAM and the 11.31 "private" Stanza: 0.4 seconds/sentence (0.8 s/sentence with 20 words)

# The 11.22 version introduced the dynamic activation of all available CPUs
# The 11.29 changed to private mode from shared mode speed 0.6 seconds/sentence
# The 11.31 introduced:
#       the "private" initialization of Stanza in each thread first used on corpus on 24-06-2024, WORKS
#       removed lock, after adding it againg to safely have checkpoints and a stable TQDM bar. Lock used too much RAM 105 Gb.


import os
import time
from tqdm import tqdm
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import stanza
from flair.models import TextClassifier
from flair.data import Sentence
from joblib import Parallel, delayed, parallel_backend
from threading import Lock
from multiprocessing import cpu_count

# Initialize Sentiment Analysis tools
stanza.download('en')
pbar = tqdm(total=0, desc="Sentiment Analysis", disable=True)
stanza_retrained_model_dir = None


def perform_sentiment_analysis(files, ask_flair):
    global stanza_retrained_model_dir
    print("Performing sentiment analysis...")
    data = []
    stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    vader_analyzer = SentimentIntensityAnalyzer()
    flair_classifier = TextClassifier.load('en-sentiment')
    # Initialize the re-trained Stanza model
    stanza_retrained_nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment', sentiment_model_path=stanza_retrained_model_dir,
                                        tokenize_pretokenized=True, tokenize_no_ssplit=True)

    for file in files:
        file_name = os.path.basename(file)
        parts = file_name.split('_')
        if len(parts) < 5:  # Adjust based on the naming convention
            continue

        org_acro, pro_code, doc_type, sen_nglo, sen_ndoc = parts[0], parts[1], parts[2], parts[-2], parts[-1].split('.')[0]
        
        with open(file, 'r', encoding='utf-8') as file:
            content = file.read()

        # VADER Sentiment Analysis
        vader_scores = vader_analyzer.polarity_scores(content)
        sen_vader = "{:.3f}".format(vader_scores['compound'])

        # TextBlob Sentiment Analysis
        sen_textblob = "{:.3f}".format(TextBlob(content).sentiment.polarity)

        # Stanza Sentiment Analysis
        doc = stanza_nlp(content)
        stanza_score = doc.sentences[0].sentiment if doc.sentences else -1  # Use -1 as N/A
        sen_stanza = "{:.3f}".format(stanza_score)
        
        # Calculate SenSTANZA-1
        sen_stanza_minus_one = "{:.3f}".format(float(sen_stanza) - 1) if stanza_score != -1 else "N/A"

        # Retrained Stanza Sentiment Analysis
        try:
            max_seq_length = 512
            if len(content.split()) > max_seq_length:
                content = ' '.join(content.split()[:max_seq_length])  # Truncate content

            doc = stanza_retrained_nlp(content)
            stanza_score = doc.sentences[0].sentiment if doc.sentences else -1  # Use -1 as N/A
            sen_stanza_retrained = "{:.3f}".format(stanza_score)
            sen_stanza_retrained_minus_one = "{:.3f}".format(float(sen_stanza_retrained) - 1) if stanza_score != -1 else "N/A"
        except RuntimeError as e:
            print(f"Error processing file {file_name} with re-trained Stanza model: {e}")
            sen_stanza_retrained = "Error"
            sen_stanza_retrained_minus_one = "Error"

        # Flair Sentiment Analysis
        sen_flair = "Not Analyzed"
        if ask_flair == "yes":
            flair_sentence = Sentence(content)
            flair_classifier.predict(flair_sentence)
            flair_score = flair_sentence.labels[0].score
            sen_flair = "{:.3f}".format(flair_score) if flair_sentence.labels[0].value == 'POSITIVE' else "-{:.3f}".format(flair_score)

        data.append({
            "orgAcro": org_acro,
            "proCode": pro_code,
            "docType": doc_type,
            "senNGlo": sen_nglo,
            "senNDoc": sen_ndoc,
            "SenVADER": sen_vader,
            "SenTEXTBLOB": sen_textblob,
            "SenSTANZA": sen_stanza,
            "SenSTANZA-1": sen_stanza_minus_one,
            "SenSTANZA_retrained": sen_stanza_retrained,
            "SenSTANZA_retrained-1": sen_stanza_retrained_minus_one,
            "SenFLAIR": sen_flair,
            "Text": content
        })
        pbar.update(1)

    df = pd.DataFrame(data)
    return df


def main():
    global pbar,stanza_retrained_model_dir

    # Paths should be set dynamically for Jupyter Lab environment
    current_dir = os.getcwd()
    
    # Ask if the user wants to re-use sentences from previous steps
    reuse = input("\nDo you want to re-use the sentences of the previous steps as input for the sentiment analysis? yes or no: \n").lower().strip() == "yes"
    if reuse:
        # If yes, get the path to the lemmatized TXT files from step 2
        input_dir = input("Enter the path where your lemmatized TXT files from step 2 are saved: ")
    else:
        # If no, get the path to the input sentences for sentiment analysis
        input_dir = input("\nPlease write the path to the input sentences for sentiment analysis.\n")
    
    # Get the output directory for saving the processed polarity score file
    output_dir = input("\nEnter the DIRECTORY path for processed polarity score file saving:\n")
    output_path = os.path.join(output_dir, "sentiment_analysis_results_private.xlsx")

    # Ask if the user wants to use Flair for sentiment analysis
    ask_flair = input("\nFLAIR takes a bit longer. Do you want to carry out the sentiment analysis with FLAIR: yes/no?\n").lower().strip()

    # Get the directory for the re-trained Stanza model
    stanza_retrained_model_dir = input("\nPlease input the complete Re-Trained Stanza model path to the aida_test file including file name:\n")

    # Expand user paths for input and output directories
    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.expanduser(output_dir)
    stanza_retrained_model_dir = os.path.expanduser(stanza_retrained_model_dir)

    # Ensure input and output directories exist
    if not os.path.isdir(input_dir):
        print(f"Error: The directory {input_dir} does not exist.")
        return
    if not os.path.isdir(output_dir):
        print(f"Error: The directory {output_dir} does not exist.")
        return

    

    # Get the list of TXT files in the input directory
    files = sorted(os.listdir(input_dir))
    files = [f'{input_dir}/{file}' for file in files if file.endswith(".txt")]
    pbar = tqdm(total=len(files), desc="Sentiment Analysis")

    # Split files into batches, according to number of jobs
    n_jobs = cpu_count()  # This can be made dynamic based on user input if needed
    step = len(files) // n_jobs
    file_batches = []
    for job in range(n_jobs):
        if job != n_jobs - 1:
            file_batches.append(files[job * step:(job + 1) * step])
        else:
            file_batches.append(files[job * step:])

    # Perform sentiment analysis
    results = Parallel(n_jobs=n_jobs,backend='threading')(
        delayed(perform_sentiment_analysis)(
            file_batch, ask_flair
        ) for file_batch in tqdm(file_batches, desc='Performing sentiment analysis')
    )


    # Combine results into a single DataFrame
    df = pd.concat(results, axis=0)
    df.to_excel(output_path, index=False)
    print(f"Sentiment analysis results saved to {output_path}.")

if __name__ == "__main__":
    main()



# %%
# Step 5: semantic similarity between corpus sentences and factors

# Imports and setting the environment: serialized tokenization, parallelized embeddings
# Jupyter Notebook ref.: step5-11.31-01-experiment-sequential-tokenization-parallel-embedding-WORKS.ipynb
# First test on corpus: 24/06/2024

# Develops common corpus (project) and factor tfidf_vectorizer.fit_transform dictionary and tfidf_vectorizer.pkl to re-use in Step 7 and Step 8
# 29-08-2024, modified sBERT chunking into "factor_chunk_size = max(1, len(factor_sentences) // n_jobs)" to cater for no. factors (jobs) < no. CPU.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from scipy.sparse import save_npz
from joblib import Parallel, delayed
import joblib

print("\nI will hereafter ask you no. 5 input.\n")

# Input paths and headers
corpus_sentences_path = input("\n1) Please write here the path to the corpus sentences to embed:\n")
partnership_factors_path = input("\n2) Please write here the path to the XLSX partnership factor data-frame to embed, including the complete file name:\n")
factor_column_header = input("\n3) Please write here the header of the exact data-frame factor column to process:\n")
polarity_scores_path = input("\n4) Please write here the path to the polarity score data-frame, including the complete file name:\n")
selected_polarity_scores_header = input("\n5) Please write here the header of the selected polarity scores to use in the data-frame:\n")

# Ensure paths are expanded correctly to handle both home directory and external drives
corpus_sentences_path = os.path.expanduser(corpus_sentences_path)
partnership_factors_path = os.path.expanduser(partnership_factors_path)
polarity_scores_path = os.path.expanduser(polarity_scores_path)

# Function to read all TXT files in a directory, return a list of sentences, and keep track of sentence numbering
def read_corpus_sentences_and_extract_details(directory_path):
    sentences = []
    sentence_details = []  # To store details extracted from file names
    
    # Ensure the directory path is expanded
    directory_path = os.path.expanduser(directory_path)

    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"The directory {directory_path} does not exist.")
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            # Extracting sentence details from the file name
            parts = filename.replace('.txt', '').split('_')
            if len(parts) < 5:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            orgAcro, docType, proCode, senNGlo, senNdoc = parts
            file_path = os.path.join(directory_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:  # Ensure the line is not empty
                        sentences.append(line)
                        # Append a dictionary with details for each sentence
                        sentence_details.append({
                            "orgAcro": orgAcro,
                            "docType": docType,
                            "proCode": proCode,
                            "senNGlo": senNGlo,
                            "senNdoc": senNdoc,
                            "sentence": line
                        })

    return sentences, sentence_details

# Loading corpus sentences from a directory and extracting details
corpus_sentences, corpus_sentence_details = read_corpus_sentences_and_extract_details(corpus_sentences_path)

# Explain the meaning of the two tqdm progress bars, focus on bar 1 (corpus sentence embedding)
print("\nThese progress bars relate to sentence embedding with sBERT's SentenceTransformer's .encode() method, namely:\n")
print("\nBar 1 relates to corpus sentences embedding.\n")

# Function to embed sentences with sBERT
def embed_sentences_with_sbert(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(sentences, show_progress_bar=True)

# Function to embed sentences with TF-IDF
def embed_sentences_with_tfidf(sentences, vectorizer):
    return vectorizer.fit_transform(sentences)

# Determine the number of available CPU cores
n_jobs = os.cpu_count()
print(f"Using {n_jobs} CPUs for parallel processing.")

# Use joblib for embedding corpus sentences with sBERT
chunk_size = len(corpus_sentences) // n_jobs
sentence_chunks = [corpus_sentences[i:i + chunk_size] for i in range(0, len(corpus_sentences), chunk_size)]

sbert_embeddings_list = Parallel(n_jobs=n_jobs)(
    delayed(embed_sentences_with_sbert)(chunk) for chunk in sentence_chunks
)

# Concatenate results
sbert_embeddings = np.vstack(sbert_embeddings_list)

# Embed the sentences recorded in the factor data-frame
factors_df = pd.read_excel(partnership_factors_path)
factor_sentences = factors_df[factor_column_header].tolist()

# Combine corpus and factor sentences for fitting the TF-IDF vectorizer
combined_sentences = corpus_sentences + factor_sentences

# Fit the TF-IDF vectorizer on the combined sentences and save it for reuse
tfidf_vectorizer = TfidfVectorizer()
tfidf_embeddings = tfidf_vectorizer.fit_transform(combined_sentences)

# Save the fitted vectorizer
vectorizer_path = input("Please write the path to save the fitted TfidfVectorizer for re-use in Step 8 TF-IDF (e.g., 'tfidf_vectorizer.pkl'):\n")
vectorizer_path = os.path.expanduser(vectorizer_path)
joblib.dump(tfidf_vectorizer, vectorizer_path)

# Transform corpus and factor sentences separately using the already fitted vectorizer
corpus_tfidf_embeddings = tfidf_vectorizer.transform(corpus_sentences)
factor_tfidf_embeddings = tfidf_vectorizer.transform(factor_sentences)

# Explain the meaning of the two tqdm progress bars, focus on bar 2 (factor sentence embedding)
print("\nBar 2 relates to factor sentences embedding.\n")

# Use joblib for embedding factor sentences with sBERT
# factor_chunk_size = len(factor_sentences) // n_jobs   # this does not work when the factor number (that is the job number) is < number of CPUs
factor_chunk_size = max(1, len(factor_sentences) // n_jobs) # when the factor number (that is the job number) is < the number of CPUs, it takes value 1.
factor_sentence_chunks = [factor_sentences[i:i + factor_chunk_size] for i in range(0, len(factor_sentences), factor_chunk_size)]

factor_sbert_embeddings_list = Parallel(n_jobs=n_jobs)(
    delayed(embed_sentences_with_sbert)(chunk) for chunk in factor_sentence_chunks
)

# Concatenate results
factor_sbert_embeddings = np.vstack(factor_sbert_embeddings_list)

# ASK for dynamic file paths to save the embeddings
factor_sbert_embeddings_path = input("\nPlease write the FILE dynamic path for the FACTORS sBERT embeddings (e.g. drago@drago-vm0:~/evaluation-sentiment-analysis/simulations/simulation-01/embeddings/sbert-embeddings/factor_sbert_embeddings.npy):\n")
corpus_sbert_embeddings_path = input("\nPlease write the FILE dynamic path for the CORPUS sBERT embeddings (e.g. drago@drago-vm0:~/evaluation-sentiment-analysis/simulations/simulation-01/embeddings/sbert-embeddings/corpus_sbert_embeddings.npy):\n")
factor_tfidf_embeddings_path = input("\nPlease write the FILE dynamic path for the FACTORS TF-IDF embeddings (e.g. drago@drago-vm0:~/evaluation-sentiment-analysis/simulations/simulation-01/embeddings/sbert-embeddings/factor_tfidf_embeddings.npz):\n")
corpus_tfidf_embeddings_path = input("\nPlease write the FILE dynamic path for the CORPUS TF-IDF embeddings (e.g. drago@drago-vm0:~/evaluation-sentiment-analysis/simulations/simulation-01/embeddings/sbert-embeddings/corpus_tfidf_embeddings.npz):\n")

# Ensure paths are expanded correctly to handle both home directory and external drives
factor_sbert_embeddings_path = os.path.expanduser(factor_sbert_embeddings_path)
corpus_sbert_embeddings_path = os.path.expanduser(corpus_sbert_embeddings_path)
factor_tfidf_embeddings_path = os.path.expanduser(factor_tfidf_embeddings_path)
corpus_tfidf_embeddings_path = os.path.expanduser(corpus_tfidf_embeddings_path)

# Save sBERT embeddings for corpus and factors to disk
np.save(corpus_sbert_embeddings_path, sbert_embeddings)
np.save(factor_sbert_embeddings_path, factor_sbert_embeddings)

# Save TF-IDF embeddings for corpus and factors to disk
save_npz(corpus_tfidf_embeddings_path, corpus_tfidf_embeddings)
save_npz(factor_tfidf_embeddings_path, factor_tfidf_embeddings)


# %%
# Step 6: factor-corpus sentence SIMILARITY TABLES frequency based on sBERT embeddings of Step 5
# This step counts the number of corpus sentences that are semantically similar to factors by factor, polarity score, and semantic similarity ratio.
# CORPUS-LEVEL FREQUENCIES, TRANSFORMER-BASED, parallelized

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

# Function to calculate cosine similarities between two sets of embeddings
def calculate_cosine_similarities(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2)

# Categorization of polarity scores based on user-defined thresholds
def categorize_polarity(score, upper_threshold, lower_threshold):
    if score > upper_threshold:
        return 'positive'
    elif score < lower_threshold:
        return 'negative'
    else:
        return 'neutral'

# Function to prepare similarity tables
def prepare_similarity_tables(embeddings_corpus, embeddings_factors, factors_list_path, polarity_scores_path, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, output_path):
    # Load factors list to retrieve the actual factor text
    factors_list_df = pd.read_excel(factors_list_path)
    
    # Check if 'factor' column exists in the factors list DataFrame
    if 'factor' not in factors_list_df.columns:
        print("'factor' column not found in the provided factors list Excel file. Please check the column name.")
        return

    # Load polarity scores DataFrame
    polarity_df = pd.read_excel(polarity_scores_path)

    # Handle "Error" values by replacing them with 0.000 and remove apostrophes
    polarity_df = polarity_df.replace("Error", 0.000)
    
    # Define the columns to modify
    columns_to_modify = ['SenVADER', 'SenTEXTBLOB', 'SenSTANZA', 'SenSTANZA-1', 'SenSTANZA_retrained', 'SenSTANZA_retrained-1', 'SenFLAIR']
    
    # Remove apostrophes and ensure float format
    for column in columns_to_modify:
        if column in polarity_df.columns:
            polarity_df[column] = polarity_df[column].astype(str).str.replace("'", "")
            polarity_df[column] = polarity_df[column].astype(float)
            polarity_df[column] = polarity_df[column].map(lambda x: f"{x:.3f}")

    # Ensure integer columns remain as integers
    integer_columns = ['senNGlo', 'senNDoc']
    for column in integer_columns:
        if column in polarity_df.columns:
            polarity_df[column] = polarity_df[column].astype(int)
    
    # Save the modified polarity scores DataFrame with the "removed_errors" and the "sbert" suffix
    polarity_scores_dir, polarity_scores_filename = os.path.split(polarity_scores_path)
    modified_polarity_scores_filename = polarity_scores_filename.replace(".xlsx", "_modified_errors_sbert.xlsx")
    modified_polarity_scores_path = os.path.join(polarity_scores_dir, modified_polarity_scores_filename)
    polarity_df.to_excel(modified_polarity_scores_path, index=False)

    # Re-load the modified polarity scores DataFrame
    polarity_df = pd.read_excel(modified_polarity_scores_path)
    polarity_df.set_index('senNGlo', inplace=True)

    # Prepare DataFrame templates for positive, neutral, and negative tables
    columns = ['factor_id', 'factor_text'] + [f'ratio_{ratio:.1f}' for ratio in similarity_ratios]
    tables = {category: pd.DataFrame(columns=columns) for category in ['positive', 'neutral', 'negative']}
    
    # Calculate cosine similarities
    cos_sim_matrix = calculate_cosine_similarities(embeddings_corpus, embeddings_factors)

    # Function to process each factor
    def process_factor(i, factor_row):
        # Retrieve the actual factor text from the factors list DataFrame
        factor_text = factors_list_df.at[i, 'factor']
        factor_id = i  # Assuming factor ID is the index, adjust as necessary

        # Initialize counts for each similarity ratio
        counts = {ratio: {'positive': 0, 'neutral': 0, 'negative': 0} for ratio in similarity_ratios}

        for j, sim_score in enumerate(cos_sim_matrix[:, i]):
            polarity_score = polarity_df.at[j, selected_polarity_scores_header]
            corpus_sen_polarity = categorize_polarity(polarity_score, upper_threshold, lower_threshold)
            for ratio in similarity_ratios[::-1]:  # Modification to prevent count reverse cumulation
                if sim_score >= ratio:
                    counts[ratio][corpus_sen_polarity] += 1
                    break  # Modification to prevent count reverse cumulation
        
        return factor_id, factor_text, counts

    # Process factors in parallel
    results = Parallel(n_jobs=-1)(delayed(process_factor)(i, factor_row) for i, factor_row in tqdm(enumerate(embeddings_factors), total=len(embeddings_factors), desc="Processing factors"))

    # Populate tables with results
    for factor_id, factor_text, counts in results:
        for category in ['positive', 'neutral', 'negative']:
            row = [factor_id, factor_text] + [counts[ratio][category] for ratio in similarity_ratios]
            new_row = pd.DataFrame([row], columns=columns)
            tables[category] = pd.concat([tables[category], new_row], ignore_index=True)

    # Save tables to XLSX
    output_path = os.path.expanduser(output_path)
    with pd.ExcelWriter(output_path) as writer:
        for category, df in tables.items():
            df.to_excel(writer, sheet_name=category, index=False)
    
    print(f"Tables saved to '{output_path}'.")

# Dynamic inputs for embeddings and polarity scores
corpus_sbert_embeddings_path = input("\nPlease write the complete path to the corpus sBERT embeddings .npy file:\n")
factor_sbert_embeddings_path = input("\nPlease write the complete path to the factor_sBERT embeddings .npy file:\n")
factors_list_path = input("\nPlease write the path to the factor list Excel file, including the complete file name:\n")
polarity_scores_path = input("\nPlease write the path to the polarity score data-frame, including the complete file name:\n")
selected_polarity_scores_header = input("\nPlease write the header of the selected polarity scores to use in the data-frame:\n")
upper_threshold = float(input("\nEnter the upper threshold for positive polarity scores: write a float (e.g. 0.6)\n"))
lower_threshold = float(input("\nEnter the lower threshold for negative polarity scores: write a float (e.g. -0.6)\n"))
similarity_ratios = np.arange(0.3, 0.9, 0.1)
output_path = input("\nPlease write the complete path where the similarity tables will be saved, including the file name:\n")

# Ensure paths are expanded correctly to handle both home directory and external drives
corpus_sbert_embeddings_path = os.path.expanduser(corpus_sbert_embeddings_path)
factor_sbert_embeddings_path = os.path.expanduser(factor_sbert_embeddings_path)
factors_list_path = os.path.expanduser(factors_list_path)
polarity_scores_path = os.path.expanduser(polarity_scores_path)
output_path = os.path.expanduser(output_path)

# Load sBERT embeddings
corpus_sbert_embeddings = np.load(corpus_sbert_embeddings_path)
factor_sbert_embeddings = np.load(factor_sbert_embeddings_path)

# Call the function with the dynamic paths provided by the user
prepare_similarity_tables(corpus_sbert_embeddings, factor_sbert_embeddings, factors_list_path, polarity_scores_path, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, output_path)


# %%
# Step 7: factor-corpus sentence similarity frequency based on TF-IDF embeddings of Step 5
# This step counts the number of corpus sentences that are semantically similar to factors by factor, polarity score, and semantic similarity ratio.
# CORPUS-LEVEL FREQUENCIES, TF-IDF-BASED,
# with sparse matrices instead of dense matrices and de-parallelized instead of parallelized, which caused excess RAM usage (154 Gb vs. 4.4 Gb)

# neuTres was lowered to 0.1 to count more sentences, and simRatio line 123 became similarity_ratios=np.arange(0.1, 0.5, 0.1)


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack
import os
from tqdm import tqdm

# Categorization of polarity scores based on user-defined thresholds
def categorize_polarity(score, upper_threshold, lower_threshold):
    if score > upper_threshold:
        return 'positive'
    elif score < lower_threshold:
        return 'negative'
    else:
        return 'neutral'

def prepare_similarity_tables_tfidf(embeddings_corpus_tfidf, embeddings_factors_tfidf, factors_list_path, polarity_scores_path, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, output_path, chunk_size=1000):
    factors_list_df = pd.read_excel(factors_list_path)

    if 'factor' not in factors_list_df.columns:
        print("'factor' column not found in the provided factors list Excel file. Please check the column name.")
        return

    # Load polarity scores DataFrame
    print("Loading polarity scores DataFrame")
    polarity_df = pd.read_excel(polarity_scores_path)

    # Handle "Error" values by replacing them with 0.000 and remove apostrophes
    polarity_df = polarity_df.replace("Error", 0.000)

    # Define the columns to modify
    columns_to_modify = ['SenVADER', 'SenTEXTBLOB', 'SenSTANZA', 'SenSTANZA-1', 'SenSTANZA_retrained', 'SenSTANZA_retrained-1', 'SenFLAIR']

    print("Before modification:\n", polarity_df.head())

    # Remove apostrophes and ensure float format
    for column in columns_to_modify:
        if column in polarity_df.columns:
            polarity_df[column] = polarity_df[column].astype(str).str.replace("'", "")
            polarity_df[column] = polarity_df[column].astype(float)
            polarity_df[column] = polarity_df[column].map(lambda x: f"{x:.3f}")

    print("After modification:\n", polarity_df.head())

    # Ensure integer columns remain as integers
    integer_columns = ['senNGlo', 'senNDoc']
    for column in integer_columns:
        if column in polarity_df.columns:
            polarity_df[column] = polarity_df[column].astype(int)

    # Save the modified polarity scores DataFrame with the "removed_errors" and the "tfidf" suffix
    polarity_scores_dir, polarity_scores_filename = os.path.split(polarity_scores_path)
    modified_polarity_scores_filename = polarity_scores_filename.replace(".xlsx", "_modified_errors_tfidf.xlsx")
    modified_polarity_scores_path = os.path.join(polarity_scores_dir, modified_polarity_scores_filename)

    print("Saving modified polarity scores DataFrame to:", modified_polarity_scores_path)
    try:
        polarity_df.to_excel(modified_polarity_scores_path, index=False)
        print("File saved successfully.")
    except Exception as e:
        print("Error saving file:", e)

    # Re-load the modified polarity scores DataFrame
    print("Re-loading modified polarity scores DataFrame")
    polarity_df = pd.read_excel(modified_polarity_scores_path)
    polarity_df.set_index('senNGlo', inplace=True)

    # Prepare DataFrame templates for positive, neutral, and negative tables
    columns = ['factor_id', 'factor_text'] + [f'ratio_{ratio:.1f}' for ratio in similarity_ratios]
    tables_tfidf = {category: pd.DataFrame(columns=columns) for category in ['positive', 'neutral', 'negative']}

    # Chunking
    num_factors = embeddings_factors_tfidf.shape[0]
    num_chunks = (num_factors + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_factors)

        chunk_factors = embeddings_factors_tfidf[start_idx:end_idx]
        chunk_cos_sim_matrix = cosine_similarity(embeddings_corpus_tfidf, chunk_factors)

        for factor_idx in range(start_idx, end_idx):
            factor_text = factors_list_df.at[factor_idx, 'factor']
            counts = {ratio: {'positive': 0, 'neutral': 0, 'negative': 0} for ratio in similarity_ratios}

            for corpus_idx, sim_score in enumerate(chunk_cos_sim_matrix[:, factor_idx - start_idx]):
                polarity_score = polarity_df.at[corpus_idx, selected_polarity_scores_header]
                corpus_sen_polarity = categorize_polarity(polarity_score, upper_threshold, lower_threshold)
                for ratio in similarity_ratios[::-1]:
                    if sim_score >= ratio:
                        counts[ratio][corpus_sen_polarity] += 1
                        break

            for category in ['positive', 'neutral', 'negative']:
                row = [factor_idx, factor_text] + [counts[ratio][category] for ratio in similarity_ratios]
                new_row = pd.DataFrame([row], columns=tables_tfidf[category].columns)
                tables_tfidf[category] = pd.concat([tables_tfidf[category], new_row], ignore_index=True)

    output_excel_path_tfidf = os.path.expanduser(output_path)
    with pd.ExcelWriter(output_excel_path_tfidf) as writer:
        for category, df in tables_tfidf.items():
            df.to_excel(writer, sheet_name=category, index=False)

    print(f"TF-IDF-based similarity tables saved to '{output_excel_path_tfidf}'.")

# Dynamic inputs for embeddings and polarity scores
corpus_tfidf_embeddings_path = input("\nPlease write the complete path to the corpus TF-IDF embeddings .npz file:\n")
factor_tfidf_embeddings_path = input("\nPlease write the complete path to the factor TF-IDF embeddings .npz file:\n")
factors_list_path = input("\nPlease write the path to the factor list Excel file, including the complete file name:\n")
polarity_scores_path = input("\nPlease write the path to the polarity score data-frame, including the complete file name:\n")
selected_polarity_scores_header = input("\nPlease write the header of the selected polarity scores to use in the data-frame:\n")
upper_threshold = float(input("\nEnter the upper threshold for positive polarity scores: write a float (e.g. 0.6)\n"))
lower_threshold = float(input("\nEnter the lower threshold for negative polarity scores: write a float (e.g. -0.6)\n"))
similarity_ratios = np.arange(0.1, 0.5, 0.1)
output_path = input("\nPlease write the complete path where the similarity tables will be saved, including the file name:\n")

# Ensure paths are expanded correctly to handle both home directory and external drives
corpus_tfidf_embeddings_path = os.path.expanduser(corpus_tfidf_embeddings_path)
factor_tfidf_embeddings_path = os.path.expanduser(factor_tfidf_embeddings_path)
factors_list_path = os.path.expanduser(factors_list_path)
polarity_scores_path = os.path.expanduser(polarity_scores_path)
output_path = os.path.expanduser(output_path)

# Load TF-IDF embeddings
corpus_tfidf_embeddings = load_npz(corpus_tfidf_embeddings_path)  # Keep as sparse matrix
factor_tfidf_embeddings = load_npz(factor_tfidf_embeddings_path)  # Keep as sparse matrix

# Execute the function with TF-IDF embeddings
prepare_similarity_tables_tfidf(corpus_tfidf_embeddings, factor_tfidf_embeddings, factors_list_path, polarity_scores_path, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, output_path)


# %%
# Step 8 sBERT: from txt evaluation sentences and embeddings to a weights data-frame with the frequencies (Wldkj)
# of positive and negative polarity score sentences semantically similar to partnership factors. 
# Version 10 parallelizes calculations on all available CPUs (can be adjusted to the number of available CPUs) to reduce calculation time. 

# This version filters non-rated projects but is NOT Jupyter Lab's version "step8-11.31-experiment-rated-projects-only.ipynb", which adds
# filtering devices.

# Updated to start weights from w_0_+ and w_0_- according to factor re-numbering.
# added logging at the beginning to accelerate input repetition in case of reiteration.
# After this cell there is a "Step 8 TF-IDF" cell to focus on explainable algorithms.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import os
from joblib import Parallel, delayed, parallel_backend
import psutil
import pickle
import signal
import re
import logging
from sentence_transformers import SentenceTransformer

# Suppress Huggingface tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logging
logging.basicConfig(filename='Step-8-log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# Function to save intermediate results
def save_intermediate_results(results_dfs, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(results_dfs, f)
    print(f"Intermediate results saved to {save_path}")

# Function to load intermediate results
def load_intermediate_results(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    return None

# Function to calculate cosine similarities between two sets of embeddings
def calculate_cosine_similarities(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2)

# Function to categorize polarity scores
def categorize_polarity(score, upper_threshold, lower_threshold):
    if score > upper_threshold:
        return 'positive'
    elif score < lower_threshold:
        return 'negative'
    return None

# Improved function to retrieve country code and project year from the original PDF file names
def retrieve_couCode_proYear_from_pdf(pdf_dir, proCode):
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf') and f'_{proCode}_' in filename:
            match = re.search(r'_([A-Z]{3})_(\d{4})', filename)
            if match:
                couCode = match.group(1)
                proYear = match.group(2)
                return couCode, proYear
    return None, None

# Function to read all TXT files for a project, return a list of sentences and their details
def read_project_sentences_and_details(directory_path, project_code):
    sentences = []
    sentence_details = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt') and f'_{project_code}_' in filename:
            parts = filename.replace('.txt', '').split('_')
            if len(parts) < 5:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            orgAcro, docType, proCode, senNGlo, senNdoc = parts
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        sentences.append(line)
                        sentence_details.append({
                            "orgAcro": orgAcro,
                            "docType": docType,
                            "proCode": proCode,
                            "senNGlo": senNGlo,
                            "senNdoc": senNdoc,
                            "sentence": line
                        })

    return sentences, sentence_details

# Function to process a single project
def process_project(project_code, projects_info_df, factors_list_df, polarity_df, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, factor_embeddings, factor_id_column, factor_text_column, pdf_dir, sbert_model, corpus_sentences_dir):
    project_data = projects_info_df[projects_info_df['proCode'] == project_code]
    orgAcro = project_data['orgAcro'].iloc[0]
    coFin = project_data['coFin'].iloc[0]
    proRate = project_data['proRate'].iloc[0]
    couCode, proYear = retrieve_couCode_proYear_from_pdf(pdf_dir, project_code)  # Retrieve country code and project year from PDF file names

    project_results = {ratio: {'orgAcro': orgAcro, 'proCode': project_code, 'coFin': coFin, 'proRate': proRate, 'couCode': couCode, 'proYear': proYear} for ratio in similarity_ratios}

    # Read sentences and embeddings for the current project
    project_sentences, _ = read_project_sentences_and_details(corpus_sentences_dir, project_code)
    if not project_sentences:
        print(f"No sentences found for project {project_code}")
        return project_results

    project_embeddings = sbert_model.encode(project_sentences, show_progress_bar=False)
    cos_sim_matrix = calculate_cosine_similarities(project_embeddings, factor_embeddings)

    for ratio in similarity_ratios:
        counts = {f'w_{i}_+': 0 for i in range(len(factors_list_df))}
        counts.update({f'w_{i}_-': 0 for i in range(len(factors_list_df))})
        for i in range(len(factor_embeddings)):
            factor_id = factors_list_df.at[i, factor_id_column]
            w_plus_key = f'w_{factor_id}_+'
            w_minus_key = f'w_{factor_id}_-'
            for j, sim_score in enumerate(cos_sim_matrix[:, i]):
                corpus_sen_polarity = categorize_polarity(polarity_df.iloc[j][selected_polarity_scores_header], upper_threshold, lower_threshold)
                if corpus_sen_polarity and sim_score >= ratio:
                    if corpus_sen_polarity == 'positive':
                        counts[w_plus_key] += 1
                    elif corpus_sen_polarity == 'negative':
                        counts[w_minus_key] += 1
        project_results[ratio].update(counts)

    return project_results

# Function to prepare similarity tables with periodic checkpoints and interruption handling
def prepare_similarity_tables(factors_list_path, factor_text_column, polarity_scores_path, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, projects_info_path, factor_id_column, pdf_dir, corpus_sentences_dir, save_path, checkpoint_interval=10, n_jobs=4):
    # Expand user paths to handle tilde and other relative paths
    factors_list_path = os.path.expanduser(factors_list_path)
    polarity_scores_path = os.path.expanduser(polarity_scores_path)
    projects_info_path = os.path.expanduser(projects_info_path)
    save_path = os.path.expanduser(save_path)
    pdf_dir = os.path.expanduser(pdf_dir)
    corpus_sentences_dir = os.path.expanduser(corpus_sentences_dir)

    factors_list_df = pd.read_excel(factors_list_path)
    polarity_df = pd.read_excel(polarity_scores_path)
    projects_info_df = pd.read_excel(projects_info_path)

    # Verify 'proRate' column exists
    if 'proRate' not in projects_info_df.columns:
        raise KeyError("'proRate' column not found in the projects_info_df DataFrame.")

    projects_info_df = projects_info_df[projects_info_df['proRate'].notna() & projects_info_df['proRate'] != '']
    polarity_df.set_index('senNGlo', inplace=True)

    # Check if the necessary columns exist in the DataFrame
    required_columns = ['orgAcro', 'proCode', 'coFin', 'proRate']
    missing_columns = [col for col in required_columns if col not in projects_info_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in projects_info_df: {missing_columns}")

    # Load intermediate results if they exist
    results_dfs = load_intermediate_results(save_path) or {ratio: pd.DataFrame(columns=['orgAcro', 'proCode', 'coFin', 'proRate', 'couCode', 'proYear'] + sum([[f'w_{i}_+', f'w_{i}_-'] for i in range(len(factors_list_df))], [])) for ratio in similarity_ratios}

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    factor_embeddings = sbert_model.encode(factors_list_df[factor_text_column].tolist(), show_progress_bar=True)

    processed_projects = set(results_dfs[similarity_ratios[0]]['proCode'].unique())
    projects_to_process = [project for project in projects_info_df['proCode'].unique() if project not in processed_projects]

    if not projects_to_process:
        print("All projects have already been processed.")
        return results_dfs

    print(f"Processing {len(projects_to_process)} projects out of {len(projects_info_df['proCode'].unique())}...")

    def save_and_exit(signum, frame):
        save_intermediate_results(results_dfs, save_path)
        print("Process interrupted and results saved. Exiting.")
        exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    # Parallel processing of projects
    with parallel_backend('loky'):
        project_results = Parallel(n_jobs=n_jobs)(
            delayed(process_project)(
                project_code, projects_info_df, factors_list_df, polarity_df, similarity_ratios,
                upper_threshold, lower_threshold, selected_polarity_scores_header, factor_embeddings,
                factor_id_column, factor_text_column, pdf_dir, sbert_model, corpus_sentences_dir
            ) for project_code in tqdm(projects_to_process, desc='Processing projects')
        )

    # Consolidate results from parallel processing
    for project_result in project_results:
        for ratio in similarity_ratios:
            results_dfs[ratio] = pd.concat([results_dfs[ratio], pd.DataFrame([project_result[ratio]])], ignore_index=True)

        if len(project_results) % checkpoint_interval == 0:
            save_intermediate_results(results_dfs, save_path)

    save_intermediate_results(results_dfs, save_path)

    return results_dfs

if __name__ == "__main__":
    corpus_embeddings_path = input("For CORPUS: Please specify the CORPUS NPY FILE path (e.g., ~/evaluation-sentiment-analysis/embeddings-15words-with-negative-stopwords/corpus_sbert_embeddings.npy): ")
    factor_embeddings_path = input("For FACTORS: Please specify the FACTOR NPY FILE path (e.g., ~/evaluation-sentiment-analysis/embeddings-15words-with-negative-stopwords/factor_sbert_embeddings.npy): ")
    factors_list_path = input("Path to the factor list Excel file, including the complete file name: ")
    factor_text_column = input("Column name for the factor text in the factor list file: ")
    polarity_scores_path = input("Path to the polarity score data-frame, including the complete file name: ")
    projects_info_path = input("Path to the projects ratings Excel file, including the complete file name (e.g. unified-project-ratings-codes-13-05-2024.xlsx): ")
    selected_polarity_scores_header = input("Header of the selected polarity scores to use in the data-frame: ")
    factor_id_column = input("Column name for factor ID in the factor list file: ")
    upper_threshold = float(input("Upper threshold for positive polarity scores (e.g., 0.6): "))
    lower_threshold = float(input("Lower threshold for negative polarity scores (e.g., -0.6): "))
    similarity_ratios = np.arange(0.3, 0.9, 0.1)
    save_path = input("Enter the file path for saving intermediate results (e.g., 'intermediate_results.pkl'): ")
    checkpoint_interval = int(input("Enter the number of projects to process before saving a checkpoint (e.g., 10): "))
    pdf_dir = input("Please write the path to the directory with the original PDF files from which the script will take the country codes and project years:\n")
    corpus_sentences_dir = input("Enter the path to the directory containing the corpus sentences:\n")

    # Initialize logging right after inputs are collected
    logging.info(f"Corpus embeddings path: {corpus_embeddings_path}")
    logging.info(f"Factor embeddings path: {factor_embeddings_path}")
    logging.info(f"Factors list path: {factors_list_path}")
    logging.info(f"Factor text column: {factor_text_column}")
    logging.info(f"Polarity scores path: {polarity_scores_path}")
    logging.info(f"Projects info path: {projects_info_path}")
    logging.info(f"Selected polarity scores header: {selected_polarity_scores_header}")
    logging.info(f"Factor ID column: {factor_id_column}")
    logging.info(f"Upper threshold: {upper_threshold}")
    logging.info(f"Lower threshold: {lower_threshold}")
    logging.info(f"Similarity ratios: {similarity_ratios}")
    logging.info(f"Save path: {save_path}")
    logging.info(f"Checkpoint interval: {checkpoint_interval}")
    logging.info(f"PDF directory: {pdf_dir}")
    logging.info(f"Corpus sentences directory: {corpus_sentences_dir}")

    corpus_embeddings_path = os.path.expanduser(corpus_embeddings_path)
    factor_embeddings_path = os.path.expanduser(factor_embeddings_path)
    factors_list_path = os.path.expanduser(factors_list_path)
    polarity_scores_path = os.path.expanduser(polarity_scores_path)
    projects_info_path = os.path.expanduser(projects_info_path)
    save_path = os.path.expanduser(save_path)
    pdf_dir = os.path.expanduser(pdf_dir)
    corpus_sentences_dir = os.path.expanduser(corpus_sentences_dir)

    n_jobs = psutil.cpu_count(logical=True)
    results_dfs = prepare_similarity_tables(factors_list_path, factor_text_column, polarity_scores_path, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, projects_info_path, factor_id_column, pdf_dir, corpus_sentences_dir, save_path, checkpoint_interval, n_jobs=n_jobs)

    output_directory = input("\nPlease write the directory path where you want to save the final output DataFrame:\n")
    output_filename = input("\nPlease write the filename for the final output DataFrame (e.g.'factor-sentence-count.xlsx'):\n")

    output_directory = os.path.expanduser(output_directory)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    file_path = os.path.join(output_directory, output_filename)

    factors_list_df = pd.read_excel(factors_list_path)
    with pd.ExcelWriter(file_path) as writer:
        for ratio, df in results_dfs.items():
            sheet_name = f'Ratio_{ratio:.1f}'
            sorted_columns = ['orgAcro', 'proCode', 'coFin', 'proRate', 'couCode', 'proYear'] + sum([[f'w_{i}_+', f'w_{i}_-'] for i in range(len(factors_list_df))], [])
            df = df[sorted_columns]  # Ensure columns are in the correct order
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Final data saved to '{file_path}' with one sheet per similarity ratio.")




# %%
# Step 8 TF-IDF: from txt evaluation sentences and embeddings to a weights data-frame with the frequencies (Wldkj)
# of positive and negative polarity score sentences semantically similar to partnership factors. 
# Version 10 parallelizes calculations on all available CPUs (can be adjusted to the number of available CPUs) to reduce calculation time. 

# This version filters non-rated projects but is NOT Jupyter Lab's version "step8-11.31-experiment-rated-projects-only.ipynb", which adds
# filtering devices.

# Updated to start weights from w_0_+ and w_0_- according to factor re-numbering.
# Added logging at the beginning to accelerate input repetition in case of reiteration.
# Takes tdfidf_vectorizer.pkl prepared and unified project (former corpus) + factor TF-IDF .npz dictionary from modified Step 5
# Before this cell, there is a "Step 8 sBERT" cell to focus on transformer algorithms.

# neuTres was lowered to 0.1 to count more sentences: line 226 became similarity_ratios=np.arange(0.1, 0.5, 0.1)

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import os
from joblib import Parallel, delayed, parallel_backend
import psutil
import pickle
import signal
import re
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

# Suppress warnings if needed
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to save intermediate results
def save_intermediate_results(results_dfs, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(results_dfs, f)
    print(f"Intermediate results saved to {save_path}")

# Function to load intermediate results
def load_intermediate_results(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    return None

# Function to calculate cosine similarities between two sets of embeddings
def calculate_cosine_similarities(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2)

# Function to categorize polarity scores
def categorize_polarity(score, upper_threshold, lower_threshold):
    if score > upper_threshold:
        return 'positive'
    elif score < lower_threshold:
        return 'negative'
    return None

# Improved function to retrieve country code and project year from the original PDF file names
def retrieve_couCode_proYear_from_pdf(pdf_dir, proCode):
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf') and f'_{proCode}_' in filename:
            match = re.search(r'_([A-Z]{3})_(\d{4})', filename)
            if match:
                couCode = match.group(1)
                proYear = match.group(2)
                return couCode, proYear
    return None, None

# Function to read all TXT files for a project, return a list of sentences and their details
def read_project_sentences_and_details(directory_path, project_code):
    sentences = []
    sentence_details = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt') and f'_{project_code}_' in filename:
            parts = filename.replace('.txt', '').split('_')
            if len(parts) < 5:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            orgAcro, docType, proCode, senNGlo, senNdoc = parts
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        sentences.append(line)
                        sentence_details.append({
                            "orgAcro": orgAcro,
                            "docType": docType,
                            "proCode": proCode,
                            "senNGlo": senNGlo,
                            "senNdoc": senNdoc,
                            "sentence": line
                        })

    return sentences, sentence_details

# Function to process a single project
def process_project(project_code, projects_info_df, factors_list_df, polarity_df, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, factor_embeddings, factor_id_column, factor_text_column, pdf_dir, tfidf_vectorizer, corpus_sentences_dir):
    project_data = projects_info_df[projects_info_df['proCode'] == project_code]
    orgAcro = project_data['orgAcro'].iloc[0]
    coFin = project_data['coFin'].iloc[0]
    proRate = project_data['proRate'].iloc[0]
    couCode, proYear = retrieve_couCode_proYear_from_pdf(pdf_dir, project_code)  # Retrieve country code and project year from PDF file names

    project_results = {ratio: {'orgAcro': orgAcro, 'proCode': project_code, 'coFin': coFin, 'proRate': proRate, 'couCode': couCode, 'proYear': proYear} for ratio in similarity_ratios}

    # Read sentences and embeddings for the current project
    project_sentences, _ = read_project_sentences_and_details(corpus_sentences_dir, project_code)
    if not project_sentences:
        print(f"No sentences found for project {project_code}")
        return project_results

    project_embeddings = tfidf_vectorizer.transform(project_sentences)
    cos_sim_matrix = calculate_cosine_similarities(project_embeddings, factor_embeddings)

    for ratio in similarity_ratios:
        counts = {f'w_{i}_+': 0 for i in range(factor_embeddings.shape[0])}
        counts.update({f'w_{i}_-': 0 for i in range(factor_embeddings.shape[0])})
        for i in range(factor_embeddings.shape[0]):
            factor_id = factors_list_df.at[i, factor_id_column]
            w_plus_key = f'w_{factor_id}_+'
            w_minus_key = f'w_{factor_id}_-'
            for j, sim_score in enumerate(cos_sim_matrix[:, i]):
                corpus_sen_polarity = categorize_polarity(polarity_df.iloc[j][selected_polarity_scores_header], upper_threshold, lower_threshold)
                if corpus_sen_polarity and sim_score >= ratio:
                    if corpus_sen_polarity == 'positive':
                        counts[w_plus_key] += 1
                    elif corpus_sen_polarity == 'negative':
                        counts[w_minus_key] += 1
        project_results[ratio].update(counts)

    return project_results


# Function to prepare similarity tables with periodic checkpoints and interruption handling
def prepare_similarity_tables(factors_list_path, factor_text_column, polarity_scores_path, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, projects_info_path, factor_id_column, pdf_dir, corpus_sentences_dir, save_path, checkpoint_interval=10, n_jobs=4):
    # Expand user paths to handle tilde and other relative paths
    factors_list_path = os.path.expanduser(factors_list_path)
    polarity_scores_path = os.path.expanduser(polarity_scores_path)
    projects_info_path = os.path.expanduser(projects_info_path)
    save_path = os.path.expanduser(save_path)
    pdf_dir = os.path.expanduser(pdf_dir)
    corpus_sentences_dir = os.path.expanduser(corpus_sentences_dir)

    factors_list_df = pd.read_excel(factors_list_path)
    polarity_df = pd.read_excel(polarity_scores_path)
    projects_info_df = pd.read_excel(projects_info_path)

    # Verify 'proRate' column exists
    if 'proRate' not in projects_info_df.columns:
        raise KeyError("'proRate' column not found in the projects_info_df DataFrame.")

    projects_info_df = projects_info_df[projects_info_df['proRate'].notna() & projects_info_df['proRate'] != '']
    polarity_df.set_index('senNGlo', inplace=True)

    # Check if the necessary columns exist in the DataFrame
    required_columns = ['orgAcro', 'proCode', 'coFin', 'proRate']
    missing_columns = [col for col in required_columns if col not in projects_info_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in projects_info_df: {missing_columns}")

    # Load intermediate results if they exist
    results_dfs = load_intermediate_results(save_path) or {ratio: pd.DataFrame(columns=['orgAcro', 'proCode', 'coFin', 'proRate', 'couCode', 'proYear'] + sum([[f'w_{i}_+', f'w_{i}_-'] for i in range(len(factors_list_df))], [])) for ratio in similarity_ratios}

    # Load TF-IDF embeddings
    factor_embeddings = load_npz(factor_embeddings_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)

    processed_projects = set(results_dfs[similarity_ratios[0]]['proCode'].unique())
    projects_to_process = [project for project in projects_info_df['proCode'].unique() if project not in processed_projects]

    if not projects_to_process:
        print("All projects have already been processed.")
        return results_dfs

    print(f"Processing {len(projects_to_process)} projects out of {len(projects_info_df['proCode'].unique())}...")

    def save_and_exit(signum, frame):
        save_intermediate_results(results_dfs, save_path)
        print("Process interrupted and results saved. Exiting.")
        exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    # Parallel processing of projects
    with parallel_backend('loky'):
        project_results = Parallel(n_jobs=n_jobs)(
            delayed(process_project)(
                project_code, projects_info_df, factors_list_df, polarity_df, similarity_ratios,
                upper_threshold, lower_threshold, selected_polarity_scores_header, factor_embeddings,
                factor_id_column, factor_text_column, pdf_dir, tfidf_vectorizer, corpus_sentences_dir
            ) for project_code in tqdm(projects_to_process, desc='Processing projects')
        )

    # Consolidate results from parallel processing
    for project_result in project_results:
        for ratio in similarity_ratios:
            results_dfs[ratio] = pd.concat([results_dfs[ratio], pd.DataFrame([project_result[ratio]])], ignore_index=True)

        if len(project_results) % checkpoint_interval == 0:
            save_intermediate_results(results_dfs, save_path)

    save_intermediate_results(results_dfs, save_path)

    return results_dfs

if __name__ == "__main__":
    # Add logging to save the inputs for easier repetition
    logging.basicConfig(filename='Step-8-log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

    corpus_embeddings_path = input("Please write the complete path to the corpus TF-IDF embeddings .npz file: ")
    factor_embeddings_path = input("Please write the complete path to the factor TF-IDF embeddings .npz file: ")
    vectorizer_path = input("Please write the path to the Step 5 fitted TfidfVectorizer (e.g., '~/evaluation-sentiment-analysis/embeddings-10words-with-negative-stopwords/tfidf_vectorizer.pkl'): ")
    factors_list_path = input("Path to the factor list Excel file, including the complete file name: ")
    factor_text_column = input("Column name for the factor text in the factor list file: ")
    polarity_scores_path = input("Path to the Step 7 MODIFIED_ERROR polarity score data-frame, including the complete file name: ")
    projects_info_path = input("Path to the projects ratings Excel file, including the complete file name (e.g. unified-project-ratings-codes-13-05-2024.xlsx): ")
    selected_polarity_scores_header = input("Header of the selected polarity scores to use in the data-frame: ")
    factor_id_column = input("Column name for factor ID in the factor list file: ")
    upper_threshold = float(input("Upper threshold for positive polarity scores (e.g., 0.6): "))
    lower_threshold = float(input("Lower threshold for negative polarity scores (e.g., -0.6): "))
    similarity_ratios = np.arange(0.1, 0.5, 0.1)
    save_path = input("Enter the file path for saving intermediate results (e.g., 'intermediate_results.pkl'): ")
    checkpoint_interval = int(input("Enter the number of projects to process before saving a checkpoint (e.g., 10): "))
    pdf_dir = input("Please write the path to the directory with the original PDF files from which the script will take the country codes and project years:\n")
    corpus_sentences_dir = input("Enter the path to the directory containing the corpus sentences:\n")

    # Log inputs for easier repetition
    logging.info(f"Corpus TF-IDF embeddings path: {corpus_embeddings_path}")
    logging.info(f"Factor TF-IDF embeddings path: {factor_embeddings_path}")
    logging.info(f"Vectorizer path: {vectorizer_path}")
    logging.info(f"Factors list path: {factors_list_path}")
    logging.info(f"Factor text column: {factor_text_column}")
    logging.info(f"Polarity scores path: {polarity_scores_path}")
    logging.info(f"Projects info path: {projects_info_path}")
    logging.info(f"Selected polarity scores header: {selected_polarity_scores_header}")
    logging.info(f"Factor ID column: {factor_id_column}")
    logging.info(f"Upper threshold: {upper_threshold}")
    logging.info(f"Lower threshold: {lower_threshold}")
    logging.info(f"Similarity ratios: {similarity_ratios}")
    logging.info(f"Save path: {save_path}")
    logging.info(f"Checkpoint interval: {checkpoint_interval}")
    logging.info(f"PDF directory: {pdf_dir}")
    logging.info(f"Corpus sentences directory: {corpus_sentences_dir}")

    corpus_embeddings_path = os.path.expanduser(corpus_embeddings_path)
    factor_embeddings_path = os.path.expanduser(factor_embeddings_path)
    vectorizer_path = os.path.expanduser(vectorizer_path)
    factors_list_path = os.path.expanduser(factors_list_path)
    polarity_scores_path = os.path.expanduser(polarity_scores_path)
    projects_info_path = os.path.expanduser(projects_info_path)
    save_path = os.path.expanduser(save_path)
    pdf_dir = os.path.expanduser(pdf_dir)
    corpus_sentences_dir = os.path.expanduser(corpus_sentences_dir)

    n_jobs = psutil.cpu_count(logical=True)
    results_dfs = prepare_similarity_tables(factors_list_path, factor_text_column, polarity_scores_path, similarity_ratios, upper_threshold, lower_threshold, selected_polarity_scores_header, projects_info_path, factor_id_column, pdf_dir, corpus_sentences_dir, save_path, checkpoint_interval, n_jobs=n_jobs)

    # Add logging here to confirm reaching this point
    logging.info("Finished preparing similarity tables, now prompting for output directory and filename.")

    # Prompt the user for the output directory
    output_directory = input(
        "\nPlease write the directory path where you want to save the final output DataFrame.\n"
        "Make sure to end the path with a forward slash '/' (e.g., '~/evaluation-sentiment-analysis/sim-38/weights-standardization/'):\n"
    )

    # Check for empty output directory
    if not output_directory.strip():
        raise ValueError("The output directory path cannot be empty. Please provide a valid directory path.")

    # Ensure the output directory path ends with a slash
    if not output_directory.endswith('/'):
        output_directory += '/'
        print("Note: Added '/' to the end of the directory path to ensure it's a valid directory.")

    # Expand the directory path to handle user home (~) or other relative paths
    output_directory = os.path.expanduser(output_directory)

    # Ensure the directory exists or create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Prompt the user for the output filename
    output_filename = input(
        "\nPlease write the filename for the final output DataFrame (e.g.'sim-38-step8tfidf-weights.xlsx'):\n"
    )

    # Validate the filename to ensure it doesn't contain path separators
    if '/' in output_filename or '\\' in output_filename:
        raise ValueError("The output filename should not contain any path separators (like '/' or '\\'). Please provide just the filename.")

    # Construct the full path for the file
    file_path = os.path.join(output_directory, output_filename)

    # Add logging here to confirm the directory and filename inputs
    logging.info(f"Output directory: {output_directory}")
    logging.info(f"Output filename: {output_filename}")
    logging.info(f"Full file path: {file_path}")

    # Load the factors list
    factors_list_df = pd.read_excel(factors_list_path)

    # Save the results to an Excel file
    with pd.ExcelWriter(file_path) as writer:
        for ratio, df in results_dfs.items():
            sheet_name = f'Ratio_{ratio:.1f}'
            sorted_columns = ['orgAcro', 'proCode', 'coFin', 'proRate', 'couCode', 'proYear'] + sum([[f'w_{i}_+', f'w_{i}_-'] for i in range(len(factors_list_df))], [])
            df = df[sorted_columns]  # Ensure columns are in the correct order
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logging.info(f"Final data saved to '{file_path}' with one sheet per similarity ratio.")
    print(f"Final data saved to '{file_path}' with one sheet per similarity ratio.")



# %%
# Step 9 LOOCV: from corpus data-set to optimized and trained RF, XGB, NN, LR models and pre-processed data-frame

# ULTRA-LONG: 3 days on whole corpus and 753 factors, 1.5 hours with whole corpus and 24 factors

# LOOCV version for a very small sample data-frame resulting from # Step 8.
# LOOCV stands for Leave-One-Out Cross-Validation.
# A separate version with train/test/validate data split is needed for the complete # Step 8 dataframe.
# This Step produces: Data Pre-processing, Model Training & Evaluation, and Hyperparameter Optimization;
# WARNINGS:
#       due to class unbalance were mostly blocked to keep the output clean - class balance depends on 
#       factor sentence formulation.
#       removed standard deviation (std dev) calculation from Optuna as it caused failures with mean + std dev
# MISSING VALUES: 
#       Missing values are due to rated projects without evaluation documents in the corpus or txt sentences.
#       Missing values would cause NaN data, jeopardizing the model trials and stopping the script.
#       The function "remove_missing_data" was added to remove rows with missing values in specified columns.

# Modifications introduced on the following dates (the hardest problem to solve has been the continuous mismatch between the features in Step 9 LOOCV's machine learning argument and Step 10's SHAP argument).
#       Jul. 26th, 2024: data normalization, instead of standardization, to allow for processing of ex-ante documents (e.g. feasibility studies, draft MoU, press releases).
#       Merges country (couCode) and year (proYear) into one sole feature with Feature Interaction, though not normalized, to give contextualization.
#       Simplifies the number of output files skipping the former "clean" version.
#	    Jul. 27th, 2024: Simplified X_processed and X_processed_machine_learning into the sole X_processed_machine_learning.
#	    Jul. 29th, 2024: since the Jul. 27 2024 script did not drop 'orgAcro', 'proCode', 'couCode', 'proYear', it dropped the last 'w_i_+' and the last two 'w_i_-' features, and it did not retain the combined feature 'couCode-proYear', I made different modifications.


import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from tqdm.notebook import tqdm
import os
import joblib
from joblib import Parallel, delayed
import logging

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, precision, recall, f1, mcc

# Function to optimize Logistic Regression using Optuna
def objective_lr(trial, X_scaled, y):
    param = {
        'C': trial.suggest_loguniform('C', 1e-5, 1e2),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
        'max_iter': 1000
    }
    loo = LeaveOneOut()
    metrics_list = Parallel(n_jobs=-1)(delayed(_train_and_evaluate_lr)(param, X_scaled, y, train_index, test_index)
                                       for train_index, test_index in loo.split(X_scaled))
    mean_score = np.mean(metrics_list)
    return mean_score

def _train_and_evaluate_lr(param, X_scaled, y, train_index, test_index):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if len(np.unique(y_train)) < 2:
        return 0
    lr = LogisticRegression(**param, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return evaluate_model(y_test, y_pred)[0]

# Function to optimize RandomForest using Optuna
def objective_rf(trial, X_scaled, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }
    loo = LeaveOneOut()
    metrics_list = Parallel(n_jobs=-1)(delayed(_train_and_evaluate_rf)(param, X_scaled, y, train_index, test_index)
                                       for train_index, test_index in loo.split(X_scaled))
    mean_score = np.mean(metrics_list)
    return mean_score

def _train_and_evaluate_rf(param, X_scaled, y, train_index, test_index):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if len(np.unique(y_train)) < 2:
        return 0
    rf = RandomForestClassifier(**param, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return evaluate_model(y_test, y_pred)[0]

# Function to optimize XGBoost using Optuna
def objective_xgb(trial, X_scaled, y):
    param = {
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'eta': trial.suggest_float('eta', 0.01, 1.0),
    }
    loo = LeaveOneOut()
    metrics_list = Parallel(n_jobs=-1)(delayed(_train_and_evaluate_xgb)(param, X_scaled, y, train_index, test_index)
                                       for train_index, test_index in loo.split(X_scaled))
    mean_score = np.mean(metrics_list)
    return mean_score

def _train_and_evaluate_xgb(param, X_scaled, y, train_index, test_index):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if len(np.unique(y_train)) < 2:
        return 0
    xgb = XGBClassifier(**param, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    return evaluate_model(y_test, y_pred)[0]

# Function to optimize Neural Network using Optuna
def objective_nn(trial, X_scaled, y):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = [trial.suggest_int(f'n_units_l{i}', 4, 128) for i in range(n_layers)]
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
    alpha = trial.suggest_float('alpha', 1e-8, 1e-1, log=True)
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
    loo = LeaveOneOut()
    metrics_list = Parallel(n_jobs=-1)(delayed(_train_and_evaluate_nn)(layers, activation, solver, alpha, learning_rate, X_scaled, y, train_index, test_index)
                                       for train_index, test_index in loo.split(X_scaled))
    mean_score = np.mean(metrics_list)
    return mean_score

def _train_and_evaluate_nn(layers, activation, solver, alpha, learning_rate, X_scaled, y, train_index, test_index):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if len(np.unique(y_train)) < 2:
        return 0
    nn = MLPClassifier(hidden_layer_sizes=tuple(layers), activation=activation, solver=solver, alpha=alpha,
                       learning_rate=learning_rate, max_iter=500, random_state=42)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    return evaluate_model(y_test, y_pred)[0]

# Function to evaluate model with LOOCV
def evaluate_model_loocv(model, X, y):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_index, test_index in tqdm(loo.split(X), total=len(X), desc="LOOCV"):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if len(np.unique(y_train)) < 2:
            continue
        model.fit(X_train, y_train)
        y_pred.extend(model.predict(X_test))
        y_true.extend(y_test)
    return evaluate_model(y_true, y_pred)

# Function to evaluate Neural Network with LOOCV
def evaluate_neural_network_loocv(X, y, best_nn_params):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    layers = [best_nn_params[f'n_units_l{i}'] for i in range(best_nn_params['n_layers'])]
    for train_index, test_index in tqdm(loo.split(X), total=len(X), desc="NN LOOCV"):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if len(np.unique(y_train)) < 2:
            continue
        nn = MLPClassifier(hidden_layer_sizes=tuple(layers), activation=best_nn_params['activation'],
                           solver=best_nn_params['solver'], alpha=best_nn_params['alpha'],
                           learning_rate=best_nn_params['learning_rate'], max_iter=500, random_state=42)
        nn.fit(X_train, y_train)
        y_pred.extend(nn.predict(X_test))
        y_true.extend(y_test)
    return evaluate_model(y_true, y_pred)

# Function to remove rows with missing values in specific columns
def remove_missing_data(df):
    return df.dropna()

def log_inputs(log_file, input_file, sheet_name, output_file, preprocessed_output_file, model_save_dir, preprocessed_data_dir, non_numeric_mapping_file, min_max_file):
    with open(log_file, 'w') as f:
        f.write(f"Input File: {input_file}\n")
        f.write(f"Sheet Name: {sheet_name}\n")
        f.write(f"Output File: {output_file}\n")
        f.write(f"Preprocessed Output File: {preprocessed_output_file}\n")
        f.write(f"Model Save Directory: {model_save_dir}\n")
        f.write(f"Preprocessed Data Directory: {preprocessed_data_dir}\n")
        f.write(f"Non-numeric Mapping File: {non_numeric_mapping_file}\n")
        f.write(f"Min-Max File: {min_max_file}\n")

def main():
    # Dynamic input prompts
    input_file = input("Please enter the path to the input factor-weight data-frame Excel file: ")
    sheet_name = input("Please enter the sheet name: ")
    output_file = input("Please enter the path to the output optimized hyperparameter and evaluation file (e.g. hyperparameters_evaluation_i7_11.07.xlsx): ")
    preprocessed_output_file = input("Please enter the path to save the normalized data-frame XLSX file (e.g. sim-XX-step9-weights-normalized.xlsx): ")
    model_save_dir = input("Please write the directory path to save the trained RF, XGB, NN, and LR models (e.g. /Documenti/phd-trial-standardized-small/trained-models/rf-XGB-nn-lr): ")
    preprocessed_data_dir = input("Please write the directory path to save the pre-processed NPZ data (e.g. ~/Documenti/phd-trial-standardized-small/output/intermediate-results): ")
    non_numeric_mapping_file = input("Please enter the path to save the non-numeric feature mapping JSON file (e.g. ~/Documenti/phd-trial-standardized-small/non-numeric-feature-maps/non_numeric_mapping.json): ")

    # Dynamically ask for the path to save the min and max values
    min_max_file = input("\nPath to save the FILE with the MIN and MAX value of the weights to predict (e.g. ~/path-to/sim-XX-step9-min-max-weights.xlsx): ")
    min_max_file = os.path.expanduser(min_max_file)

    # Log the inputs
    log_file = "Step-9-log.txt"
    log_inputs(log_file, input_file, sheet_name, output_file, preprocessed_output_file, model_save_dir, preprocessed_data_dir, non_numeric_mapping_file, min_max_file)

    # Expand user paths
    input_file = os.path.expanduser(input_file)
    output_file = os.path.expanduser(output_file)
    preprocessed_output_file = os.path.expanduser(preprocessed_output_file)
    model_save_dir = os.path.expanduser(model_save_dir)
    preprocessed_data_dir = os.path.expanduser(preprocessed_data_dir)
    non_numeric_mapping_file = os.path.expanduser(non_numeric_mapping_file)

    # Ensure directories exist
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(non_numeric_mapping_file), exist_ok=True)

    # Load the dataset
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    # Remove rows with missing values
    df = remove_missing_data(df)

    # Define X and y
    X = df.drop('proRate', axis=1)
    y = df['proRate']

    # Map non-numeric columns to numeric values and save the mapping
    non_numeric_columns = X.select_dtypes(include=['object']).columns
    non_numeric_mapping = {}

    for col in non_numeric_columns:
        unique_values = X[col].unique()
        mapping = {val: i for i, val in enumerate(unique_values)}
        non_numeric_mapping[col] = mapping
        X[col] = X[col].map(mapping)


    # Create the combined feature for couCode and proYear
    X['couCode-proYear'] = X['couCode'].astype(str) + '-' + X['proYear'].astype(str)
    print("\nIntroducing the feature based on Feature Interaction between couCode and proYear: couCode-proYear")
    # Map the new combined feature to numeric values
    combined_mapping = {val: i for i, val in enumerate(X['couCode-proYear'].unique())}
    X['couCode-proYear'] = X['couCode-proYear'].map(combined_mapping)
    
    
    # Add combined_mapping to the non_numeric_mapping dictionary
    non_numeric_mapping['couCode-proYear'] = combined_mapping


    # Save the non-numeric mapping
    with open(non_numeric_mapping_file, 'w') as f:
        json.dump(non_numeric_mapping, f, indent=4)


    # Normalize numeric columns
    numeric_columns = X.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[numeric_columns])
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_columns)

    # Compute the min and max values for each column
    min_max_values = {
        "min": X[numeric_columns].min(),
        "max": X[numeric_columns].max()
    }

    # Save the min and max values to an Excel file
    min_max_df = pd.DataFrame(min_max_values)
    min_max_df.to_excel(min_max_file, index=True)

    # Use only scaled numeric columns to create X_processed_machine_learning
    X_processed_machine_learning = X_scaled_df.copy()

    # Retain only the required columns
    features_to_retain = ['coFin', 'couCode-proYear'] + [col for col in X_processed_machine_learning.columns if col.startswith('w_') and (col.endswith('_+') or col.endswith('_-'))]
    X_processed_machine_learning = X_processed_machine_learning[features_to_retain]

    # Ensure all necessary features are retained and unwanted features are dropped
    X_processed_machine_learning = X_processed_machine_learning.drop(['orgAcro', 'proCode', 'couCode', 'proYear'], axis=1, errors='ignore')

    # Save the updated processed data-frame to an Excel file
    X_processed_machine_learning.to_excel(preprocessed_output_file, index=False)

    # Ensure only numeric columns in X_processed_machine_learning
    X_processed_machine_learning = X_processed_machine_learning.apply(pd.to_numeric)

    # Print the features to be fed to the machine learning models
    print("\nThese are the features that will be fed to the machine learning models:\n")
    print(X_processed_machine_learning.columns)

    # Ask the user if they want to proceed with the machine learning models training
    proceed = input("\nDo you want to proceed to the machine learning models trainings? Write Yes or No: ")

    if proceed.lower() == 'yes':
        # Proceed to the training
        print("\nProceeding to the machine learning models training...\n")
    else:
        # Exit the script
        print("\nYou said you do not want to proceed with the machine learning training. Please restart the Step 9 LOOCV script with the correct inputs or relevant script modifications.")
        exit()

    # Verify dimensions
    feature_names = X_processed_machine_learning.columns.to_numpy()
    if X_processed_machine_learning.shape[1] != len(feature_names):
        logging.error(f"Mismatch between feature dimensions and X_processed_machine_learning: {X_processed_machine_learning.shape[1]} vs {len(feature_names)}")
        print(f"Mismatch between feature dimensions and X_processed_machine_learning: {X_processed_machine_learning.shape[1]} vs {len(feature_names)}")
    else:
        print("Feature dimensions match.")

    # Adjust class labels to start from 0
    y = y - 1


    # Print and verify the features
    print("\nThese are the features that will be fed to the machine learning models:\n")
    print(X_processed_machine_learning.columns)
    proceed = input("\nDo you want to proceed to the machine learning models training? Write Yes or No: ")
    if proceed.lower() != "yes":
        print("\nYou said you do not want to proceed with the machine learning training. Please restart the Step 9 LOOCV script with the correct inputs or relevant script modifications.")
        return

    # Initialize TQDM progress bar
    total_steps = 4 + 1 + 1 + 1  # 4 optimizations, 1 evaluations, 1 save file
    with tqdm(total=total_steps, desc="Overall Progress") as pbar:

        # Optimize Logistic Regression using Optuna
        study_lr = optuna.create_study(direction='maximize')
        study_lr.optimize(lambda trial: objective_lr(trial, X_processed_machine_learning.to_numpy(), y.to_numpy()), n_trials=30)
        best_lr_params = study_lr.best_params
        pbar.update(1)

        # Optimize RandomForest using Optuna
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(lambda trial: objective_rf(trial, X_processed_machine_learning.to_numpy(), y.to_numpy()), n_trials=30)
        best_rf_params = study_rf.best_params
        pbar.update(1)

        # Optimize XGBoost using Optuna
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(lambda trial: objective_xgb(trial, X_processed_machine_learning.to_numpy(), y.to_numpy()), n_trials=30)
        best_xgb_params = study_xgb.best_params
        pbar.update(1)

        # Optimize Neural Network using Optuna
        study_nn = optuna.create_study(direction='maximize')
        study_nn.optimize(lambda trial: objective_nn(trial, X_processed_machine_learning.to_numpy(), y.to_numpy()), n_trials=30)
        best_nn_params = study_nn.best_params
        pbar.update(1)

        # Evaluate models with LOOCV

        lr_model = LogisticRegression(**best_lr_params, random_state=42)
        lr_metrics = evaluate_model_loocv(lr_model, X_processed_machine_learning.to_numpy(), y.to_numpy())
        print(f"Logistic Regression LOOCV Metrics:\n Accuracy: {lr_metrics[0]}\n Precision: {lr_metrics[1]}\n Recall: {lr_metrics[2]}\n F1 Score: {lr_metrics[3]}\n MCC: {lr_metrics[4]}")
        pbar.update(1)

        rf_model = RandomForestClassifier(**best_rf_params, random_state=42)
        rf_metrics = evaluate_model_loocv(rf_model, X_processed_machine_learning.to_numpy(), y.to_numpy())
        print(f"Random Forest LOOCV Metrics:\n Accuracy: {rf_metrics[0]}\n Precision: {rf_metrics[1]}\n Recall: {rf_metrics[2]}\n F1 Score: {rf_metrics[3]}\n MCC: {rf_metrics[4]}")
        pbar.update(1)

        xgb_model = XGBClassifier(**best_xgb_params, random_state=42)
        xgb_metrics = evaluate_model_loocv(xgb_model, X_processed_machine_learning.to_numpy(), y.to_numpy())
        print(f"XGBoost LOOCV Metrics:\n Accuracy: {xgb_metrics[0]}\n Precision: {xgb_metrics[1]}\n Recall: {xgb_metrics[2]}\n F1 Score: {xgb_metrics[3]}\n MCC: {xgb_metrics[4]}")
        pbar.update(1)

        nn_model = MLPClassifier(hidden_layer_sizes=tuple([best_nn_params[f'n_units_l{i}'] for i in range(best_nn_params['n_layers'])]),
                                 activation=best_nn_params['activation'],
                                 solver=best_nn_params['solver'],
                                 alpha=best_nn_params['alpha'],
                                 learning_rate=best_nn_params['learning_rate'],
                                 max_iter=500,
                                 random_state=42)
        nn_model.fit(X_processed_machine_learning.to_numpy(), y.to_numpy())
        nn_metrics = evaluate_neural_network_loocv(X_processed_machine_learning.to_numpy(), y.to_numpy(), best_nn_params)
        print(f"Neural Network LOOCV Metrics:\n Accuracy: {nn_metrics[0]}\n Precision: {nn_metrics[1]}\n Recall: {nn_metrics[2]}\n F1 Score: {nn_metrics[3]}\n MCC: {nn_metrics[4]}")
        pbar.update(1)

        # Save hyperparameters and evaluation metrics
        with open(output_file, 'w') as f:
            f.write("Best Logistic Regression Hyperparameters:\n")
            f.write(str(best_lr_params) + "\n")
            f.write("Best Random Forest Hyperparameters:\n")
            f.write(str(best_rf_params) + "\n")
            f.write("Best XGBoost Hyperparameters:\n")
            f.write(str(best_xgb_params) + "\n")
            f.write("Best Neural Network Hyperparameters:\n")
            f.write(str(best_nn_params) + "\n")
            f.write("\nLogistic Regression LOOCV Metrics:\n")
            f.write(f"Accuracy: {lr_metrics[0]}\nPrecision: {lr_metrics[1]}\nRecall: {lr_metrics[2]}\nF1 Score: {lr_metrics[3]}\nMCC: {lr_metrics[4]}\n")
            f.write("\nRandom Forest LOOCV Metrics:\n")
            f.write(f"Accuracy: {rf_metrics[0]}\nPrecision: {rf_metrics[1]}\nRecall: {rf_metrics[2]}\nF1 Score: {rf_metrics[3]}\nMCC: {rf_metrics[4]}\n")
            f.write("\nXGBoost LOOCV Metrics:\n")
            f.write(f"Accuracy: {xgb_metrics[0]}\nPrecision: {xgb_metrics[1]}\nRecall: {xgb_metrics[2]}\nF1 Score: {xgb_metrics[3]}\nMCC: {xgb_metrics[4]}\n")
            f.write("\nNeural Network LOOCV Metrics:\n")
            f.write(f"Accuracy: {nn_metrics[0]}\nPrecision: {nn_metrics[1]}\nRecall: {nn_metrics[2]}\nF1 Score: {nn_metrics[3]}\nMCC: {nn_metrics[4]}\n")
        pbar.update(1)

        # Save the processed data and models for SHAP analysis in the second part
        np.savez(os.path.join(preprocessed_data_dir, "processed_data.npz"), X_processed_machine_learning=X_processed_machine_learning.to_numpy(), y=y.to_numpy())
        joblib.dump(lr_model, os.path.join(model_save_dir, 'lr_model.joblib'))
        joblib.dump(rf_model, os.path.join(model_save_dir, 'rf_model.joblib'))
        joblib.dump(xgb_model, os.path.join(model_save_dir, 'xgb_model.joblib'))
        joblib.dump(nn_model, os.path.join(model_save_dir, 'nn_model.joblib'))

        pbar.update(1)

if __name__ == "__main__":
    main()




# %%
# STEP 9 ADDITION OF K-FOLD RANDOMIZED CROSS VALIDATION(RCV)

# The purpose of adding RCV is comparing the evaluation results between more detailed LOOCV and lighter RCV
# Since The least populated class in y has only 5 members, the number of RCV k-fold splits is 5.
# With Logistic Regression, dynamic file paths, pickle file intermediate saving

# 27 and 29 July 2028 modifications: introduced X_processed_machine_learning consistent with Step 9 LOOCV and Step 10

import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
import os
from tqdm.notebook import tqdm
import pickle

# Define custom scoring functions
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0),
    'mcc': make_scorer(matthews_corrcoef)
}

# Function to perform RCV
def perform_rcv(model, X, y, n_splits=5, n_iter=5, save_path=None):  # Reduced n_splits to 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {metric: [] for metric in scoring}
    
    try:
        for i in tqdm(range(n_iter), desc="RCV Iterations"):
            for metric, scorer in scoring.items():
                score = cross_val_score(model, X, y, cv=skf, scoring=scorer, n_jobs=-1)
                scores[metric].append(score)
                # Save intermediate results
                if save_path:
                    with open(save_path, 'wb') as f:
                        pickle.dump(scores, f)
    except KeyboardInterrupt:
        print("Process interrupted! Saving partial results...")
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(scores, f)
        print(f"Partial results saved to {save_path}")
        raise
    
    mean_scores = {metric: np.mean(scores[metric]) for metric in scores}
    std_scores = {metric: np.std(scores[metric]) for metric in scores}
    return mean_scores, std_scores

# Dynamic input prompt for data path
data_path = os.path.expanduser(input("\nPlease state the FILE path for the intermediate result .npz file to use (e.g.: ~/evaluation-sentiment-analysis/simulations/sim-21/intermediate-results-npz-pkl/processed_data.npz):\n"))
data = np.load(data_path)
X_processed_machine_learning = data['X_processed_machine_learning']
y = data['y']

# Dynamic input prompts for model paths
lr_model_path = os.path.expanduser(input("\nPlease state the FILE path for the Logistic Regression model to utilize (e.g. ~/evaluation-sentiment-analysis/simulations/sim-21/models-hyperparameters-evaluation/lr_model.joblib):\n"))
rf_model_path = os.path.expanduser(input("\nPlease state the FILE path for the Random Forests model to utilize (e.g. ~/evaluation-sentiment-analysis/simulations/sim-21/models-hyperparameters-evaluation/rf_model.joblib):\n"))
xgb_model_path = os.path.expanduser(input("\nPlease state the FILE path for the XGBoost model to utilize (e.g. ~/evaluation-sentiment-analysis/simulations/sim-21/models-hyperparameters-evaluation/xgb_model.joblib):\n"))
nn_model_path = os.path.expanduser(input("\nPlease state the FILE path for the Neural Network model to utilize (e.g. ~/evaluation-sentiment-analysis/simulations/sim-21/models-hyperparameters-evaluation/nn_model.joblib):\n"))

# Load the trained models
lr_model = joblib.load(lr_model_path)
rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)
nn_model = joblib.load(nn_model_path)

# Define paths to save partial results
lr_save_path = os.path.expanduser(input("\nPlease state the FILE path to save partial results for Logistic Regression (e.g.: ~/evaluation-sentiment-analysis/simulations/sim-21/intermediate-results-npz-pkl/lr_partial_results.pkl):\n"))
rf_save_path = os.path.expanduser(input("\nPlease state the FILE path to save partial results for Random Forest (e.g.: ~/evaluation-sentiment-analysis/simulations/sim-21/intermediate-results-npz-pkl/rf_partial_results.pkl):\n"))
xgb_save_path = os.path.expanduser(input("\nPlease state the FILE path to save partial results for XGBoost (e.g.: ~/evaluation-sentiment-analysis/simulations/sim-21/intermediate-results-npz-pkl/xgb_partial_results.pkl):\n"))
nn_save_path = os.path.expanduser(input("\nPlease state the FILE path to save partial results for Neural Network (e.g.: ~/evaluation-sentiment-analysis/simulations/sim-21/intermediate-results-npz-pkl/nn_partial_results.pkl):\n"))

# Perform RCV on LogisticRegression model
lr_mean_scores, lr_std_scores = perform_rcv(lr_model, X_processed_machine_learning, y, save_path=lr_save_path)
print(f"LogisticRegression RCV Mean Scores: {lr_mean_scores}")
print(f"LogisticRegression RCV Std Scores: {lr_std_scores}")

# Perform RCV on RandomForest model
rf_mean_scores, rf_std_scores = perform_rcv(rf_model, X_processed_machine_learning, y, save_path=rf_save_path)
print(f"RandomForest RCV Mean Scores: {rf_mean_scores}")
print(f"RandomForest RCV Std Scores: {rf_std_scores}")

# Perform RCV on XGBoost model
xgb_mean_scores, xgb_std_scores = perform_rcv(xgb_model, X_processed_machine_learning, y, save_path=xgb_save_path)
print(f"XGBoost RCV Mean Scores: {xgb_mean_scores}")
print(f"XGBoost RCV Std Scores: {xgb_std_scores}")

# Perform RCV on Neural Network model
nn_mean_scores, nn_std_scores = perform_rcv(nn_model, X_processed_machine_learning, y, save_path=nn_save_path)
print(f"Neural Network RCV Mean Scores: {nn_mean_scores}")
print(f"Neural Network RCV Std Scores: {nn_std_scores}")


# %%
# Step 10 SCREEN XGB VERSION: from multidimensional factor weighs to factor importance visualization by SHAP value.

# This step produces: SHAP value saving in HDF5 and JSON formats in a way that: 
#         (1) graphically highlights the most important factors (features) with the SHAP graphs libraries. 
#         (2) facilitates performing future causal analyses.

# After several unsuccessful attempts, I gave up preparing 2D xlsx dataset extractions, preferring to retain the data in the above formats.

# Step 10 version trials
#       11.21 DOES NOT WORK: Mismatch between feature dimensions and X_processed: 1514 vs 1511, and it seems the Step 9 processed_data.npz file is all 0 (zero)
#       11.26, WORKS, slow (8 hours despite parallelization), does not finalize graphs, says no instances for coFin=1 or coFin=0, 
#              requires http://localhost:8888/lab/tree/shap-visualization-02.ipynb to visualize graphs from h5 file.
#       26-27 July 2024 modifications to TEST: simplified and made more consistent with Step 9 after, focuses now on X_processed_machine_learning - did not work because of mismatch between X_processed_machine_learning and SHAP features.
#       29 Jul 2024 modifications aimed at standardizing X_processed_machine_learning Step 9 and Step 10 features, #-commented XGB parts, some graphs for LR and RF.
#       06 Aug 2024 modifications from Gonçalo: changed the argument of KernelExplainer, which outputed an array of shape (1,1), to predict_proba aligned to TreeExplainer, which outputs an array of shape (1,n_classes), by proRate class 1 and 4. I think to #-commented all XGB parts to prevent errors
#       11 Aug 2024 updated to xgboost==2.1.0 from xgboost==2.0.3 and shap==0.46 from shap==0.43, which were used in all other packages.Added shap_values_xgb = np.asarray(shap_values_xgb,dtype='float64') into all models. #-commented all XGB parts.
#       24 Aug 2024 changed X_processed_subset (with subset_size==10) into X_processed_machine_learning to align all models in Shap calculation. 
#       25 Aug 2024 keep "xgb_model.set_params(booster='gbtree')" #-commented since the training booster is dart, open up all the rest of XGB.

# Check XGB booster in Step 9 LOOCV!!!
#       IF 'booster' is 'gblinear'  THEN use: explainer_xgb = shap.LinearExplainer(xgb_model, X_processed_machine_learning) NOT a tree-based booster (TreeExplainer or Dart). Do not add: masker=X_processed_machine_learning in the argument, it would be redundant.
#       IF 'booster' is 'dart'      THEN use: explainer_xgb = shap.TreeExplainer(xgb_model)

# Step 10 SCREEN XGB VERSION: from multidimensional factor weights to factor importance visualization by SHAP value.

# This step produces: SHAP value saving in HDF5 and JSON formats in a way that: 
#         (1) graphically highlights the most important factors (features) with SHAP graphs.
#         (2) facilitates performing future causal analyses.

# 26/09/2024, aligned confusion matrix labels (1,2,3,4) and proRate classes (0,1,2,3)

import numpy as np
import h5py
import shap
import pandas as pd
import joblib
import os
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Hardcoded input/output paths
shap_output_file_base = "/absolute/path/to/sim-XX-step10-shap"
model_dir = "/absolute/path/to/model/directory"
data_file = "/absolute/path/to/sim-xx/processed_data.npz"
preprocessed_output_file = "/absolute/path/to/sim-xx-step9-weights-normalized.xlsx"
logging_file = "/absolute/path/to/sim-xx/Step-10-log.txt"

# Expand user paths
model_dir = os.path.expanduser(model_dir)
data_file = os.path.expanduser(data_file)
preprocessed_output_file = os.path.expanduser(preprocessed_output_file)
shap_output_file_base = os.path.expanduser(shap_output_file_base)
logging_file = os.path.expanduser(logging_file)

# Configure logging
logging.basicConfig(filename=logging_file, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

# Log input information
logging.info(f"SHAP output file base: {shap_output_file_base}")
logging.info(f"Model directory: {model_dir}")
logging.info(f"Data file: {data_file}")
logging.info(f"Preprocessed output file: {preprocessed_output_file}")
logging.info(f"Logging file: {logging_file}")

# Create output directory if it doesn't exist
output_dir = os.path.dirname(shap_output_file_base)
os.makedirs(output_dir, exist_ok=True)

# Load the processed data
data = np.load(data_file, allow_pickle=True)
X_processed_machine_learning = data["X_processed_machine_learning"]
y = data["y"]  # proRate classes: 0, 1, 2, 3
logging.info("Processed data loaded successfully.")
logging.info(f"X_processed_machine_learning shape: {X_processed_machine_learning.shape}")
logging.info(f"y shape: {y.shape}")

# Load feature names from the preprocessed data saved earlier
X_preprocessed_df = pd.read_excel(preprocessed_output_file)
logging.info("Feature names loaded successfully.")

# Extract feature names
feature_names = X_preprocessed_df.columns.to_numpy()

# Verify dimensions
if X_processed_machine_learning.shape[1] != len(feature_names):
    logging.error(f"Mismatch between feature dimensions and X_processed_machine_learning: {X_processed_machine_learning.shape[1]} vs {len(feature_names)}")
assert X_processed_machine_learning.shape[1] == len(feature_names), f"Mismatch between feature dimensions and X_processed_machine_learning: {X_processed_machine_learning.shape[1]} vs {len(feature_names)}"
logging.info("Feature dimensions match.")

# Load the trained models
xgb_model = joblib.load(os.path.join(model_dir, 'xgb_model.joblib'))
rf_model = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
nn_model = joblib.load(os.path.join(model_dir, 'nn_model.joblib'))
lr_model = joblib.load(os.path.join(model_dir, 'lr_model.joblib'))
logging.info("Trained models loaded successfully.")

# Calculate SHAP values for the entire dataset (classes: 0, 1, 2, 3)
logging.info("Calculating SHAP values for XGBoost model...")
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_processed_machine_learning)
shap_values_xgb = np.asarray(shap_values_xgb, dtype='float64')

logging.info("Calculating SHAP values for Random Forest model...")
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_processed_machine_learning)
shap_values_rf = np.asarray(shap_values_rf, dtype='float64')

logging.info("Calculating SHAP values for Neural Network model...")
explainer_nn = shap.KernelExplainer(nn_model.predict_proba, X_processed_machine_learning)
shap_values_nn = explainer_nn.shap_values(X_processed_machine_learning)
shap_values_nn = np.asarray(shap_values_nn, dtype='float64')

logging.info("Calculating SHAP values for Logistic Regression model...")
explainer_lr = shap.LinearExplainer(lr_model, X_processed_machine_learning)
shap_values_lr = explainer_lr.shap_values(X_processed_machine_learning)
shap_values_lr = np.asarray(shap_values_lr, dtype='float64')

# Save SHAP values and metadata to HDF5 file
hdf5_file = shap_output_file_base + '.h5'
with h5py.File(hdf5_file, 'w') as hf:
    hf.create_dataset('shap_values_xgb', data=shap_values_xgb)
    hf.create_dataset('shap_values_rf', data=shap_values_rf)
    hf.create_dataset('shap_values_nn', data=shap_values_nn)
    hf.create_dataset('shap_values_lr', data=shap_values_lr)
    hf.create_dataset('feature_names', data=np.array(feature_names, dtype='S'))
    hf.create_dataset('target', data=y)

# Save SHAP values and metadata to JSON file
json_file = shap_output_file_base + '.json'
shap_values_dict = {
    'shap_values_xgb': shap_values_xgb.tolist(),
    'shap_values_rf': shap_values_rf.tolist(),
    'shap_values_nn': shap_values_nn.tolist(),
    'shap_values_lr': shap_values_lr.tolist(),
    'feature_names': feature_names.tolist(),
    'target': y.tolist()
}

with open(json_file, 'w') as json_file:
    json.dump(shap_values_dict, json_file)

# Separate indices for coFin=1 and coFin=0
coFin_index = np.where(feature_names == 'coFin')[0][0]
coFin_1_indices = np.where(X_processed_machine_learning[:, coFin_index] == 1)[0]
coFin_0_indices = np.where(X_processed_machine_learning[:, coFin_index] == 0)[0]

def plot_feature_importance_and_summary(shap_values, model_name, indices, condition, class_index):
    if indices.size == 0:
        logging.warning(f"No instances for {condition}")
        return
    
    logging.info(f"Model: {model_name}| class {class_index} | Plotting feature importance and summary plots for {condition}...")

    # Feature Importance plot
    shap.summary_plot(shap_values[indices][:, :, class_index], X_processed_machine_learning[indices], feature_names=feature_names, plot_type="bar", max_display=50, show=False)
    plt.title(f"Feature Importance - {model_name} ({condition})", fontsize=18)
    plt.savefig(f"{shap_output_file_base}_{model_name}_importance_{condition}_{class_index}.png")
    plt.close()

    # Summary plot
    shap.summary_plot(shap_values[indices][:, :, class_index], X_processed_machine_learning[indices], feature_names=feature_names, max_display=50, show=False)
    plt.title(f"Summary Plot - {model_name} ({condition}) | class {class_index}", fontsize=18)
    plt.savefig(f"{shap_output_file_base}_{model_name}_summary_{condition}_{class_index}.png")
    plt.close()

# SHAP force plot for first instance
def plot_shap_force_plot(explainer, shap_values, model_name, indices, condition, class_index):
    if indices.size == 0:
        logging.warning(f"No instances for {condition}")
        return

    logging.info(f"Model: {model_name}| class {class_index} | Plotting force plot for {condition}...")
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value[class_index], shap_values[indices][:, :, class_index], X_processed_machine_learning[indices], feature_names=feature_names)
    shap.save_html(f"{shap_output_file_base}_{model_name}_force_plot_{condition}_{class_index}.html", force_plot)

# Plot feature importance and summary plots for each class and each model
classes = [0, 1, 2, 3]  # proRate classes (0 to 3)
for class_index in classes:
    # Plot for each model separated by coFin
    if coFin_1_indices.size > 0:
        plot_feature_importance_and_summary(shap_values_xgb, "XGBoost", coFin_1_indices, "coFin1", class_index)
        plot_shap_force_plot(explainer_xgb, shap_values_xgb, "XGBoost", coFin_1_indices, "coFin1", class_index)
        plot_feature_importance_and_summary(shap_values_rf, "RandomForest", coFin_1_indices, "coFin1", class_index)
        plot_shap_force_plot(explainer_rf, shap_values_rf, "RandomForest", coFin_1_indices, "coFin1", class_index)
        plot_feature_importance_and_summary(shap_values_nn, "NeuralNetwork", coFin_1_indices, "coFin1", class_index)
        plot_shap_force_plot(explainer_nn, shap_values_nn, "NeuralNetwork", coFin_1_indices, "coFin1", class_index)
        plot_feature_importance_and_summary(shap_values_lr, "LogisticRegression", coFin_1_indices, "coFin1", class_index)
        plot_shap_force_plot(explainer_lr, shap_values_lr, "LogisticRegression", coFin_1_indices, "coFin1", class_index)
    else:
        logging.info("No instances with coFin=1")
        print("No instances with coFin=1")

    if coFin_0_indices.size > 0:
        plot_feature_importance_and_summary(shap_values_xgb, "XGBoost", coFin_0_indices, "coFin0", class_index)
        plot_shap_force_plot(explainer_xgb, shap_values_xgb, "XGBoost", coFin_0_indices, "coFin0", class_index)
        plot_feature_importance_and_summary(shap_values_rf, "RandomForest", coFin_0_indices, "coFin0", class_index)
        plot_shap_force_plot(explainer_rf, shap_values_rf, "RandomForest", coFin_0_indices, "coFin0", class_index)
        plot_feature_importance_and_summary(shap_values_nn, "NeuralNetwork", coFin_0_indices, "coFin0", class_index)
        plot_shap_force_plot(explainer_nn, shap_values_nn, "NeuralNetwork", coFin_0_indices, "coFin0", class_index)
        plot_feature_importance_and_summary(shap_values_lr, "LogisticRegression", coFin_0_indices, "coFin0", class_index)
        plot_shap_force_plot(explainer_lr, shap_values_lr, "LogisticRegression", coFin_0_indices, "coFin0", class_index)
    else:
        logging.info("No instances with coFin=0")
        print("No instances with coFin=0")

logging.info("SHAP values and plots successfully saved.")
print("SHAP values and plots successfully saved.")

# Update: Confusion matrices to include proRate labels 1 to 4
def plot_confusion_matrix(model, X, y_true, model_name, condition):
    # Shift proRate classes to labels 1, 2, 3, 4 for confusion matrix
    y_true_shifted = y_true + 1
    y_pred_shifted = model.predict(X) + 1

    cm = confusion_matrix(y_true_shifted, y_pred_shifted, labels=[1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name} ({condition})", fontsize=18)
    plt.savefig(f"{shap_output_file_base}_{model_name}_confusion_matrix_{condition}.png")
    plt.close()

# Confusion matrices for coFin=1
if coFin_1_indices.size > 0:
    plot_confusion_matrix(xgb_model, X_processed_machine_learning[coFin_1_indices], y[coFin_1_indices], "XGBoost", "coFin1")
    plot_confusion_matrix(rf_model, X_processed_machine_learning[coFin_1_indices], y[coFin_1_indices], "RandomForest", "coFin1")
    plot_confusion_matrix(nn_model, X_processed_machine_learning[coFin_1_indices], y[coFin_1_indices], "NeuralNetwork", "coFin1")
    plot_confusion_matrix(lr_model, X_processed_machine_learning[coFin_1_indices], y[coFin_1_indices], "LogisticRegression", "coFin1")
else:
    logging.info("No instances with coFin=1 for confusion matrix")
    print("No instances with coFin=1 for confusion matrix")

# Confusion matrices for coFin=0
if coFin_0_indices.size > 0:
    plot_confusion_matrix(xgb_model, X_processed_machine_learning[coFin_0_indices], y[coFin_0_indices], "XGBoost", "coFin0")
    plot_confusion_matrix(rf_model, X_processed_machine_learning[coFin_0_indices], y[coFin_0_indices], "RandomForest", "coFin0")
    plot_confusion_matrix(nn_model, X_processed_machine_learning[coFin_0_indices], y[coFin_0_indices], "NeuralNetwork", "coFin0")
    plot_confusion_matrix(lr_model, X_processed_machine_learning[coFin_0_indices], y[coFin_0_indices], "LogisticRegression", "coFin0")
else:
    logging.info("No instances with coFin=0 for confusion matrix")
    print("No instances with coFin=0 for confusion matrix")

logging.info("Confusion matrices for co-financed and non-co-financed projects successfully saved.")
print("Confusion matrices for co-financed and non-co-financed projects successfully saved.")



# %%
# STEP 10 NICER GRAPHS

# Graphs on the sole top 20 factors by SHAP value

# NOT TESTED


import numpy as np
import h5py
import shap
import matplotlib.pyplot as plt
import os

# Set font size for readability in presentations
plt.rcParams.update({'font.size': 18})

# File paths
shap_output_file_base = "/home/drago/absolute/path/to/sim-XX/sim-XX-step10-shap"
hdf5_file = shap_output_file_base + '.h5'

# Directory to save the new graphs
output_dir = os.path.dirname(shap_output_file_base)

# Load SHAP values and feature names from HDF5 file
with h5py.File(hdf5_file, 'r') as hf:
    shap_values_rf = np.array(hf['shap_values_rf'])
    shap_values_nn = np.array(hf['shap_values_nn'])
    shap_values_lr = np.array(hf['shap_values_lr'])
    feature_names = np.array(hf['feature_names']).astype(str)
   
    shap_values_xgb = np.array(hf['shap_values_xgb'])  # Uncomment if XGBoost SHAP values were saved

# Function to get top 20 features based on mean absolute SHAP values
def get_top_features(shap_values, feature_names, top_n=20):
    mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]  # Get indices of top features
    top_feature_names = feature_names[top_indices]
    top_shap_values = shap_values[:, top_indices, :]
    return top_feature_names, top_shap_values

# Get top 20 features for each model
top_features_rf, top_shap_values_rf = get_top_features(shap_values_rf, feature_names)
top_features_nn, top_shap_values_nn = get_top_features(shap_values_nn, feature_names)
top_features_lr, top_shap_values_lr = get_top_features(shap_values_lr, feature_names)

top_features_xgb, top_shap_values_xgb = get_top_features(shap_values_xgb, feature_names) # Uncomment if using XGBoost

# Plot function for nicer graphs
def plot_nicer_shap_summary(shap_values, feature_names, model_name):
    shap.summary_plot(shap_values, 
                      feature_names=feature_names, 
                      plot_type="bar", 
                      max_display=len(feature_names), 
                      show=False)
    plt.title(f"Top {len(feature_names)} Feature Importance - {model_name}", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_Top{len(feature_names)}_Feature_Importance.png"))
    plt.close()

    shap.summary_plot(shap_values, 
                      feature_names=feature_names, 
                      max_display=len(feature_names), 
                      show=False)
    plt.title(f"Top {len(feature_names)} SHAP Summary - {model_name}", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_Top{len(feature_names)}_SHAP_Summary.png"))
    plt.close()

# Generate plots for Random Forest
plot_nicer_shap_summary(top_shap_values_rf, top_features_rf, "RandomForest")

# Generate plots for Neural Network
plot_nicer_shap_summary(top_shap_values_nn, top_features_nn, "NeuralNetwork")

# Generate plots for Logistic Regression
plot_nicer_shap_summary(top_shap_values_lr, top_features_lr, "LogisticRegression")

# Uncomment if using XGBoost
plot_nicer_shap_summary(top_shap_values_xgb, top_features_xgb, "XGBoost")

print("Nicer SHAP summary plots for top 20 features have been saved.")





# %%
# Step 11 Bootstrap Sampling by model, coFin, proRate

# 07/09/2024, sophisticated version. NOTE: when the yerr value is negative (frequent in LR), the script puts value Zero to prevent MatplotLib errors 
# 09/09/2024, list of MatPlotLib colors at: https://matplotlib.org/stable/gallery/color/named_colors.html; for simulation 01 (sim-21) I used "skyblue", simulation-04 (sim-42) "green"


import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def main():
    # Define paths for the .h5 and .npz files
    h5_file_path = "/absolute/path/to/sim-XX/shap-values-graphs/sim-XX-step10-shap.h5"
    npz_file_path = "/absolute/path/to/sim-XX/processed_data.npz"
    output_file = "/absolute/path/to/sim-XX//shap-values-graphs/sim-XX-shap-confidence.xlsx"

    # Directory to save the plots (same as the Excel output file)
    plot_dir = os.path.dirname(output_file)

    # Load the SHAP values and feature names from the HDF5 file
    with h5py.File(h5_file_path, 'r') as file:
        shap_values_dict = {
            'LR': file['shap_values_lr'][:],
            'RF': file['shap_values_rf'][:],
            'XGB': file['shap_values_xgb'][:],
            'NN': file['shap_values_nn'][:]
        }
        feature_names = file['feature_names'][:].astype(str)

    # Load the feature matrix (X) and proRate (y) from the .npz file
    data = np.load(npz_file_path)
    X = data['X_processed_machine_learning']  # Feature matrix
    proRate = data['y']  # proRate (target variable)

    # Convert feature names to numpy array of strings
    feature_names = np.array(feature_names).astype(str)

    # Find the index of 'coFin' feature
    coFin_index = np.where(feature_names == 'coFin')[0][0]

    # Prompt the user for the ML model
    model_choice = input("Enter the model you want to assess (LR, RF, XGB, NN): ").upper()
    if model_choice not in shap_values_dict:
        print(f"Invalid model choice. Available options: LR, RF, XGB, NN")
        return

    # Load the SHAP values for the selected model
    shap_values = shap_values_dict[model_choice]

    # Prompt the user for coFin and proRate values
    coFin_value = int(input("Enter the value for coFin (e.g., 0 or 1): "))
    proRate_value = int(input("Enter the value for proRate (e.g., 1, 2, 3, 4): ")) - 1  # Adjust proRate to 0-3 internally

    # Get indices of instances matching the user's criteria
    filtered_indices = np.where(
        (X[:, coFin_index] == coFin_value) &
        (proRate == proRate_value)
    )[0]

    if len(filtered_indices) == 0:
        print("No data points match the specified coFin and proRate values.")
        return

    # Number of bootstrap samples
    n_bootstrap = 1000

    # Initialize array to store bootstrap results
    bootstrap_results = np.zeros((n_bootstrap, shap_values.shape[1]))

    # Perform bootstrap sampling
    for i in tqdm(range(n_bootstrap), desc="Bootstrap Sampling"):
        sample_indices = np.random.choice(filtered_indices, size=len(filtered_indices), replace=True)
        # The third dimension (2) of shap_values corresponds to proRate, so we select the proRate_value slice
        bootstrap_results[i] = np.mean(shap_values[sample_indices, :, proRate_value], axis=0)

    # Calculate the mean SHAP values and confidence intervals
    mean_shap = np.mean(bootstrap_results, axis=0)
    ci_low = np.percentile(bootstrap_results, 2.5, axis=0)
    ci_high = np.percentile(bootstrap_results, 97.5, axis=0)

    # Create a DataFrame for the results
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean SHAP Value': mean_shap,
        'CI Lower': ci_low,
        'CI Upper': ci_high
    }).sort_values(by='Mean SHAP Value', ascending=False)

    # Select the top 10 features
    top_features = df.head(10)

    # Adjusted proRate value (1-based for display purposes)
    adjusted_proRate_value = proRate_value + 1

    # Plotting
    plt.figure(figsize=(12, 6))
    yerr = [
        np.clip(top_features['Mean SHAP Value'] - top_features['CI Lower'], 0, None),  # Clip negative values to 0
        np.clip(top_features['CI Upper'] - top_features['Mean SHAP Value'], 0, None)   # Clip negative values to 0
    ]
    plt.bar(
        top_features['Feature'],
        top_features['Mean SHAP Value'],
        yerr=yerr,
        capsize=5,
        ecolor='black',
        color='skyblue'
    )
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(fontsize=18)  # Increase Y-axis font size to 18 points
    plt.xlabel('Features', fontsize=18)
    plt.ylabel('Mean SHAP Value', fontsize=18)
    plt.title(f'Top 10 Features for {model_choice} (coFin={coFin_value}, proRate={adjusted_proRate_value})', fontsize=18)
    plt.tight_layout()


    # Save plot to the same directory as the Excel file
    plot_filename = os.path.join(plot_dir, f"{model_choice}_Top10_Features_coFin_{coFin_value}_proRate_{adjusted_proRate_value}.png")
    plt.savefig(plot_filename)
    plt.close()

    # Save the top features to the Excel file with model name, coFin, and proRate suffix
    excel_file_with_suffix = f"{os.path.splitext(output_file)[0]}_{model_choice}_coFin_{coFin_value}_proRate_{adjusted_proRate_value}.xlsx"
    with pd.ExcelWriter(excel_file_with_suffix) as writer:
        df.to_excel(writer, sheet_name=f'{model_choice}_coFin_{coFin_value}_proRate_{adjusted_proRate_value}')

if __name__ == "__main__":
    main()


# %%
STEP 12: SAGE values

Step added on Aug. 16th, based on fryers-2021 warnings on SHAP value misuse for feature selection, and covert-2020 recommendation for SAGE.
DID NOT WORK!!! RE-TEST!!!


import numpy as np
import h5py
import pandas as pd
import joblib
import os
import json
import logging
import matplotlib.pyplot as plt
import sage
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Dynamic input prompts
sage_output_file_base = input("Output: Please state the BASE FILE path to save the SAGE value files (e.g., sage-values-11.08); the script will automatically create an HDF5 and a JSON file: ")
model_dir = input("Input: Please write the dynamic DIRECTORY path for the trained models you want to utilize (e.g., ~/Documenti/phd-trial-standardized-small/trained-models/rf-XGB-nn-lr): ")
data_file = input("Input: Please write the dynamic FILE path for the pre-processed data-frame in .npz format (e.g., ~/Documenti/phd-trial-standardized-small/output/intermediate-results/processed_data.npz): ")
preprocessed_output_file = input("Input: Please enter the path to the standardized data-frame Excel FILE (e.g., factor_weights_sigma_std_11.03.xlsx): ")
logging_file = input("Output: Please enter the path for the logging FILE (e.g., ~/Documenti/phd-trial-standardized-small/output/logs/step11.log): ")

# Expand user paths
model_dir = os.path.expanduser(model_dir)
data_file = os.path.expanduser(data_file)
preprocessed_output_file = os.path.expanduser(preprocessed_output_file)
sage_output_file_base = os.path.expanduser(sage_output_file_base)
logging_file = os.path.expanduser(logging_file)

# Configure logging
logging.basicConfig(filename=logging_file, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

# Log input information
logging.info(f"SAGE output file base: {sage_output_file_base}")
logging.info(f"Model directory: {model_dir}")
logging.info(f"Data file: {data_file}")
logging.info(f"Preprocessed output file: {preprocessed_output_file}")
logging.info(f"Logging file: {logging_file}")

# Create output directory if it doesn't exist
output_dir = os.path.dirname(sage_output_file_base)
os.makedirs(output_dir, exist_ok=True)

# Load the processed data
data = np.load(data_file, allow_pickle=True)
X_processed_machine_learning = data["X_processed_machine_learning"]
y = data["y"]
logging.info("Processed data loaded successfully.")
logging.info(f"X_processed_machine_learning shape: {X_processed_machine_learning.shape}")
logging.info(f"y shape: {y.shape}")

# Load feature names from the preprocessed data saved earlier
X_preprocessed_df = pd.read_excel(preprocessed_output_file)
logging.info("Feature names loaded successfully.")

# Extract feature names
feature_names = X_preprocessed_df.columns.to_numpy()

# Verify dimensions
if X_processed_machine_learning.shape[1] != len(feature_names):
    logging.error(f"Mismatch between feature dimensions and X_processed_machine_learning: {X_processed_machine_learning.shape[1]} vs {len(feature_names)}")
assert X_processed_machine_learning.shape[1] == len(feature_names), f"Mismatch between feature dimensions and X_processed_machine_learning: {X_processed_machine_learning.shape[1]} vs {len(feature_names)}"
logging.info("Feature dimensions match.")

# Identify and handle zero variance features
logging.info("Data variance check:")
variance = np.var(X_processed_machine_learning, axis=0)
logging.info(f"Variance: {variance}")
zero_variance_features = np.where(variance == 0)[0]
if len(zero_variance_features) > 0:
    logging.warning(f"Features with zero variance: {zero_variance_features}")

# Load the trained models
rf_model = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
nn_model = joblib.load(os.path.join(model_dir, 'nn_model.joblib'))
lr_model = joblib.load(os.path.join(model_dir, 'lr_model.joblib'))
# xgb_model = joblib.load(os.path.join(model_dir, 'xgb_model.joblib')) # XGBoost model

logging.info("Trained models loaded successfully.")

# Define the loss function: Cross-Entropy for classification tasks
loss_fn = sage.CrossEntropy()

# Initialize SAGE explainers for each model
logging.info("Calculating SAGE values for Random Forest model...")
explainer_rf = sage.PermutationExplainer(rf_model.predict_proba, X_processed_machine_learning, loss_fn)
sage_values_rf = explainer_rf(X_processed_machine_learning)
logging.info("SAGE values calculated for Random Forest model.")

logging.info("Calculating SAGE values for Neural Network model...")
explainer_nn = sage.PermutationExplainer(nn_model.predict_proba, X_processed_machine_learning, loss_fn)
sage_values_nn = explainer_nn(X_processed_machine_learning)
logging.info("SAGE values calculated for Neural Network model.")

logging.info("Calculating SAGE values for Logistic Regression model...")
explainer_lr = sage.PermutationExplainer(lr_model.predict_proba, X_processed_machine_learning, loss_fn)
sage_values_lr = explainer_lr(X_processed_machine_learning)
logging.info("SAGE values calculated for Logistic Regression model.")

# XGBoost SAGE value calculation, commented out for later trials
# logging.info("Calculating SAGE values for XGBoost model...")
# explainer_xgb = sage.PermutationExplainer(xgb_model.predict_proba, X_processed_machine_learning, loss_fn)
# sage_values_xgb = explainer_xgb(X_processed_machine_learning)
# logging.info("SAGE values calculated for XGBoost model.")

# Save SAGE values and metadata to HDF5 file
hdf5_file = sage_output_file_base + '.h5'
with h5py.File(hdf5_file, 'w') as hf:
    hf.create_dataset('sage_values_rf', data=np.array(sage_values_rf.values))
    hf.create_dataset('sage_values_nn', data=np.array(sage_values_nn.values))
    hf.create_dataset('sage_values_lr', data=np.array(sage_values_lr.values))
    # hf.create_dataset('sage_values_xgb', data=np.array(sage_values_xgb.values)) # Commented out for later trials
    hf.create_dataset('feature_names', data=np.array(feature_names, dtype='S'))
    hf.create_dataset('target', data=y)

# Save SAGE values and metadata to JSON file
sage_values_dict = {
    'sage_values_rf': sage_values_rf.values.tolist(),
    'sage_values_nn': sage_values_nn.values.tolist(),
    'sage_values_lr': sage_values_lr.values.tolist(),
    # 'sage_values_xgb': sage_values_xgb.values.tolist(), # Commented out for later trials
    'feature_names': feature_names.tolist(),
    'target': y.tolist()
}

json_file = sage_output_file_base + '.json'
with open(json_file, 'w') as f:
    json.dump(sage_values_dict, f)

# Function to plot SAGE feature importance and save as PNG
def plot_sage_importance(sage_values, model_name):
    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, sage_values.values)
    plt.xlabel("SAGE Value")
    plt.ylabel("Features")
    plt.title(f"SAGE Feature Importance - {model_name}")
    plt.savefig(f"{sage_output_file_base}_{model_name}_importance.png")
    plt.close()

# Plot SAGE importance for each model
plot_sage_importance(sage_values_rf, "RandomForest")
plot_sage_importance(sage_values_nn, "NeuralNetwork")
plot_sage_importance(sage_values_lr, "LogisticRegression")
# plot_sage_importance(sage_values_xgb, "XGBoost") # Commented out for later trials

logging.info("SAGE values and plots successfully saved.")
print("SAGE values and plots successfully saved.")

# Optional: Confusion matrix plotting function
def plot_confusion_matrix(model, X, y_true, model_name):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{sage_output_file_base}_{model_name}_confusion_matrix.png")
    plt.close()

# Plot confusion matrices for each model
plot_confusion_matrix(rf_model, X_processed_machine_learning, y, "RandomForest")
plot_confusion_matrix(nn_model, X_processed_machine_learning, y, "NeuralNetwork")
plot_confusion_matrix(lr_model, X_processed_machine_learning, y, "LogisticRegression")
# plot_confusion_matrix(xgb_model, X_processed_machine_learning, y, "XGBoost") # Commented out for later trials

logging.info("Confusion matrices successfully saved.")
print("Confusion matrices successfully saved.")


# %%
# STEP 13: pre-script to try bypassing factor ranking Shapley values while fixing Step 10

# Based on the "Methodological hypothesis 2 (Naive Counterfactual Difference Model -NCDM-)" described in  "Methodology for supervised learning_07.docx"
# Another part shall be added based on the Methodological hypothesis 1 (Naive Counterfactual Exclusion Model -NCEM-)

import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the dataset
file_path = '~/evaluation-sentiment-analysis/simulations/sim-21/weights-standardization/sim-21-step9-weights-standardized-clean.xlsx'  # Replace with the path to your dataset file

# Expand the user path
file_path = os.path.expanduser(file_path)

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} was not found. Please check the file path.")

df = pd.read_excel(file_path)

# Extract relevant columns
orgAcro, proCode, coFin, proRate, couCode, proYear = df.columns[:6]

# Calculate the sum of frequencies for each factor
sums_pos = df[df.columns[6::2]].sum()
sums_neg = df[df.columns[7::2]].sum()

# Dynamically set the threshold (use median for this example)
threshold_pos = sums_pos.quantile(0.25) # you can use the lower quartile Q1: sums_pos.quantile(0.25) or you can use the median sums_pos.median() or zero 0
threshold_neg = sums_neg.quantile(0.25) # you can use the lower quartile Q1: sums_neg.quantile(0.25) or you can use the median sums_neg.median() or zero 0

# Filter out factors with sums below the threshold
filtered_pos_columns = sums_pos[sums_pos >= threshold_pos].index
filtered_neg_columns = sums_neg[sums_neg >= threshold_neg].index

# Calculate the differences for the filtered factors
differences = {}
for pos_col, neg_col in zip(filtered_pos_columns, filtered_neg_columns):
    if pos_col in df.columns and neg_col in df.columns:
        νCO_k_pos = df.loc[df['coFin'] == 1, pos_col]
        νCO_k_neg = df.loc[df['coFin'] == 1, neg_col]
        
        # Calculate the differences (νCO+k - νCO-k) and ||(νCO-k - νCO+k)||
        differences[pos_col] = (νCO_k_pos - νCO_k_neg)
        differences[neg_col] = abs(νCO_k_neg - νCO_k_pos)

# Convert differences to a DataFrame
differences_df = pd.DataFrame(differences)

# Plot the distribution of the differences
plt.figure(figsize=(12, 6))
differences_df.stack().hist(bins=50)
plt.title(f'Distribution of Differences (νCO+k - νCO-k and ||(νCO-k - νCO+k)||)\nafter removing factors with sums frequency below Q1 or median: {threshold_pos}')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Compute Q3 for the differences across the entire population
q3_differences = differences_df.stack().quantile(0.75)

# Print Q3 value to verify
print(f"Q3 value for differences (entire population after filtering out factors with sums frequency below Q1 or median: {threshold_pos}):", q3_differences)

# Find and print the first 20 w_k_+ and w_k_- for which the difference is below Q3
below_q3_columns = []
for pos_col, neg_col in zip(filtered_pos_columns, filtered_neg_columns):
    if pos_col in df.columns and neg_col in df.columns:
        νCO_k_pos = df.loc[df['coFin'] == 1, pos_col]
        νCO_k_neg = df.loc[df['coFin'] == 1, neg_col]

        νCO_pos_diff = νCO_k_pos - νCO_k_neg
        νCO_neg_diff = abs(νCO_k_neg - νCO_k_pos)

        if (νCO_pos_diff < q3_differences).all() or (νCO_neg_diff < q3_differences).all():
            below_q3_columns.append((pos_col, neg_col))
            if len(below_q3_columns) >= 20:
                break

print("First 20 w_k_+ and w_k_- columns for which the difference is below Q3:")
for pos_col, neg_col in below_q3_columns:
    print(f"{pos_col}, {neg_col}")

# Initialize sets for CSF and CFF
csf = set()
cff = set()

# Extract CSF and CFF
for pos_col, neg_col in zip(filtered_pos_columns, filtered_neg_columns):
    if pos_col in df.columns and neg_col in df.columns:
        νCO_k_pos = df.loc[df['coFin'] == 1, pos_col]
        νCO_k_neg = df.loc[df['coFin'] == 1, neg_col]
        νCG_k_pos = df.loc[df['coFin'] == 0, pos_col]
        νCG_k_neg = df.loc[df['coFin'] == 0, neg_col]

        # Calculate the differences
        νCO_pos_diff = νCO_k_pos - νCO_k_neg
        νCO_neg_diff = abs(νCO_k_neg - νCO_k_pos)

        # Apply conditions for CSF and CFF
        if (νCO_pos_diff >= q3_differences).all() and (νCG_k_pos < q3_differences).all():
            csf.add(pos_col.split('_')[1])
            
        if (νCO_neg_diff >= q3_differences).all() and (νCG_k_neg < q3_differences).all():
            cff.add(neg_col.split('_')[1])

# Output the results
print("CSF (Critical Success Factors):", csf)
print("CFF (Critical Failure Factors):", cff)




