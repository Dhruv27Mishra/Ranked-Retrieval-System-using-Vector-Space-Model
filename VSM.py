import os
import math
from collections import defaultdict, Counter

# Helper functions
def calculate_document_lengths(documents):
    lengths = {}
    for doc_id, text in documents.items():
        term_freq = Counter(text.split())
        length = math.sqrt(sum(freq ** 2 for freq in term_freq.values()))
        lengths[doc_id] = length
    return lengths

def calculate_idf(documents):
    idf = {}
    df = defaultdict(int)
    total_docs = len(documents)
    
    for text in documents.values():
        terms = set(text.split())
        for term in terms:
            df[term] += 1
    
    for term, freq in df.items():
        idf[term] = math.log10(total_docs / freq)
    
    return idf, df

def calculate_tf_idf(documents, idf):
    tf_idf = defaultdict(lambda: defaultdict(float))
    
    for doc_id, text in documents.items():
        term_freq = Counter(text.split())
        length = math.sqrt(sum(freq ** 2 for freq in term_freq.values()))
        for term, freq in term_freq.items():
            tf = 1 + math.log10(freq)
            tf_idf[doc_id][term] = tf * idf[term] / length
    
    return tf_idf

def write_postings_to_file(idf, df, tf_idf):
    with open("posting.txt", "w") as f:
        for term, idf_value in idf.items():
            f.write(f"Term: {term}\n")
            f.write(f"Document Frequency (df): {df[term]}\n")
            f.write(f"Inverse Document Frequency (idf): {idf_value:.6f}\n")
            f.write("Posting List:\n")
            for doc_id, tf_idf_value in tf_idf.items():
                if term in tf_idf_value:
                    f.write(f"  Document ID: {doc_id}, TF-IDF: {tf_idf_value[term]:.6f}\n")
            f.write("\n")

def cosine_similarity(query_vector, doc_vector):
    numerator = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) for term in query_vector)
    query_norm = math.sqrt(sum(v ** 2 for v in query_vector.values()))
    doc_norm = math.sqrt(sum(v ** 2 for v in doc_vector.values()))
    return numerator / (query_norm * doc_norm) if query_norm and doc_norm else 0

def rank_documents(query, tf_idf, document_lengths):
    query_terms = query.split()
    query_tf = Counter(query_terms)
    idf = calculate_idf(documents)[0]
    query_vector = {term: (1 + math.log10(freq)) * idf.get(term, 0) for term, freq in query_tf.items()}
    
    scores = []
    for doc_id, doc_vector in tf_idf.items():
        score = cosine_similarity(query_vector, doc_vector)
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: (-x[1], x[0]))  # Sort by score (desc) and docID (asc)
    return scores[:10]

# Load documents from directory
def load_documents(corpus_dir):
    documents = {}
    for filename in os.listdir(corpus_dir):
        doc_id, _ = os.path.splitext(filename)
        with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as file:
            documents[doc_id] = file.read().strip()
    return documents

# Directory containing the documents
corpus_dir = './Corpus'

# Load documents
documents = load_documents(corpus_dir)

# Calculate document lengths
document_lengths = calculate_document_lengths(documents)

# Calculate IDF and DF
idf, df = calculate_idf(documents)

# Calculate TF-IDF for each document
tf_idf = calculate_tf_idf(documents, idf)

# Write term information to posting.txt
write_postings_to_file(idf, df, tf_idf)

# Main loop for user input
while True:
    query = input("Enter your search query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Rank documents for the query
    ranked_docs = rank_documents(query, tf_idf, document_lengths)

    # Output results
    print("Ranked Documents:")
    for doc_id, score in ranked_docs:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")
