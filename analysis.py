import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))


def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    return tokens


def extract_bigrams(text):
    tokens = tokenize(text)
    he_counts = Counter()
    she_counts = Counter()

    for i in range(len(tokens) - 1):
        if tokens[i] == 'he':
            next_word = tokens[i + 1]
            if next_word not in stop_words and len(next_word) > 2:
                he_counts[next_word] += 1
        elif tokens[i] == 'she':
            next_word = tokens[i + 1]
            if next_word not in stop_words and len(next_word) > 2:
                she_counts[next_word] += 1

    return he_counts, she_counts


def log_odds_ratio(a, b):
    total_a = sum(a.values())
    total_b = sum(b.values())
    vocab = set(a) | set(b)
    results = {}

    for word in vocab:
        count_a = a.get(word, 0) + 1
        count_b = b.get(word, 0) + 1
        results[word] = np.log((count_a / total_a) / (count_b / total_b))

    return results


def analyze_script(text, top_n=15):
    he_counts, she_counts = extract_bigrams(text)

    if sum(he_counts.values()) == 0 and sum(she_counts.values()) == 0:
        return None, None, None, None

    scores = log_odds_ratio(he_counts, she_counts)
    df = pd.DataFrame(scores.items(), columns=['word', 'log_odds']).sort_values('log_odds')

    # top_she: most negative log-odds (most associated with SHE), reversed for top-to-bottom display
    top_she = df.head(top_n).iloc[::-1]
    # top_he: most positive log-odds (most associated with HE)
    top_he = df.tail(top_n)

    return he_counts, she_counts, top_he, top_she
