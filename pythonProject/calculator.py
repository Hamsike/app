from collections import Counter
from math import log
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class TfIdfCalculator:
    def __init__(self):
        self.stop_words = set(stopwords.words("english")) | set(string.punctuation)

    def tokenize(self, text):
        tokens = word_tokenize(text.lower())
        return [word for word in tokens if word not in self.stop_words]

    def calculate_tf_idf(self, corpus):
        tf_dicts = []

        # Подсчет частот встречаемости каждого слова в каждом документе
        for doc in corpus:
            words = self.tokenize(doc)
            counter = Counter(words)
            total_words = len(words)

            tf_values = {w: count / total_words for w, count in counter.items()}
            tf_dicts.append(tf_values)

        # Подсчет общего количества документов, содержащих каждое слово
        df_counts = {}
        for tf_dict in tf_dicts:
            for word in tf_dict.keys():
                df_counts[word] = df_counts.get(word, 0) + 1

        num_docs = len(corpus)

        # Расчет IDF
        idf_values = {
            word: log(num_docs / float(count)) for word, count in df_counts.items()
        }

        results = []
        for i, tf_dict in enumerate(tf_dicts):
            for word, tf_value in tf_dict.items():
                results.append({
                    'word': word,
                    'tf': round(tf_value, 4),
                    'idf': round(idf_values.get(word), 4)
                })

        sorted_results = sorted(results, key=lambda x: x['idf'], reverse=True)
        return sorted_results[:50]
