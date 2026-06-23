"""Data loading and preprocessing for spam classification."""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
import scipy.sparse as sp
from enum import Enum




class DataLoader:
    """Class to load and preprocess data for spam classification."""

    def __init__(self):
        # Download required NLTK resources
        self._download_nltk_resources()

        # Initialize data containers
        self.data = None
        self.raw_text = None
        self.cleaned_text = None
        self.is_spam = None
        self.text_class = None

    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("stopwords")

    def load_data(self, filepath):
        """
        Load data from CSV file.

        Args:
            filepath (str): Path to the CSV file
        """
        self.data = pd.read_csv(filepath)
        self._process_data()
        return self.data

    def _preprocess_text(self, text):
        """
        Preprocess text by tokenizing and removing stopwords.

        Args:
            text (str): Input text to preprocess

        Returns:
            list: List of preprocessed tokens
        """
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(str(text).lower())
        tokens = np.array(
            [word for word in tokens if word.isalpha() and word not in stop_words]
        )
        return tokens

    def _process_data(self):
        """Process the loaded data and prepare features."""
        # Extract relevant columns
        self.raw_text = self.data["text"]
        preprocessed_text = self.data["text"]#.apply(self._preprocess_text)
        self.cleaned_text = np.asarray(preprocessed_text)
        self.is_spam = self.data["label_num"]
        self.text_class = self.data["classification"]

    def get_processed_data(self, raw=False, concat=False):
        """
        Get all processed data.

        Returns:
            tuple: (cleaned_text, is_spam, text_class)
        """
        if raw:
            return self.raw_text, self.is_spam, self.text_class

        # Return processed data
        if concat:
            # Concatenate cleaned text into a single string
            concat_cleaned_text = [" ".join(text) for text in self.cleaned_text]
            return np.asarray(concat_cleaned_text), self.is_spam, self.text_class

        return self.cleaned_text, self.is_spam, self.text_class
    
    def get_processed_text_as_strings(self):
        """
        Get processed text as a list of strings.
        
        Returns:
            list: List where each element is a space-joined string of tokens
        """
        processed_text = [' '.join(tokens) for tokens in self.cleaned_text]
        return np.asarray(processed_text)

    def get_raw_dataframe(self):
        """
        Get raw dataframe.

        Returns:
            pandas.DataFrame: Raw data
        """
        return self.data

    def get_raw_text(self):
        """
        Get raw text.

        Returns:
            list: List of raw text
        """
        return self.raw_text


class DataPaths:
    """Class to store data file paths."""

    SPAM_DATA = "classified_spam.csv"
    SPAM_SALES_DATA = "classified_spam+sales.csv"
    
class DataEncodings(Enum):
    COUNT = "count"
    TFIDF = "tfidf"
    WORD2VEC = "word2vec"
    TOKENIZE = "tokenize"
    
class DataEncoder:
    """Class to encode data for training."""
    
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        
    def _sparse_tensor_to_csr_matrix(self, sparse_tensor):
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        shape = sparse_tensor.dense_shape
        return sp.csr_matrix((values, (indices[:, 0], indices[:, 1])), shape=shape)

    def vectorize_text_data(self, data, encoding: DataEncodings = DataEncodings.COUNT, sparse = False, csr_matrix=False):
        """
        Vectorize text data using TensorFlow's TextVectorization layer.
        Args:
            data (array-like): Text data to vectorize
        Returns:
            tf.keras.layers.TextVectorization: Vectorized text data
        """
        encoder = None
        if encoding == DataEncodings.COUNT:
            encoder = TextVectorization(
                max_tokens=self.vocab_size,
                ngrams=(1,1),
                output_mode="count",
                sparse=sparse
            )
        elif encoding == DataEncodings.TFIDF:
            encoder = TextVectorization(
                max_tokens=self.vocab_size,
                output_mode="tf_idf",
                sparse=sparse
            )
            
        elif encoding == DataEncodings.TOKENIZE:
            encoder = TextVectorization(
                max_tokens=self.vocab_size,
                output_mode="int",
                sparse=sparse
            )
        encoder.adapt(data)
        if sparse and csr_matrix:
            return self._sparse_tensor_to_csr_matrix(encoder(data))
        return encoder(data)


def cross_validate_split(data, labels, test_size=0.2):
    """
    Split data into training and testing sets.
    Args:
        data (array-like): Feature data
        labels (array-like): Labels
        test_size (float): Proportion of the dataset to include in the test split
    Returns:
        tuple: (X_train, X_test, Y_train, Y_test)
    """
    return train_test_split(data, labels, test_size=test_size, random_state=10)




# Usage example:
if __name__ == "__main__":
    # Initialize data loader
    data_loader = DataLoader()

    # Load spam data
    spam_data = data_loader.load_data(DataPaths.SPAM_DATA)

    # Get processed data
    cleaned_text, is_spam, text_class = data_loader.get_processed_data()

    # Print sample of data
    print("Sample of processed data:")
    print(spam_data[["text", "label_num", "classification"]].head())
