import math, os, pickle, re
from typing import Tuple, List, Dict


class BayesClassifier:
    """A simple BayesClassifier implementation

    Attributes:
        pos_freqs - dictionary of frequencies of positive words
        neg_freqs - dictionary of frequencies of negative words
        pos_filename - name of positive dictionary cache file
        neg_filename - name of negative dictionary cache file
        training_data_directory - relative path to training directory
        neg_file_prefix - prefix of negative reviews
        pos_file_prefix - prefix of positive reviews
    """

    def __init__(self):
        """Constructor initializes and trains the Naive Bayes Sentiment Classifier. If a
        pickled version of a trained classifier is stored in the current folder it is loaded,
        otherwise the system will proceed through training.  Once constructed the
        classifier is ready to classify input text."""
        # initialize attributes
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.pos_filename: str = "pos.dat"
        self.neg_filename: str = "neg.dat"
        self.training_data_directory: str = "movie_reviews/"
        self.neg_file_prefix: str = "movies-1"
        self.pos_file_prefix: str = "movies-5"

        # check if both cached classifiers exist within the current directory
        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
            print("Data files found - loading to use pickled values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
        else:
            print("Data files not found - running training...")
            self.train()

    def train(self) -> None:
        """Trains the Naive Bayes Sentiment Classifier

        Train here means generates 'self.pos_freqs' and 'self.neg_freqs' dictionaries with frequencies of
        words in corresponding positive/negative reviews
        """
        files: List[str] = next(os.walk(self.training_data_directory))[2]

        for index, filename in enumerate(files, 1):
            print(f"Training on file {index} of {len(files)}")
            text = self.load_file(os.path.join(self.training_data_directory, filename))
            tokenlist = self.tokenize(text)
            if filename.startswith(self.pos_file_prefix):
                self.update_dict(tokenlist, self.pos_freqs)
            else:
                self.update_dict(tokenlist, self.neg_freqs)

    def classify(self, text: str) -> str:
        posprob = 0
        negprob = 0
        sumofwordspos = 0
        sumofwordsneg = 0

        texttoken = self.tokenize(text)

        for word in self.pos_freqs.keys():
            sumofwordspos += self.pos_freqs[word]

        for word in self.neg_freqs.keys():
            sumofwordsneg += self.neg_freqs[word]

        for token in texttoken:
            if token in self.pos_freqs:
                posprob += math.log((self.pos_freqs[token]+1)/sumofwordspos)
            else:
                posprob += math.log((1)/ sumofwordspos)

            if token in self.neg_freqs:
                negprob += math.log((self.neg_freqs[token]+1)/ sumofwordsneg)
            else:
                negprob += math.log((1)/ sumofwordsneg)

        if posprob > negprob:
            return "positive"
        else:
            return "negative"

    def load_file(self, filepath: str) -> str:

        with open(filepath, "r", encoding='utf8') as f:
            return f.read()

    def save_dict(self, dict: Dict, filepath: str) -> None:
        """Pickles given dictionary to a file with the given name
        """
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> Dict:
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())
        return tokens

    def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
        for word in words:
            if word in freqs.keys():
                freqs[word] += 1
            else:
                freqs[word] = 1
