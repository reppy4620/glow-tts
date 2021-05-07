class Tokenizer:
    def __init__(self, dictionary_path):
        self.dictionary = self.load_dictionary(dictionary_path)

    def tokenize(self, phonemes):
        return [self.dictionary[s] for s in phonemes.split('-')]

    def load_dictionary(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return [w.strip() for w in lines]

    def __len__(self):
        return len(self.dictionary)
