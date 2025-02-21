class Tokenizer:
    def __init__(self, dictionary_path='./filelists/word_index.txt'):
        self.dictionary = self.load_dictionary(dictionary_path)

    def __call__(self, phonemes):
        return [self.dictionary[s] for s in phonemes.split('-')]

    def load_dictionary(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        dictionary = dict()
        for i, w in enumerate([w.strip() for w in lines]):
            dictionary[w] = i
        return dictionary

    def __len__(self):
        return len(self.dictionary)
