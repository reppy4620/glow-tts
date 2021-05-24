class Tokenizer:
    def __init__(self, dictionary_path='./filelists/word_index.txt'):
        self.dictionary = self.load_dictionary(dictionary_path)
        self.coef_a1 = 15

    def __call__(self, phonemes, a1s, f2s):
        phoneme_ids = [self.dictionary[s] for s in phonemes.split('-')]
        a1s = [float(a1) / self.coef_a1 for a1 in a1s.split('_')]
        f2s = [int(f2) for f2 in f2s.split('_')]
        return phoneme_ids, a1s, f2s

    @staticmethod
    def load_dictionary(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        dictionary = dict()
        for i, w in enumerate([w.strip() for w in lines]):
            dictionary[w] = i
        return dictionary

    def __len__(self):
        return len(self.dictionary)
