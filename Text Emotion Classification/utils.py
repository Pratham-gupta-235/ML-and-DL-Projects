import re
import collections
import numpy as np

def pad_sequences(sequences, maxlen):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded_seq = seq[:maxlen]
        else:
            padded_seq = [0] * (maxlen - len(seq)) + seq
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

# Simple tokenizer class to replace Keras tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.word_counts = collections.Counter()
        
    def fit_on_texts(self, texts):
        # Create vocabulary from texts
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            self.word_counts.update(words)
        
        # Create word_index
        for i, (word, _) in enumerate(self.word_counts.most_common()):
            self.word_index[word] = i + 1  # Start at 1
            self.index_word[i + 1] = word
        
        return self
        
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            seq = [self.word_index.get(word, 0) for word in words]
            sequences.append(seq)
        return sequences

# Emoji map for better output
emoji_map = {
    'happy': '😊', 
    'sad': '😢', 
    'angry': '😠', 
    'surprised': '😲',
    'neutral': '😐',
    'fear': '😨',
    'disgust': '🤢'
}

# Color map for emotions
color_map = {
    'happy': '#28a745',
    'sad': '#6c757d',
    'angry': '#dc3545',
    'surprised': '#17a2b8',
    'neutral': '#6c757d',
    'fear': '#6610f2',
    'disgust': '#fd7e14'
}
