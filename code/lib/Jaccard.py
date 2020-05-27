import nltk
import numpy as np

class Jaccard:
    """
    Jaccard Index applied to two words. Example:
    jaccard('one', 'onna') -> 0.5
    """
    def word_jaccard(self, w1, w2):
        cw1 = [char.lower() for char in w1]
        cw2 = [char.lower() for char in w2]
        
        c = max(len(cw1), len(cw2)) - abs(nltk.edit_distance(w1, w2))
        
        return float(c) / (len(cw1) + len(cw2) - c)
    
    """
    Jaccard Index applied to pseudo-binary sequences
    bin_jaccard([[1,0,0], [1,0,1]], [[1,0,1], [1,1,1]])
    """
    def bin_jaccard(self, y_true, y_pred):
        intersection = y_true * y_pred
        # Necessary step if the content of y_true and y_pred are items with differents lengths
        a = np.sum([e for s in y_true for e in s])
        b = np.sum([e for s in y_pred for e in s])
        c = np.sum([e for s in intersection for e in s])
        return float(c) / (a + b - c)
    