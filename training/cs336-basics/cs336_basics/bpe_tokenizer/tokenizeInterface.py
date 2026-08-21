from abc import ABC, abstractmethod

class AbstractPreTokenizer(ABC):
    def __init__(self):
        """Initialize the defaul vocabulary."""
        # self.default_vocab = default_vocab
        # self.count_dict = {}
    

    @abstractmethod
    def pretokenization(self, text):
        """
        Return a pretokenization dict: 
        {
            (byte1, byte2, ...): appearance, 
            (): appearance, 
            ...
        }
        
        eg. {
            (l,o,w): 5, 
            (l,o,w,e,r): 2, 
            (w,i,d,est): 3, 
            (n,e,w,est): 6
            }
        """
        pass

    @abstractmethod
    def _merge(self):
        pass


    @abstractmethod
    def train(self, text):
        pass

