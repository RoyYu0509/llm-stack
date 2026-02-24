from cs336_basics.bpe_tokenizer.tokenizeInterface import AbstractPreTokenizer
from cs336_basics.bpe_tokenizer.helpers.bpe_helper import *
import regex as re
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from tqdm import tqdm
import time
ECHO = False
from typing import Dict, List, Tuple


class BBPE(AbstractPreTokenizer):
    """
    A Byte-Pair Encoding (BPE) Tokenizer. 
    Starting with 256 basic bytes, iteratively merging frequent bytes pair to create new vocabulary from the basic bytes.



    Instance Variables:
    ===================
    - byte_2_id:        Dictionary stores the Bytes Vocabulary to ID.
                        eg. {b"l": 23, 
                             b"o": 71, 
                             b"w": 93}

    - byte_2_id_size:   The number of stored Vocabulary Bytes.

    - id_2_bytes:       Dict[int:byte] Dictionary stores ID to the Vocabulary Bytes
                        eg. {23: b"l", ...}

    - pretok_dict:      Counter[Tuple(bytes):int] Counter that counts each Pretokenized Subtring (represented in the Bytes Vocabulary's ID) to its Count
                        eg. "low low" -> ["low", "low"]
                                      -> [b"l", b"o", b"w"], [b"l", b"o", b"w"] 
                                      -> [23,71,93], [23,71,93]
                                      -> pretok_dict = {(23,71,93): 2}

    - freq_dict:        Dict[Tuple(bytes, bytes): int] Dictionary stores the frequence of each pretoken pair.
                        eg. freq_dict = {(23,71): 2, (71,93):2} <- pretok_dict = {[23,71,93]: 2}

    - special_tokens:   List[str] A list of special tokens that should not be break into sinlge bytes while pretokenizaiotn and encoding

    - merge_sequence:   List[Tuple(bytes, bytes)] A List of bytes tuple, recording the resulted sequence of merging bytes pairs from training phase.
    """
    def __init__(self, max_vocab_size, special_tokens):
        super().__init__()
        
        self.byte_2_id = intialize_b2id_dict(special_tokens)
        self.byte_2_id_size = len(list(self.byte_2_id.keys()))
        self.id_2_bytes = {v: k for k, v in self.byte_2_id.items()}
        self.pretok_dict = Counter()
        self.freq_dict = {}
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [sp_tok.encode("utf-8") for sp_tok in special_tokens] if special_tokens else []
        self.merge_sequence = []

        # Define string splitting function
        self.PAT = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Define safe partition skipping all special_tokens
        if self.special_tokens:
            split_pattern = b"(" + b"|".join(
                re.escape(spt.encode("utf-8")) for spt in self.special_tokens
            ) + b")"
            self.SPLIT = re.compile(split_pattern)
        else:
            self.SPLIT = None

        # Define Speical Tokens Split: [non-special, special, non-special, ..., non-special, special]
        self.SPECIAL_RE = re.compile(
            f"({"|".join(sorted(map(re.escape, self.special_tokens), key=len, reverse=True))})"
        ) if self.special_tokens else re.compile("$^")

    """
    Build PreTokenizer
    """
    def pretokenization(
            self, 
            file_path: str, 
            desired_num_chunks: int, 
            split_special_token: bytes
        ):
        """
        Return a pretokenization dict counts the pretoken appearance: 
            {
            counter[(byte1, byte2, ...)]: <count>, 
            counter[(b"...")]: <count>, 
            ...
            }
        Each Pretoken is stored in tuple of single bytes representation (byte1, byte2, ...).
        Read the file_path in several splits, splited based on the split_special_token.

        NOTE: For the Special Tokens we keep it as one token: counter[(b"<|endoftext|>",)] = <count>
            
        Precondition: The bytes in the text must in the default_vocab
        """
        # Compute the boundaries for multiprocessing
        with open(file_path, "rb") as f:
            num_processes = desired_num_chunks
            boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

        # Parallelizing 
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            boundaries = list(zip(boundaries[:-1], boundaries[1:])) # eg. zip([0, 100, 200, 300], [100, 200, 300, 400]) produces (0, 100), (100, 200), (200, 300), (300, 400)
            chunk_pretks = [
                executor.submit(process_chunk, boundary, file_path, self.SPLIT, self.PAT, self.special_tokens_bytes) 
                for boundary in boundaries
            ]
        
        # Aggregate the counts from all processes
        for future in tqdm(as_completed(chunk_pretks), total=len(chunk_pretks),
                           desc="Pretokenizing chunks", unit="chunk"):
            counter = future.result()
            if not counter:
                print(f"Warning: empty counter from chunk")
            self.pretok_dict.update(counter)

    """
    Train Tokenizer
    """
    def train(self, input_path, vocab_size,
              num_processes = 3, split_special_token = b"<|endoftext|>") -> Tuple[Dict, List]:
        """
        Train a BPE Tokenizer, return the resulted vocabular and merges sequence.

        Parameters:
            input_path: str Path to a text file with BPE tokenizer training data.

            vocab_size: int A positive integer that defines the maximum final vocabulary size 
                            (including the initial byte vocabulary, vocabulary items produced 
                            from merging, and any special tokens).
            
            special_tokens: list[str] A list of strings to add to the vocabulary. These 
                                      special tokens do not otherwise affect BPE training.
        """
        start = time.time()
        print(f"Building Pretokenizer and Frequency Dictionary")
        self.pretokenization(input_path, num_processes, split_special_token)
        self._build_freq_dict()
        end = time.time()
        print(f"Finished building Pretokenizer and Frequency Dictionary in {end - start:.2f} seconds")

        merges_sequence = []

        start_size = self.byte_2_id_size
        total_merges_needed = max(0, vocab_size - start_size)
        pbar = tqdm(total=total_merges_needed, desc="Merging BPE pairs", unit="merge")

        # Keep merging to collect vocabulary
        while self.byte_2_id_size < vocab_size:
            merged_bts = self._merge()  # Both b2id & id2b mappings are updated during merges
            merges_sequence.append(merged_bts)
            pbar.update(1)

        pbar.close()
        self.merge_sequence = merges_sequence
        return self.id_2_bytes, merges_sequence

    def _build_freq_dict(self):
        """
        Based on the stored pretokenization dictionary, build 
        the byte level pair-wise frequency dictionary.
        """
        # Initialize a new freq dict 
        self.freq_dict = {}

        # Go through the pairs
        for byte_id_seq, count in self.pretok_dict.items():
            # Skip the byte_id_seq have at least two elements to form a pair
            if len(byte_id_seq) < 2:
                # print(f"The sting {bytes([byte_id_seq[0]])} byte sequence has length {len(byte_id_seq)}, less than 2")
                continue
            # Count the pair: {(l,o,w): 7} -> {b"lo": 7, b"ow": 7}
            
            for curr, next in zip(byte_id_seq[:-1], byte_id_seq[1:]):
                self.freq_dict[(curr, next)] = self.freq_dict.get((curr, next), 0) + count
        # If there is no pair to merge
        if not self.freq_dict:
            print("There is no byte pair to merge!")

    def _merge(self) -> tuple[bytes, bytes]:
        """
        Find the most frequent pair of bytes from the frequency dictionary 
        and Merge them into one vocab.

        DO:
            1. Add the merged token into vocabulary (id_2_byte/byte_2_id). 
                eg. >>> bbpe.merge()
                    b"l", b"ow" -> b"low"
                    >>> id_2_byte[NEW_ID] = b"low"
                    >>> byte_2_id[b"low"] = NEW_ID
            2. Update the pretokenization dictionary based on the new vocabulary.
                >>> # pretokenization = {(b"l", b"ow"):COUNT}
                >>> UPDATE# pretokenization = {(b"low"):COUNT} 
        """
        # Find the most frequent pair in our constructed frequency pair
        most_freq_pairs = []
        max_freq = max(self.freq_dict.values())
        for pair, freq in self.freq_dict.items():
            if freq == max_freq:
                most_freq_pairs.append(pair)
        merged_pair = max(most_freq_pairs)

        # print(merged_pair)
        
        # Update the bytes representation in pretokenization dictionary
        bt1, bt2 = merged_pair

        # Use id to look up the actual pair
        merged_bytes = bt1 + bt2 # b"l" + b"o" -> b"lo"
        # print(self.pretok_dict)
        new_pretok_dict = Counter()
        for byte_id_seq, count in self.pretok_dict.items():
            # print(byte_id_seq)
            new_seq = []
            i = 0
            while i < len(byte_id_seq):
                if i+1 < len(byte_id_seq) and byte_id_seq[i] == bt1 and byte_id_seq[i+1] == bt2:
                    new_seq.append(merged_bytes)
                    i+=2
                else:
                    new_seq.append(byte_id_seq[i])
                    i+=1
            # Add the new merged bytes representation
            new_pretok_dict[tuple(new_seq)] = count

        # Update the pretokenization dictionary
        self.pretok_dict = new_pretok_dict
        
        # Add it to our vocabulary
        next_id = self.byte_2_id_size
        self.byte_2_id[merged_bytes] = next_id
        self.id_2_bytes[next_id] = merged_bytes
        self.byte_2_id_size += 1


        # Rebuild the Freqeuncy dictionary
        self._build_freq_dict()

        return (bt1, bt2)
    
    
    """
    Input Text Encoder
    """
    def encoding(self, text:str):
        """
        Return a List of bytes ID represents the encoded text.

        The IDs are get from merging the text's single bytes accroding to the stored merge_sequence.
        """
        if not self.merge_sequence:
            raise ValueError("Must obtain a merge sequence before encoding.")
        
        rslt = []
        
        # Use a Iterable to stream the pretoken (processed pretoken by pretoken)
        pretok_stream_iterable = self._text_2_pretoken_iterator(text)

        # Encode the pretoken byte sequence
        for pretok in pretok_stream_iterable:
            if ECHO:
                print(f"Processing Pretoken {pretok}")
            # Look up the merges
            merge = self._apply_merges(pretok)
            for single_byte in merge:
                rslt.append(self.byte_2_id[single_byte])
        return rslt

    def _text_2_pretoken_iterator(self, text: str):
        """
        Convert a sequence of text to a list of bytes, perserving special tokens
        """
        # Split into: non-special, special, non-special, ...
        parts = re.split(self.SPECIAL_RE, text)
        for part in parts:
            if part == "" or part is None:  # Skip empty
                continue
            if part in self.special_tokens: # Perserve single bytes for special token
                yield [part.encode("utf-8")]
            else:
                # normal text → pretokenize with PAT, then down to single bytes
                for m in self.PAT.finditer(part):
                    s = m.group().encode("utf-8")
                    yield [bytes([b]) for b in s]
    
    def _apply_merges(self, pretok_sequence: list[bytes]):
        """
        Return a List representation of the pretok_sequence after apply merging.
        """
        # Special Case: length 1 pretoken sequence
        if len(pretok_sequence) == 1:
            return pretok_sequence

        prev_encode = pretok_sequence
        # Encode the byte sequence using self.merges seqeuence
        for merge in self.merge_sequence:
            i = 0
            j = 0
            curr_encode = []
            while i < len(prev_encode):
                if prev_encode[i] in self.special_tokens_bytes:
                    curr_encode.append(prev_encode[i])
                    i += 1
                    j += 1

                # If we see a matching merging byte pair
                elif i+1<len(prev_encode) and prev_encode[i] == merge[0] and prev_encode[i+1] == merge[1]:
                    if ECHO:
                        print(f"Applying merge: {merge} to the pretoken sequence")
                    curr_encode.append(prev_encode[i] + prev_encode[i+1])
                    i += 2
                    j += 1

                # If not merge
                else:
                    curr_encode.append(prev_encode[i])
                    i += 1
                    j += 1
            prev_encode = curr_encode
        return curr_encode


    """
    Decoder
    """
    def decoding(self, byte_id_seq: list[int]):
        """Decode a bytes sequence back to string using tokenizer's vocabulary"""
        # Build byte chunks first
        byte_chunks = []
        for tid in byte_id_seq:
            bs = self.id_2_bytes.get(tid)
            if bs is None:
                # U+FFFD in UTF-8
                byte_chunks.append(b"\xEF\xBF\xBD")
            else:
                byte_chunks.append(bs)

        # Decode once so multi-byte sequences across token boundaries are handled
        return (b"".join(byte_chunks)).decode("utf-8", errors="replace")
