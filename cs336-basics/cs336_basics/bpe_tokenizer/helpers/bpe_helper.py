import os
from typing import BinaryIO
import regex as re
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from tqdm import tqdm
import time

BYTES = [bytes([i]) for i in range(256)]
def bytes_2_tuple(bts: bytes):
    return tuple(BYTES[b] for b in bts)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    ) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.

    We initialize the boundary guess and start at each boundary guess to find
    the actual position of the nearest `split_special_token`.

    For each iteration, we read a small mini_chunk, look for the split token, 
    and if we don’t find it, we move the pointer forward and keep searching 
    until we do (or hit EOF).
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def intialize_b2id_dict(special_tokens):
    byte_2_id = {}
    i = 0
    if special_tokens is not None:
        for special_token in special_tokens:
            byte_2_id[special_token.encode("utf-8")] = i
            i+=1
    for j in range(256):
        byte_2_id[bytes([j])] = i
        i+=1
    return byte_2_id

# Parallelize the processing
def process_chunk(start_end:tuple[int, int], file_path, split, pat, special_tokens_bytes):
    """
    Process the text bytes file in the chunk [start:end]

    Return the pretokenizer counter for this chunk
    """
    with open(file_path, "rb") as f:
        # build counter
        counter = Counter()
        start = start_end[0]
        end = start_end[1]

        # Process each chunk
        f.seek(start)
        chunk = f.read(end - start)

        # Pre-split based on a list of special tokens
        if split is not None:
            safe_chunks = split.split(chunk)  # Prevent split on the middle of the special tokens
        else:
            safe_chunks = [chunk]

        for safe_chunk in safe_chunks:
            # Skip the empty chunk produced by `safe_chunks`-splitting (Side Effect)
            if not safe_chunk:  # skip empty
                continue
            if special_tokens_bytes and safe_chunk in special_tokens_bytes:
                counter[(safe_chunk,)] += 1
                continue

            # Apply string splitting and get an sting Iterable
            # String"some text" -> Iterable['some', ' text']
            for match in re.finditer(pat, safe_chunk.decode("utf-8", errors="ignore")): # Ignore the
                # subtring (b"low) -> (b"l", b"o", b"w")
                substr_enc = match.group().encode("utf-8")
                tokens = bytes_2_tuple(substr_enc)
                # Record the appearance of a bytes representation:  {(b"l", b"o", b"w"): 5}
                counter[tokens] += 1
        return counter
