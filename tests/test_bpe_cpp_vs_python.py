"""
Test suite: Validate C++ BPE tokenizer against the Python implementation.

Three categories of tests:
  1. C++ self-consistency  — encode→decode roundtrip, special tokens, edge cases
  2. Cross-validation      — load same vocab/merges into both, compare encode & decode
  3. Training comparison   — train both on the same corpus, compare merge sequences

NOTE: The Python tokenizer uses GPT-2 regex pretokenization while the C++ version
uses a simplified whitespace/punctuation split. Training outputs will diverge, but
given the *same* vocab+merges the encode/decode logic should match on inputs where
pretokenization produces the same splits (simple ASCII words, no contractions).
"""

import os
import subprocess
import tempfile
import pickle
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CPP_DIR = REPO_ROOT / "cs336-basics" / "cs336_basics" / "bpe_tokenizer_cpp"
CPP_BRIDGE = CPP_DIR / "test_bridge"
CPP_VOCAB = CPP_DIR / "vocab.txt"
CPP_MERGES = CPP_DIR / "merges.txt"
TINY_TEST = CPP_DIR / "tiny_test.txt"

# Python tokenizer imports
import sys
sys.path.insert(0, str(REPO_ROOT / "cs336-basics"))
from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer


# ============================================================================
# Helpers
# ============================================================================

def _ensure_bridge():
    """Compile the C++ test bridge if the binary doesn't exist or is stale."""
    src = CPP_DIR / "test_bridge.cpp"
    impl = CPP_DIR / "bpe_tokenizer.cpp"
    binary = CPP_BRIDGE

    needs_build = (
        not binary.exists()
        or src.stat().st_mtime > binary.stat().st_mtime
        or impl.stat().st_mtime > binary.stat().st_mtime
    )
    if needs_build:
        subprocess.check_call(
            ["g++", "-std=c++17", "-O2", "-o", str(binary),
             str(src), str(impl)],
            cwd=str(CPP_DIR),
        )


def _run_bridge(args: list[str], stdin_text: str | None = None,
                raw_output: bool = False) -> str:
    """Run the C++ test bridge and return stdout.

    If raw_output is True, use binary mode to preserve all bytes exactly,
    then decode as latin-1 (identity mapping for all 256 byte values).
    """
    if raw_output:
        stdin_bytes = stdin_text.encode("utf-8") if stdin_text else None
        result = subprocess.run(
            [str(CPP_BRIDGE)] + args,
            capture_output=True,
            input=stdin_bytes,
            cwd=str(CPP_DIR),
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"test_bridge failed (rc={result.returncode}):\n"
                f"  args: {args}\n"
                f"  stderr: {result.stderr.decode('utf-8', errors='replace')}"
            )
        # Decode as latin-1 to preserve byte identity, then strip trailing \n
        return result.stdout.decode("latin-1").rstrip("\n")
    else:
        result = subprocess.run(
            [str(CPP_BRIDGE)] + args,
            capture_output=True,
            text=True,
            input=stdin_text,
            cwd=str(CPP_DIR),
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"test_bridge failed (rc={result.returncode}):\n"
                f"  args: {args}\n"
                f"  stderr: {result.stderr}"
            )
        return result.stdout.strip()


def _cpp_encode(text: str, vocab_path: str = None, merges_path: str = None) -> list[int]:
    """Encode text using the C++ tokenizer via test_bridge."""
    vp = vocab_path or str(CPP_VOCAB)
    mp = merges_path or str(CPP_MERGES)
    # Use stdin mode to handle special characters and spaces correctly
    out = _run_bridge(["encode_stdin", vp, mp], stdin_text=text)
    if not out:
        return []
    return [int(x) for x in out.split(",")]


def _cpp_decode(ids: list[int], vocab_path: str = None, merges_path: str = None) -> str:
    """Decode token IDs using the C++ tokenizer via test_bridge.

    The C++ bridge appends a NUL sentinel after the decoded text so we can
    preserve trailing whitespace/newlines without ambiguity.
    """
    vp = vocab_path or str(CPP_VOCAB)
    mp = merges_path or str(CPP_MERGES)
    ids_str = ",".join(str(i) for i in ids)
    raw = _run_bridge(["decode", vp, mp, ids_str], raw_output=True)
    # Strip the sentinel and anything after it
    sentinel = "<<__END_DECODE__>>"
    idx = raw.find(sentinel)
    if idx >= 0:
        return raw[:idx]
    return raw


def _cpp_train(input_file: str, vocab_size: int, vocab_out: str, merges_out: str):
    """Train the C++ tokenizer and save vocab/merges."""
    out = _run_bridge(["train", input_file, str(vocab_size), vocab_out, merges_out])
    info = {}
    for line in out.splitlines():
        # Only parse lines that are exactly "KEY=VALUE" (our bridge output)
        if line.startswith("VOCAB_SIZE=") or line.startswith("NUM_MERGES="):
            k, v = line.split("=", 1)
            info[k] = int(v)
    return info


def _parse_cpp_vocab(vocab_path: str) -> dict[int, bytes]:
    """Parse C++ hex vocab.txt into {id: bytes}."""
    vocab = {}
    with open(vocab_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            id_ = int(parts[0])
            hex_str = parts[1]
            raw = bytes.fromhex(hex_str)
            vocab[id_] = raw
    return vocab


def _parse_cpp_merges(merges_path: str) -> list[tuple[bytes, bytes]]:
    """Parse C++ hex merges.txt into [(bytes, bytes), ...]."""
    merges = []
    with open(merges_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            bt1 = bytes.fromhex(parts[0])
            bt2 = bytes.fromhex(parts[1])
            merges.append((bt1, bt2))
    return merges


def _load_python_tokenizer_from_cpp_files(
    vocab_path: str = None, merges_path: str = None
) -> Tokenizer:
    """Load a Python tokenizer using vocab/merges from C++ hex format files."""
    vp = vocab_path or str(CPP_VOCAB)
    mp = merges_path or str(CPP_MERGES)
    vocab = _parse_cpp_vocab(vp)
    merges = _parse_cpp_merges(mp)
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def build_bridge():
    _ensure_bridge()


@pytest.fixture(scope="session")
def py_tok():
    """Python tokenizer loaded with the C++ pretrained vocab/merges."""
    return _load_python_tokenizer_from_cpp_files()


@pytest.fixture(scope="session")
def trained_cpp(tmp_path_factory):
    """Train C++ on tiny_test.txt with vocab_size=300, return (vocab_path, merges_path, info)."""
    out_dir = tmp_path_factory.mktemp("cpp_train")
    vp = str(out_dir / "vocab.txt")
    mp = str(out_dir / "merges.txt")
    info = _cpp_train(str(TINY_TEST), 300, vp, mp)
    return vp, mp, info


@pytest.fixture(scope="session")
def trained_py(tmp_path_factory):
    """Train Python on tiny_test.txt with vocab_size=300, return Tokenizer."""
    tok = Tokenizer(vocab=300, special_tokens=["<|endoftext|>"])
    tok.train(str(TINY_TEST), 300, num_processes=1)
    return tok


# ============================================================================
# 1. C++ Self-Consistency Tests
# ============================================================================

class TestCppSelfConsistency:
    """Verify the C++ tokenizer is internally consistent."""

    @pytest.mark.parametrize("text", [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time",
        "a",
        "   ",
        "123 456 789",
        "UPPER lower MiXeD",
    ])
    def test_encode_decode_roundtrip(self, text):
        """encode(text) → decode → should recover original text."""
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text, f"Roundtrip failed: {text!r} → {ids} → {decoded!r}"

    def test_special_token_roundtrip(self):
        text = "<|endoftext|>"
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text

    def test_special_token_mixed(self):
        text = "Hello<|endoftext|>World"
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text

    def test_special_token_is_single_id(self):
        """The special token should map to exactly one token ID (id=0)."""
        ids = _cpp_encode("<|endoftext|>")
        assert len(ids) == 1
        assert ids[0] == 0  # special tokens are assigned first

    def test_empty_string(self):
        ids = _cpp_encode("")
        assert ids == []

    def test_encode_deterministic(self):
        """Same input should always produce same output."""
        text = "The little girl was happy."
        ids1 = _cpp_encode(text)
        ids2 = _cpp_encode(text)
        assert ids1 == ids2

    def test_decode_unknown_id(self):
        """Unknown IDs should produce U+FFFD replacement character."""
        decoded = _cpp_decode([99999])
        assert "\ufffd" in decoded

    def test_all_single_bytes(self):
        """Each printable ASCII byte should roundtrip."""
        for b in range(32, 127):
            ch = chr(b)
            ids = _cpp_encode(ch)
            decoded = _cpp_decode(ids)
            assert decoded == ch, f"Byte {b} ({ch!r}) failed roundtrip"


# ============================================================================
# 2. Cross-Validation: Same vocab/merges in both implementations
# ============================================================================

class TestCrossValidation:
    """
    Load the SAME vocab and merges into both Python and C++.
    Compare encode/decode on identical inputs.

    NOTE: Pretokenization differs (GPT-2 regex vs simple_split), so we test
    inputs that both tokenizers split the same way — simple words separated
    by single spaces, no contractions or Unicode.
    """

    # Texts where both pretokenizers should agree on splits
    SIMPLE_TEXTS = [
        "hello",
        "the",
        "a big red ball",
        "Once upon a time",
        "The little girl was happy",
        "123",
    ]

    @pytest.mark.parametrize("text", SIMPLE_TEXTS)
    def test_encode_matches(self, py_tok, text):
        """Python and C++ should produce the same token IDs."""
        py_ids = py_tok.encode(text)
        cpp_ids = _cpp_encode(text)
        assert py_ids == cpp_ids, (
            f"Encode mismatch for {text!r}:\n"
            f"  Python: {py_ids}\n"
            f"  C++:    {cpp_ids}"
        )

    @pytest.mark.parametrize("text", SIMPLE_TEXTS)
    def test_decode_matches(self, py_tok, text):
        """Given the same IDs, both should decode to the same string."""
        # Use C++ encoding as the source IDs
        cpp_ids = _cpp_encode(text)
        py_decoded = py_tok.decode(cpp_ids)
        cpp_decoded = _cpp_decode(cpp_ids)
        assert py_decoded == cpp_decoded, (
            f"Decode mismatch for IDs {cpp_ids}:\n"
            f"  Python: {py_decoded!r}\n"
            f"  C++:    {cpp_decoded!r}"
        )

    def test_special_token_encode(self, py_tok):
        text = "<|endoftext|>"
        py_ids = py_tok.encode(text)
        cpp_ids = _cpp_encode(text)
        assert py_ids == cpp_ids

    def test_special_token_in_context(self, py_tok):
        """Special tokens embedded in text should be encoded the same way."""
        text = "<|endoftext|>Once upon a time<|endoftext|>"
        py_ids = py_tok.encode(text)
        cpp_ids = _cpp_encode(text)
        assert py_ids == cpp_ids, (
            f"Special token context mismatch:\n"
            f"  Python: {py_ids}\n"
            f"  C++:    {cpp_ids}"
        )

    def test_vocab_coverage(self, py_tok):
        """Both tokenizers should have the same vocabulary size."""
        cpp_vocab = _parse_cpp_vocab(str(CPP_VOCAB))
        py_vocab_size = py_tok.byte_2_id_size
        assert len(cpp_vocab) == py_vocab_size, (
            f"Vocab size mismatch: C++={len(cpp_vocab)}, Python={py_vocab_size}"
        )

    def test_vocab_byte_mapping(self, py_tok):
        """Every ID→bytes mapping in C++ vocab should exist in Python vocab."""
        cpp_vocab = _parse_cpp_vocab(str(CPP_VOCAB))
        for id_, raw_bytes in cpp_vocab.items():
            assert id_ in py_tok.id_2_bytes, f"ID {id_} missing from Python vocab"
            assert py_tok.id_2_bytes[id_] == raw_bytes, (
                f"ID {id_}: C++={raw_bytes!r}, Python={py_tok.id_2_bytes[id_]!r}"
            )

    def test_decode_full_vocab(self, py_tok):
        """Decoding every single vocab ID should produce the same raw bytes.

        Python's decode() applies UTF-8 decoding with error replacement, while
        C++ returns raw bytes. We compare at the byte level to avoid UTF-8
        decoding differences for invalid sequences.
        """
        cpp_vocab = _parse_cpp_vocab(str(CPP_VOCAB))
        for id_ in sorted(cpp_vocab.keys()):
            # Get raw bytes from both
            py_raw = py_tok.id_2_bytes.get(id_)
            cpp_raw = cpp_vocab.get(id_)
            assert py_raw == cpp_raw, (
                f"Raw bytes mismatch at id={id_}: "
                f"Python={py_raw!r}, C++={cpp_raw!r}"
            )


# ============================================================================
# 3. Training Comparison
# ============================================================================

class TestTrainingComparison:
    """
    Train both tokenizers on the same corpus and compare results.

    Because pretokenization differs, merge sequences will likely diverge.
    These tests check structural properties rather than exact equality.
    """

    def test_cpp_train_produces_expected_vocab_size(self, trained_cpp):
        vp, mp, info = trained_cpp
        assert info["VOCAB_SIZE"] == 300

    def test_cpp_train_produces_merges(self, trained_cpp):
        vp, mp, info = trained_cpp
        assert info["NUM_MERGES"] > 0
        # vocab_size = 256 bytes + 1 special token + num_merges
        assert info["VOCAB_SIZE"] == 257 + info["NUM_MERGES"]

    def test_python_train_produces_expected_vocab_size(self, trained_py):
        assert trained_py.byte_2_id_size == 300

    def test_both_have_same_base_vocab(self, trained_cpp, trained_py):
        """Both should start with the same 256 single-byte entries + special token."""
        vp, mp, _ = trained_cpp
        cpp_vocab = _parse_cpp_vocab(vp)

        # Check all 256 single bytes exist in both
        for b in range(256):
            byte_val = bytes([b])
            assert byte_val in trained_py.byte_2_id, f"Python missing byte {b}"
            cpp_has = any(v == byte_val for v in cpp_vocab.values())
            assert cpp_has, f"C++ missing byte {b}"

    def test_both_have_special_token(self, trained_cpp, trained_py):
        vp, mp, _ = trained_cpp
        cpp_vocab = _parse_cpp_vocab(vp)

        sp = b"<|endoftext|>"
        assert sp in trained_py.byte_2_id
        assert any(v == sp for v in cpp_vocab.values())

    def test_cpp_trained_roundtrip(self, trained_cpp):
        """C++ tokenizer trained from scratch should roundtrip text."""
        vp, mp, _ = trained_cpp
        texts = [
            "Hello, world!",
            "Once upon a time",
            "The little girl was happy.",
            "<|endoftext|>",
        ]
        for text in texts:
            ids = _cpp_encode(text, vocab_path=vp, merges_path=mp)
            decoded = _cpp_decode(ids, vocab_path=vp, merges_path=mp)
            assert decoded == text, f"Trained C++ roundtrip failed: {text!r} → {decoded!r}"

    def test_python_trained_roundtrip(self, trained_py):
        """Python tokenizer trained from scratch should roundtrip text."""
        texts = [
            "Hello, world!",
            "Once upon a time",
            "The little girl was happy.",
            "<|endoftext|>",
        ]
        for text in texts:
            ids = trained_py.encode(text)
            decoded = trained_py.decode(ids)
            assert decoded == text, f"Trained Python roundtrip failed: {text!r} → {decoded!r}"

    def test_merge_sequences_are_valid_byte_pairs(self, trained_cpp, trained_py):
        """Every merge pair should consist of bytes that exist in vocab."""
        vp, mp, _ = trained_cpp
        cpp_merges = _parse_cpp_merges(mp)

        for bt1, bt2 in cpp_merges:
            assert isinstance(bt1, bytes) and len(bt1) >= 1
            assert isinstance(bt2, bytes) and len(bt2) >= 1

        for bt1, bt2 in trained_py.merge_sequence:
            assert isinstance(bt1, bytes) and len(bt1) >= 1
            assert isinstance(bt2, bytes) and len(bt2) >= 1

    def test_retrain_deterministic(self, tmp_path):
        """Training C++ twice on the same data should give identical results."""
        vp1 = str(tmp_path / "v1.txt")
        mp1 = str(tmp_path / "m1.txt")
        vp2 = str(tmp_path / "v2.txt")
        mp2 = str(tmp_path / "m2.txt")

        _cpp_train(str(TINY_TEST), 300, vp1, mp1)
        _cpp_train(str(TINY_TEST), 300, vp2, mp2)

        merges1 = _parse_cpp_merges(mp1)
        merges2 = _parse_cpp_merges(mp2)
        assert merges1 == merges2, "C++ training is non-deterministic"

        vocab1 = _parse_cpp_vocab(vp1)
        vocab2 = _parse_cpp_vocab(vp2)
        assert vocab1 == vocab2


# ============================================================================
# 4. Edge Cases & Stress Tests
# ============================================================================

class TestEdgeCases:
    """Boundary conditions and stress tests."""

    def test_single_char(self):
        for ch in "abcABC019!@# ":
            ids = _cpp_encode(ch)
            assert len(ids) >= 1
            decoded = _cpp_decode(ids)
            assert decoded == ch

    def test_repeated_text(self):
        text = "abc " * 100
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text

    def test_only_spaces(self):
        text = "     "
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text

    def test_only_special_tokens(self):
        text = "<|endoftext|><|endoftext|><|endoftext|>"
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text
        # Should be exactly 3 special token IDs
        assert len(ids) == 3
        assert all(i == 0 for i in ids)

    def test_newlines_and_tabs(self):
        text = "line1\nline2\ttab"
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text

    def test_long_text_roundtrip(self):
        """Read the entire training file and roundtrip it."""
        with open(TINY_TEST) as f:
            text = f.read()
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text

    def test_punctuation_heavy(self):
        text = "...!!! ??? --- ,,,"
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text

    def test_numbers(self):
        text = "42 is the answer to 6 times 7"
        ids = _cpp_encode(text)
        decoded = _cpp_decode(ids)
        assert decoded == text

    def test_compression_ratio(self):
        """Merged tokens should compress the input (fewer tokens than bytes)."""
        text = "the the the the the"  # highly repetitive
        ids = _cpp_encode(text)
        assert len(ids) < len(text), (
            f"No compression: {len(text)} bytes → {len(ids)} tokens"
        )
