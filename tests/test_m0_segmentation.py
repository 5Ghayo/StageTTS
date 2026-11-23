import unittest

from workers.aligned_segments import (
    DEFAULT_SPEAKER,
    HARD_MAX_SECONDS,
    MAX_SECONDS,
    MIN_FRAGMENT_SECONDS,
    normalize_segments,
    _join_text,
)


def _word(text, start, end, speaker=None):
    return {
        "word": text,
        "start": float(start),
        "end": float(end),
        "speaker": speaker,
    }


def _raw(text, words, start=0.0, end=None, speaker=None):
    return {
        "text": text,
        "words": words,
        "start": float(start),
        "end": float(end if end is not None else words[-1]["end"]),
        "speaker": speaker,
    }


class SegmentationM0Test(unittest.TestCase):
    def test_continuous_18s_without_pause_or_terminal_punctuation_stays_whole_and_flagged(self):
        words = [_word(f"词{i}", i, i + 1) for i in range(18)]
        raw = _raw("".join(word["word"] for word in words), words, end=18.0)

        segments = normalize_segments([raw], diarized=False)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 18.0)
        self.assertEqual(segments[0]["orig_text"], "".join(word["word"] for word in words))
        self.assertEqual(segments[0]["words"], words)
        self.assertTrue(segments[0]["alignment_needs_review"])
        self.assertTrue(segments[0]["speaker_needs_review"])
        self.assertEqual(segments[0]["speaker"], DEFAULT_SPEAKER)

    def test_acoustic_pause_before_hard_max_is_used_as_boundary(self):
        words = [_word("甲", i, i + 0.8) for i in range(15)]
        words += [_word("乙", 15.75, 16.55)]
        text = "".join(word["word"] for word in words)
        raw = _raw(text, words, end=16.55)

        segments = normalize_segments([raw], diarized=False)

        self.assertEqual([(seg["start"], seg["end"]) for seg in segments], [(0.0, 14.8), (15.75, 16.55)])
        self.assertEqual(segments[0]["orig_text"], "".join(word["word"] for word in words[:15]))
        self.assertEqual(segments[1]["orig_text"], words[15]["word"])

    def test_one_room_cjk_compound_not_split_at_fitting_boundary(self):
        words = [_word(char, i, i + 1) for i, char in enumerate("一室一厅")]
        words += [_word("很", 4.1, 5.1), _word("好", 5.2, 6.2)]
        raw = _raw("一室一厅很好", words, end=6.2)

        segments = normalize_segments([raw], diarized=False)

        self.assertEqual([seg["orig_text"] for seg in segments], ["一室一厅很好"])

    def test_valid_speaker_change_forces_split_even_without_natural_boundary(self):
        words = [_word(f"词{i}", i, i + 1, "A") for i in range(3)]
        words += [_word(f"词{i}", i, i + 1, "B") for i in range(3, 6)]
        text = "".join(word["word"] for word in words)
        raw = _raw(text, words, end=6.0, speaker="A")

        segments = normalize_segments([raw], diarized=True)

        self.assertEqual([seg["speaker"] for seg in segments], ["A", "B"])
        self.assertEqual([seg["orig_text"] for seg in segments], ["词0词1词2", "词3词4词5"])
        self.assertFalse(any(seg["speaker_needs_review"] for seg in segments))

    def test_missing_alignment_keeps_all_text_and_review_flag(self):
        raw = {
            "text": "word1 word2",
            "words": [],
            "start": 0.0,
            "end": 3.0,
            "speaker": "A",
        }

        segments = normalize_segments([raw], diarized=False)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["orig_text"], "word1 word2")
        self.assertTrue(segments[0]["alignment_needs_review"])

    def test_unknown_speaker_review_not_erased_by_short_merge(self):
        left = [_word("Hi", 0.0, 1.0), _word("there", 1.05, 2.05)]
        right = [_word("John", 2.3, 3.0)]
        raw_left = _raw("Hi there", left, end=2.05)
        raw_right = _raw("John", right, start=2.3, end=3.0)

        segments = normalize_segments([raw_left, raw_right], diarized=False)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["orig_text"], "Hi there John")
        self.assertTrue(segments[0]["speaker_needs_review"])

    def test_punctuation_then_latin_join_gets_a_space(self):
        self.assertEqual(_join_text("Hello,", "world"), "Hello, world")
        self.assertEqual(_join_text("Hello;", "world"), "Hello; world")
        self.assertEqual(_join_text("你好，", "世界"), "你好，世界")

    def test_constants_define_controlled_overflow(self):
        self.assertGreater(MAX_SECONDS, MIN_FRAGMENT_SECONDS)
        self.assertGreater(HARD_MAX_SECONDS, MAX_SECONDS)


if __name__ == "__main__":
    unittest.main()
