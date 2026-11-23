import importlib.util
import unittest
from unittest.mock import patch

from workers.aligned_segments import normalize_segments
from workers.semantic_boundaries import (
    BoundaryCandidate, LexicalToken, generate_boundary_plan,
    load_tokenizer, rank_candidate_ids,
)


class InjectedTokenizer:
    name = "injected"
    version = "1"

    def __init__(self, tokens):
        self.tokens = tokens

    def tokenize(self, text):
        return self.tokens


def character_tokens(text):
    return InjectedTokenizer([LexicalToken(char, i, i + 1) for i, char in enumerate(text)])


def aligned_chars(text, pause_at=None, speaker_at=None):
    words = []
    extra = 0.0
    for i, char in enumerate(text):
        if i == pause_at:
            extra += 0.8
        words.append(dict(word=char, start=round(i + extra, 3), end=round(i + extra + 0.8, 3),
                          speaker="B" if speaker_at is not None and i >= speaker_at else "A"))
    return words


def raw(text, words):
    return dict(text=text, start=0.0, end=words[-1]["end"], words=words, speaker="A")


class SemanticBoundaryTest(unittest.TestCase):
    def test_intra_compound_pause_is_not_a_candidate(self):
        text = "甲一室乙"
        tokens = InjectedTokenizer([LexicalToken("甲", 0, 1), LexicalToken("一室", 1, 3), LexicalToken("乙", 3, 4)])
        plan = generate_boundary_plan(text, aligned_chars(text, pause_at=2), language="ja", tokenizer=tokens)
        self.assertEqual(plan.candidates, ())
        self.assertEqual(plan.review_reasons, ())

    def test_long_compound_is_never_cut_despite_14s_pause(self):
        text = "甲" * 13 + "一室" + "乙" * 4
        tokens = [LexicalToken("甲", i, i + 1) for i in range(13)]
        tokens += [LexicalToken("一室", 13, 15)]
        tokens += [LexicalToken("乙", i, i + 1) for i in range(15, len(text))]
        segments = normalize_segments([raw(text, aligned_chars(text, pause_at=14))],
                                      language="ja", tokenizer=InjectedTokenizer(tokens), diarized=True)
        self.assertEqual([seg["orig_text"] for seg in segments], [text])
        self.assertTrue(segments[0]["alignment_needs_review"])
        self.assertIn("no_reliable_boundary", segments[0]["boundary_review_reasons"])

    def test_extension_pause_15s_uses_aligned_times_and_provenance(self):
        text = "甲" * 19
        words = aligned_chars(text, pause_at=15)
        segments = normalize_segments([raw(text, words)], language="zh",
                                      tokenizer=character_tokens(text), diarized=True)
        self.assertEqual([len(seg["orig_text"]) for seg in segments], [15, 4])
        self.assertEqual(segments[0]["end"], words[14]["end"])
        self.assertEqual(segments[1]["start"], words[15]["start"])
        evidence = segments[0]["boundary_provenance"][0]
        self.assertEqual(evidence["char_offset"], 15)
        self.assertEqual(evidence["word_index"], 15)
        self.assertEqual(evidence["tokenizer"], "injected")
        self.assertEqual(set(evidence["reasons"]), {"aligned_pause", "lexical_token_end"})
        self.assertEqual(len(evidence["source_text_sha256"]), 64)

    def test_comma_requires_pause_terminal_does_not(self):
        text = "甲，乙。丙"
        no_pause = generate_boundary_plan(text, aligned_chars(text), tokenizer=character_tokens(text))
        self.assertEqual([c.char_offset for c in no_pause.candidates], [4])
        with_pause = generate_boundary_plan(text, aligned_chars(text, pause_at=2), tokenizer=character_tokens(text))
        self.assertEqual([c.char_offset for c in with_pause.candidates], [2, 4])

    def test_original_offsets_survive_asr_whitespace(self):
        text = "甲  一室　乙"
        tokenizer = InjectedTokenizer([LexicalToken("甲", 0, 1), LexicalToken("一室", 3, 5), LexicalToken("乙", 6, 7)])
        words = [dict(word="甲", start=0, end=1), dict(word="一 室", start=2, end=3), dict(word="乙", start=4, end=5)]
        plan = generate_boundary_plan(text, words, language="ja", tokenizer=tokenizer)
        self.assertEqual([c.char_offset for c in plan.candidates], [3, 6])
        self.assertEqual([c.word_index for c in plan.candidates], [1, 2])

    def test_tokenizer_surface_normalization_drift_is_rejected(self):
        text = "Ａ一室"
        bad = InjectedTokenizer([LexicalToken("A", 0, 1), LexicalToken("一室", 1, 3)])
        plan = generate_boundary_plan(text, aligned_chars(text, pause_at=1), language="ja", tokenizer=bad)
        self.assertEqual(plan.candidates, ())
        self.assertTrue(any("tokenizer_invalid_offsets" in reason for reason in plan.review_reasons))

    def test_incomplete_tokenizer_coverage_is_rejected(self):
        text = "甲乙丙"
        bad = InjectedTokenizer([LexicalToken("甲", 0, 1), LexicalToken("丙", 2, 3)])
        plan = generate_boundary_plan(text, aligned_chars(text, pause_at=2), language="ja", tokenizer=bad)
        self.assertEqual(plan.candidates, ())
        self.assertTrue(plan.review_reasons)

    def test_missing_package_is_conservative_and_marks_review(self):
        text = "甲" * 18
        with patch("workers.semantic_boundaries.load_tokenizer", side_effect=ImportError("absent")):
            segments = normalize_segments([raw(text, aligned_chars(text, pause_at=14))], language="ja", diarized=True)
        self.assertEqual([seg["orig_text"] for seg in segments], [text])
        self.assertTrue(segments[0]["alignment_needs_review"])
        self.assertTrue(any("tokenizer_unavailable" in reason for reason in segments[0]["boundary_review_reasons"]))

    def test_text_mismatch_retains_text_and_no_guessed_times(self):
        plan = generate_boundary_plan("甲乙", [dict(word="甲丙", start=0, end=1)], language="zh", tokenizer=character_tokens("甲乙"))
        self.assertEqual(plan.candidates, ())
        self.assertEqual(plan.review_reasons, ("alignment_text_mismatch",))

    def test_selector_accepts_only_supplied_ids_and_reorders_no_times(self):
        first = BoundaryCandidate("one", 1, 1, 1.0, 2.0, ("aligned_pause",))
        second = BoundaryCandidate("two", 2, 2, 3.0, 4.0, ("aligned_pause",))
        ranked, issues = rank_candidate_ids((first, second), lambda candidates: ["two"])
        self.assertEqual(ranked, (second, first))
        self.assertEqual(issues, ())
        for response in (["invented"], ["one", "one"], [{"id": "one", "time": 9}]):
            ranked, issues = rank_candidate_ids((first, second), lambda candidates: response)
            self.assertEqual(ranked, (first, second))
            self.assertEqual(issues, ("boundary_selector_invalid_ids",))

    def test_real_speaker_change_overrides_lexical_compound(self):
        text = "一室"
        tokens = InjectedTokenizer([LexicalToken(text, 0, 2)])
        segments = normalize_segments([raw(text, aligned_chars(text, speaker_at=1))],
                                      language="ja", tokenizer=tokens, diarized=True)
        self.assertEqual([seg["speaker"] for seg in segments], ["A", "B"])
        self.assertEqual("".join(seg["orig_text"] for seg in segments), text)
        self.assertIn("speaker_change", segments[0]["boundary_provenance"][0]["reasons"])

    def test_invalid_tokenizer_is_conservative_without_language_hint(self):
        text = "甲乙丙"
        tokenizer = InjectedTokenizer([LexicalToken("甲丙", 0, 2)])
        plan = generate_boundary_plan(text, aligned_chars(text, pause_at=1), tokenizer=tokenizer)
        self.assertEqual(plan.candidates, ())
        self.assertTrue(plan.review_reasons)

    def test_overlapping_alignment_cannot_become_a_boundary(self):
        words = [dict(word="甲。", start=0, end=2), dict(word="乙", start=1, end=3)]
        plan = generate_boundary_plan("甲。乙", words)
        self.assertEqual(plan.candidates, ())

    def test_boundary_provenance_survives_schema_serialization(self):
        from backend.schemas import Segment
        text = "甲" * 19
        words = aligned_chars(text, pause_at=15)
        segments = normalize_segments([raw(text, words)], language="zh",
                                      tokenizer=character_tokens(text), diarized=True)
        parsed = Segment(**segments[0], target_text=segments[0]["orig_text"])
        restored = Segment.model_validate_json(parsed.model_dump_json())
        self.assertEqual(restored.boundary_provenance[0].char_offset, 15)
        self.assertEqual(restored.boundary_provenance[0].source_text_sha256,
                         segments[0]["boundary_provenance"][0]["source_text_sha256"])

    @unittest.skipUnless(importlib.util.find_spec("jieba"), "optional jieba not installed")
    def test_real_jieba_compound_pause_cannot_force_a_cut(self):
        text = "甲" * 13 + "中华人民共和国" + "之后"
        tokenizer = load_tokenizer("zh")
        self.assertIn("中华人民共和国", [token.text for token in tokenizer.tokenize(text)])
        segments = normalize_segments([raw(text, aligned_chars(text, pause_at=14))],
                                      language="zh", diarized=True)
        self.assertEqual([seg["orig_text"] for seg in segments], [text])
        self.assertTrue(segments[0]["alignment_needs_review"])

    @unittest.skipUnless(importlib.util.find_spec("sudachipy") and importlib.util.find_spec("sudachidict_core"),
                         "optional Sudachi/core not installed")
    def test_real_sudachi_long_compound_pause_cannot_force_a_cut(self):
        text = "あ" * 13 + "一室で話します"
        segments = normalize_segments([raw(text, aligned_chars(text, pause_at=14))],
                                      language="ja", diarized=True)
        self.assertEqual([seg["orig_text"] for seg in segments], [text])
        self.assertTrue(segments[0]["alignment_needs_review"])

    @unittest.skipUnless(importlib.util.find_spec("jieba"), "optional jieba not installed")
    def test_real_jieba_offsets_cover_original(self):
        tokenizer = load_tokenizer("zh")
        text = "这是一室一厅的房子。"
        tokens = tokenizer.tokenize(text)
        self.assertEqual("".join(token.text for token in tokens), text)
        plan = generate_boundary_plan(text + "好", aligned_chars(text + "好"), language="zh")
        self.assertFalse(plan.review_reasons)
        self.assertEqual(plan.candidates[-1].char_offset, len(text))

    @unittest.skipUnless(importlib.util.find_spec("sudachipy") and importlib.util.find_spec("sudachidict_core"),
                         "optional Sudachi/core not installed")
    def test_real_sudachi_compound_has_no_internal_boundary(self):
        tokenizer = load_tokenizer("ja")
        text = "一室で話します。"
        tokens = tokenizer.tokenize(text)
        self.assertIn("一室", [token.text for token in tokens])
        plan = generate_boundary_plan(text, aligned_chars(text, pause_at=1), language="ja")
        self.assertFalse(plan.review_reasons)
        self.assertNotIn(1, [candidate.char_offset for candidate in plan.candidates])


if __name__ == "__main__":
    unittest.main()
