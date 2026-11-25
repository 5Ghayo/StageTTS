import importlib.util
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BanquetWorkerTest(unittest.TestCase):
    def test_banquet_worker_compiles(self):
        py_compile.compile(str(ROOT / "workers" / "banquet_worker.py"), doraise=True)

    def test_banquet_worker_separates_per_channel(self):
        missing = [mod for mod in ("torch", "numpy", "soundfile") if importlib.util.find_spec(mod) is None]
        if missing:
            self.skipTest(f"requires {', '.join(missing)}")
        import torch

        from workers import banquet_worker

        seen_batches: list[int] = []

        class FakeModel:
            def __call__(self, batch):
                audio = batch["mixture"]["audio"]
                seen_batches.append(audio.shape[0])
                return {stem: audio * (idx + 1) for idx, stem in enumerate(banquet_worker.STEMS)}

        messages: list[str] = []
        tiny_wave = torch.ones((1, 400), dtype=torch.float32)
        tiny_out = banquet_worker.separate(FakeModel(), tiny_wave, "cpu", progress=messages.append)
        self.assertEqual(seen_batches, [1])
        self.assertEqual(tiny_out["dialogue"].shape, (1, 400))

        seen_batches.clear()
        short_wave = torch.ones((2, 2000), dtype=torch.float32)
        short_out = banquet_worker.separate(FakeModel(), short_wave, "cpu", progress=messages.append)
        self.assertEqual(seen_batches, [1, 1])
        self.assertEqual(short_out["dialogue"].shape, (2, 2000))
        self.assertTrue(torch.allclose(short_out["music"], short_wave * 4))

        seen_batches.clear()
        long_len = int(9.3 * banquet_worker.SAMPLE_RATE)
        long_wave = torch.stack([torch.full((long_len,), 0.25), torch.full((long_len,), 0.5)])
        long_out = banquet_worker.separate(FakeModel(), long_wave, "cpu", progress=messages.append)
        self.assertTrue(all(batch == 1 for batch in seen_batches))
        self.assertEqual(long_out["dialogue"].shape, (2, long_len))
        self.assertTrue(torch.allclose(long_out["dialogue"], long_wave, atol=1e-4))
        self.assertTrue(any("分离进度" in msg for msg in messages))

        hop4_out = banquet_worker.separate(FakeModel(), long_wave, "cpu", hop_seconds=4.0)
        self.assertTrue(torch.allclose(hop4_out["dialogue"], long_wave, atol=1e-4))
        with self.assertRaises(ValueError):
            banquet_worker.separate(FakeModel(), short_wave, "cpu", hop_seconds=5.0)

    def test_silent_chunk_seams_preserve_samples_and_bound_memory(self):
        missing = [mod for mod in ("torch", "numpy", "soundfile") if importlib.util.find_spec(mod) is None]
        if missing:
            self.skipTest(f"requires {', '.join(missing)}")
        import numpy as np
        import soundfile as sf

        from workers import banquet_worker

        rate = banquet_worker.SAMPLE_RATE
        source = np.zeros((rate * 9 + 11, 2), dtype="float32")
        source[:, 0] = 0.15 + np.arange(len(source)) / len(source) * 0.2
        source[:, 1] = -source[:, 0]
        source[round(3.2 * rate):round(3.6 * rate)] = 0
        chunk_sizes = []

        def fake_separate(_model, samples, _device, _progress, _hop):
            chunk_sizes.append(samples.shape[-1])
            return {stem: samples * 4 if stem == "music" else samples for stem in banquet_worker.STEMS}

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.wav"
            sf.write(path, source, rate, subtype="FLOAT")
            with sf.SoundFile(path) as track:
                points = banquet_worker.chunk_boundaries(track, target_seconds=3, search_seconds=0.8)
            self.assertAlmostEqual(points[1] / rate, 3.4, delta=0.021)
            self.assertEqual(points[-1], len(source))
            out = Path(temp) / "stems"
            with patch.object(banquet_worker, "separate", side_effect=fake_separate), \
                 patch.object(banquet_worker, "CONTEXT_SECONDS", 0.5):
                banquet_worker.separate_file(None, path, out, "cpu", lambda _message: None, 0.5,
                                             target_seconds=3, search_seconds=0.8)
            self.assertEqual(len(chunk_sizes), len(points) - 1)
            self.assertLess(max(chunk_sizes), 5.5 * rate)
            for stem in banquet_worker.STEMS:
                result, out_rate = sf.read(out / f"{stem}.wav")
                self.assertEqual(out_rate, rate)
                expected = source * (0.99 / float(np.max(np.abs(source)))) if stem == "music" else source
                np.testing.assert_allclose(result, expected, atol=1 / 32768)


if __name__ == "__main__":
    unittest.main()
