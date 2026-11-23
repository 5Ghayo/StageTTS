import importlib.util
import py_compile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BanquetWorkerTest(unittest.TestCase):
    def test_banquet_worker_compiles(self):
        py_compile.compile(str(ROOT / "workers" / "banquet_worker.py"), doraise=True)

    def test_banquet_worker_separates_per_channel_and_normalizes_peak(self):
        missing = [mod for mod in ("torch", "torchaudio", "soundfile") if importlib.util.find_spec(mod) is None]
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

        clipped = banquet_worker._normalize_peak(torch.tensor([[1.5, -0.5]]))
        self.assertAlmostEqual(float(clipped.abs().max()), 0.99, places=5)
        unclipped = torch.tensor([[0.8, -0.5]])
        self.assertTrue(torch.equal(banquet_worker._normalize_peak(unclipped), unclipped))


if __name__ == "__main__":
    unittest.main()
