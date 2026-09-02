import io
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import train
from data import UNK_TOKEN, _check_complete, build_vocab, dedupe_works, encode, get_converter, normalize_script, split_poems
from model import GPT, GPTConfig


class DataTests(unittest.TestCase):
    def test_split_keeps_whole_poems(self):
        poems = [f"poem-{index}\nbody-{index}\n" for index in range(20)]
        training, validation = split_poems(poems)

        self.assertEqual(training, poems[:19])
        self.assertEqual(validation, poems[19:])
        self.assertFalse(set(training) & set(validation))

    def test_duplicates_are_dropped_by_body_regardless_of_header_and_punctuation(self):
        works = [
            "静夜思\n床前明月光，疑是地上霜。\n",
            "静夜思·李白\n床前明月光。疑是地上霜！\n",  # same poem, other punctuation
            "春晓\n春眠不觉晓，处处闻啼鸟。\n",
        ]
        kept, dropped = dedupe_works(works)
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [works[0], works[2]])

    def test_vocab_maps_rare_characters_to_unknown(self):
        text = "春春春眠眠不"
        stoi, itos = build_vocab(text, min_count=2)
        self.assertEqual(itos[0], UNK_TOKEN)
        self.assertIn("春", stoi)
        self.assertIn("眠", stoi)
        self.assertNotIn("不", stoi)
        encoded = encode("春不晓", stoi).tolist()
        self.assertEqual(encoded, [stoi["春"], 0, 0])

    def test_script_normalization_unifies_traditional_and_simplified(self):
        self.assertIsNone(get_converter("mixed"))
        self.assertEqual(normalize_script("後來", None), "後來")
        converter = get_converter("simplified")
        self.assertEqual(normalize_script("春眠不覺曉，處處聞啼鳥。", converter), "春眠不觉晓，处处闻啼鸟。")
        with self.assertRaises(ValueError):
            get_converter("klingon")

    def test_partial_corpus_is_strict_by_default(self):
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "refusing to build a partial dataset"):
                _check_complete("test", skipped=1, expected=2, allow_partial=False)

            _check_complete("test", skipped=1, expected=2, allow_partial=True)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        config = GPTConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
        )
        self.model = GPT(config).eval()
        self.prompt = torch.tensor([[1, 2]], dtype=torch.long)

    def test_temperature_zero_is_deterministic_greedy_decoding(self):
        first = self.model.generate(self.prompt, 3, temperature=0, top_k=4)
        second = self.model.generate(self.prompt, 3, temperature=0, top_k=4)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (1, 5))

    def test_generation_rejects_invalid_values(self):
        for kwargs in (
            {"temperature": -0.1},
            {"top_k": 0},
            {"top_k": -1},
            {"max_new_tokens": -1},
        ):
            arguments = {"max_new_tokens": 1, **kwargs}
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.model.generate(self.prompt, **arguments)


class TrainingTests(unittest.TestCase):
    def test_batch_sampler_includes_last_valid_start(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            np.arange(8, dtype=np.int16).tofile(data_dir / "train.bin")
            with patch.object(train, "DATA_DIR", data_dir), patch.object(
                train.torch, "randint", return_value=torch.tensor([4])
            ) as randint:
                x, y = train.get_batch("train", block_size=3, batch_size=1, device="cpu")

            randint.assert_called_once_with(5, (1,))
            self.assertEqual(x.tolist(), [[4, 5, 6]])
            self.assertEqual(y.tolist(), [[5, 6, 7]])

    def test_rng_state_round_trip(self):
        torch.manual_seed(123)
        state = train.capture_rng_state()
        expected = torch.rand(4)
        train.restore_rng_state(state)
        actual = torch.rand(4)
        self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_is_safe_loadable_and_records_next_iteration(self):
        config = GPTConfig(
            vocab_size=8,
            block_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
        )
        model = GPT(config)
        optimizer = torch.optim.AdamW(model.parameters())
        meta = {"stoi": {"春": 0}, "itos": {0: "春"}, "vocab_size": 1}
        payload = train.checkpoint_payload(model, optimizer, config, 4, 1.25, meta)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny_latest.pt"
            train.save_checkpoint(payload, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(restored["iter"], 4)
        self.assertEqual(restored["next_iter"], 5)
        self.assertIn("rng_state", restored)

    def test_training_writes_latest_and_best_then_resumes_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            checkpoint_dir = root / "checkpoints"
            data_dir.mkdir()
            checkpoint_dir.mkdir()

            tokens = np.arange(64, dtype=np.int16) % 8
            tokens.tofile(data_dir / "train.bin")
            tokens.tofile(data_dir / "val.bin")
            meta = {
                "stoi": {str(index): index for index in range(8)},
                "itos": {index: str(index) for index in range(8)},
                "vocab_size": 8,
            }
            with open(data_dir / "meta.pkl", "wb") as handle:
                pickle.dump(meta, handle)

            config = GPTConfig(
                vocab_size=8,
                block_size=4,
                n_layer=1,
                n_head=1,
                n_embd=8,
                dropout=0.0,
            )
            common_args = [
                "train.py",
                "--config",
                "tiny",
                "--device",
                "cpu",
                "--batch_size",
                "1",
                "--warmup",
                "0",
                "--eval_iters",
                "1",
                "--eval_interval",
                "1",
                "--log_interval",
                "1",
            ]

            with patch.object(train, "DATA_DIR", data_dir), patch.object(
                train, "CKPT_DIR", checkpoint_dir
            ), patch.object(GPTConfig, "tiny", return_value=config), patch.object(
                sys, "argv", common_args + ["--iters", "1"]
            ), redirect_stdout(io.StringIO()):
                train.main()

            latest_path = checkpoint_dir / "tiny_latest.pt"
            best_path = checkpoint_dir / "tiny_best.pt"
            self.assertTrue(latest_path.exists())
            self.assertTrue(best_path.exists())
            first = torch.load(latest_path, map_location="cpu", weights_only=True)
            self.assertEqual(first["next_iter"], 1)

            with patch.object(train, "DATA_DIR", data_dir), patch.object(
                train, "CKPT_DIR", checkpoint_dir
            ), patch.object(
                sys, "argv", common_args + ["--iters", "2", "--resume"]
            ), redirect_stdout(io.StringIO()):
                train.main()

            resumed = torch.load(latest_path, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["iter"], 1)
            self.assertEqual(resumed["next_iter"], 2)


if __name__ == "__main__":
    unittest.main()
