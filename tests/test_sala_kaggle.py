import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from sala_kaggle import (
    FINAL_COLUMNS,
    SETTING_MAP,
    calibrate_threshold,
    csv_dump_atomic,
    required_domains,
    source_domains,
    source_split_indices,
    stack_last_token_features,
)


class SalaKaggleTests(unittest.TestCase):
    def test_official_g3_g5_mappings_are_preserved(self):
        self.assertEqual(SETTING_MAP["G3"], {"source": "tqa", "target": "triviaqa"})
        self.assertEqual(SETTING_MAP["G5"], {"source": "sciq", "target": "nq_open"})

    def test_official_g14_mapping_uses_triviaqa_as_target(self):
        self.assertEqual(
            SETTING_MAP["G14"],
            {"source": ["tqa", "nq_open", "sciq"], "target": "triviaqa"},
        )
        self.assertEqual(source_domains("G14"), ["tqa", "nq_open", "sciq"])
        self.assertEqual(
            required_domains(["G14"]),
            ["tqa", "nq_open", "sciq", "triviaqa"],
        )

    def test_g14_splits_each_source_without_overlap(self):
        labels = np.asarray([0, 1] * 50, dtype=np.int64)
        train, validation, test = source_split_indices(labels, "G14", "tqa")
        self.assertEqual(len(train), 67)
        self.assertEqual(len(validation), 8)
        self.assertEqual(len(test), 25)
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))
        self.assertEqual(set(np.unique(labels[train])), {0, 1})

    def test_official_g15_mapping_uses_nq_open_as_target(self):
        self.assertEqual(
            SETTING_MAP["G15"],
            {"source": ["tqa", "sciq", "triviaqa"], "target": "nq_open"},
        )
        self.assertEqual(source_domains("G15"), ["tqa", "sciq", "triviaqa"])
        self.assertEqual(
            required_domains(["G15"]),
            ["tqa", "sciq", "triviaqa", "nq_open"],
        )

    def test_probability_threshold_uses_one_as_correct(self):
        threshold, details = calibrate_threshold(
            [0.1, 0.2, 0.8, 0.9],
            [0, 0, 1, 1],
        )
        predictions = [int(score >= threshold) for score in [0.1, 0.2, 0.8, 0.9]]
        self.assertEqual(predictions, [0, 0, 1, 1])
        self.assertEqual(details["balanced_accuracy"], 1.0)

    def test_csv_replaces_hide_score_with_sala_probability(self):
        row = {
            "setting": "G5",
            "source_domain": "sciq",
            "target_domain": "nq_open",
            "id": "9",
            "input_text": "Question?",
            "output_text": "Answer",
            "ground_truth_text": "Reference",
            "ground_truth_label": 1,
            "bleurt_score": 0.9,
            "sala_probability": 0.8,
            "predicted_label": 1,
            "split": "target_evaluation",
            "sala_threshold": 0.55,
            "ground_truth_answers": ["Reference", "Alias"],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            csv_dump_atomic(path, [row])
            with path.open() as stream:
                saved = list(csv.DictReader(stream))
        self.assertEqual(list(saved[0]), FINAL_COLUMNS)
        self.assertIn("sala_probability", saved[0])
        self.assertNotIn("hide_score", saved[0])
        self.assertEqual(json.loads(saved[0]["ground_truth_answers"]), ["Reference", "Alias"])

    def test_qwen_replay_produces_one_vector_per_layer(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=None,
        )
        model = Qwen2ForCausalLM(config).eval()
        input_ids = torch.tensor([[2, 3, 4, 5, 6]])
        with torch.no_grad():
            output = model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
                use_cache=False,
            )
        features = stack_last_token_features(output.hidden_states, torch)
        self.assertEqual(tuple(features.shape), (config.num_hidden_layers + 1, config.hidden_size))
        self.assertTrue(math.isfinite(float(features.mean())))


if __name__ == "__main__":
    unittest.main()
