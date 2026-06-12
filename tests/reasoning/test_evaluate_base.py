import unittest

from reasoning.evaluate_base import score_rollouts, summarize


class RewardModel:
    def batch_score(self, responses, references, batch_size):
        return [0.8 for _ in responses]


class EvaluateBaseTests(unittest.TestCase):
    def test_score_rollouts_matches_reward_space(self):
        results = score_rollouts(
            [{
                "reference": "plain answer",
                "completion": "<think>reason carefully</think> plain answer",
            }],
            RewardModel(),
        )

        self.assertEqual(results[0]["rewards"], {
            "think_format_reward": 1.0,
            "output_format_reward": 0.5,
            "doom_loop_reward": 0.0,
            "neuraltxt_reward": 0.8,
        })
        self.assertEqual(results[0]["total_reward"], 2.3)

    def test_summary_reports_failure_rates(self):
        results = score_rollouts(
            [
                {
                    "reference": "plain answer",
                    "completion": "<think>x</think> plain answer",
                },
                {
                    "reference": "plain answer",
                    "completion": '<think>x x x x</think> {"answer": 1}',
                },
            ],
            RewardModel(),
        )
        summary = summarize(results)

        self.assertEqual(summary["num_rollouts"], 2)
        self.assertEqual(summary["think_format_pass_rate"], 1.0)
        self.assertEqual(summary["schema_match_rate"], 0.5)
        self.assertEqual(summary["doom_loop_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
