"""Quick benchmark: dogfeed examples across reward models."""
import sys
import numpy as np
sys.path.insert(0, "reasoning_train")

TEST_CASES = [
    # (name, reference, response)
    ("Exact copy", 
     "The model achieves 95.3% accuracy on ImageNet with a ResNet-50 backbone.",
     "The model achieves 95.3% accuracy on ImageNet with a ResNet-50 backbone."),
    
    ("1:1 identical",
     "GRPO uses group-relative advantages to optimize policy models without a separate reward model.",
     "GRPO uses group-relative advantages to optimize policy models without a separate reward model."),
    
    ("One-word swap", 
     "The model achieves 95.3% accuracy on ImageNet with a ResNet-50 backbone.",
     "The model achieves 95.3% accuracy on ImageNet with a ResNet-101 backbone."),
    
    ("Negation",
     "The proposed method outperforms all baselines by at least 3.2%.",
     "The proposed method does not outperform all baselines by at least 3.2%."),
    
    ("Wrong number",
     "The training used a batch size of 64 and learning rate of 1e-4 for 50 epochs.",
     "The training used a batch size of 64 and learning rate of 1e-3 for 50 epochs."),
    
    ("Right facts, wrong format",
     "- The attention mechanism uses 8 heads of dimension 64.\n- Layer norm is applied before each sublayer.",
     "The attention mechanism uses 8 heads of dimension 64 and layer norm is applied before each sublayer."),
    
    ("Half response",
     "The encoder consists of 6 identical layers, each with multi-head self-attention and feed-forward sublayers.",
     "The encoder consists of 6 identical layers, each"),
    
    ("Repeated 2x",
     "The decoder uses masked self-attention to prevent attending to subsequent positions.",
     "The decoder uses masked self-attention to prevent attending to subsequent positions. The decoder uses masked self-attention to prevent attending to subsequent positions."),
    
    ("Completely irrelevant",
     "GRPO is a reinforcement learning algorithm that uses group-relative advantages for policy optimization.",
     "The Eiffel Tower was built in 1889 and is made of wrought iron."),
    
    ("Correct paraphrase",
     "The training alternates between policy evaluation and policy improvement steps.",
     "Training alternates between evaluating the policy and improving it in successive steps."),
    
    ("Wrong dataset name",
     "They evaluated on CIFAR-10 and achieved 98.2% test accuracy.",
     "They evaluated on CIFAR-100 and achieved 98.2% test accuracy."),
    
    ("Good Q&A pair",
     "**Question:** What is the batch size?\n**Answer:** The batch size is 64.",
     "**Question:** What was the batch size used?\n**Answer:** A batch size of 64 was used during training."),
    
    ("Verbose bloated",
     "The learning rate was set to 0.001.",
     "The learning rate was set to 0.001. This is a standard learning rate. Many papers use 0.001 as the default learning rate. The learning rate determines step size. A learning rate of 0.001 is commonly used in practice."),
]


def load_models():
    from reward_model import load_reward_model
    
    models = []
    model_dirs = [
        ("v3 (MiniLM)", "models/reward_model_final/reward_model_finetuned_v3"),
        ("v5 (MiniLM-1w)", "models/reward_model_final/reward_model_finetuned_v5"),
    ]
    for name, path in model_dirs:
        try:
            m, _ = load_reward_model(path)
            models.append((name, m))
        except Exception as e:
            print(f"  Skipping {name}: {e}")
    return models


def main():
    models = load_models()
    if not models:
        print("No models found")
        return

    all_scores = []
    for name, model in models:
        refs = [t[1] for t in TEST_CASES]
        resps = [t[2] for t in TEST_CASES]
        scores = np.clip(np.array(model.score_batch(refs, resps)), 0, 1)
        all_scores.append((name, scores))

    # Header
    header = f"{'Case':<28}"
    for name, _ in all_scores:
        header += f" {name:>8}"
    print(header)
    print("-" * (28 + 10 * len(all_scores)))

    for i, (case_name, _, _) in enumerate(TEST_CASES):
        row = f"{case_name:<28}"
        for _, scores in all_scores:
            s = scores[i]
            row += f" {s:>8.3f}"
        print(row)


if __name__ == "__main__":
    main()
