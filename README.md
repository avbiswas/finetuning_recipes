# Finetuning Recipes

End-to-end post-training recipes for small language models — from continued pre-training to preference alignment. Part of the [Neural Breakdown YouTube course](https://www.youtube.com/@avb_fj).

## What This Covers

| Stage | Description |
|-------|-------------|
| **CPT** (Continued Pre-Training) | Further pre-train a base model on domain-specific tasks |
| **SFT** (Supervised Fine-Tuning) | Instruction-tune the CPT model on instruction data |
| **DPO** (Direct Preference Optimization) | Align the SFT model using pairwise human/AI preferences |
| **RL** (Reinforcement learning) | Training Reward models and using them to train neural models |

## Related Repos

- **[text-albumentations](https://github.com/avbiswas/text-albumentations)** — dataset generation & augmentation library
- **[neural-txt](https://github.com/avbiswas/neural-txt)** — Inference harness
- **[paper_instructions_300K-v1](https://huggingface.co/datasets/paperbd/paper_instructions_300K-v1)** — instruction dataset generated from arXiv papers


## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)
