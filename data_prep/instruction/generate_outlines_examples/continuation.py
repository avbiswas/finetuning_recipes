from utils import AlpacaDataset


def split_words(text: str) -> list[str]:
    return text.split()


def join_words(words: list[str]) -> str:
    return " ".join(words).strip()


def get_slice_size(word_count: int, fraction: float) -> int:
    if word_count <= 0:
        return 0
    return max(1, int(word_count * fraction))


def convert_to_dataset(chunk: str) -> list[AlpacaDataset]:
    words = split_words(chunk)
    word_count = len(words)

    if word_count < 10:
        return []

    ten_percent = get_slice_size(word_count, 0.10)
    twenty_percent = get_slice_size(word_count, 0.20)

    prefix_10 = join_words(words[:ten_percent])
    rest_after_prefix_10 = join_words(words[ten_percent:])

    prefix_20 = join_words(words[:twenty_percent])
    rest_after_prefix_20 = join_words(words[twenty_percent:])

    prefix_10_suffix_10 = (
        f"Beginning:\n{prefix_10}\n\n"
        f"Ending:\n{join_words(words[-ten_percent:])}"
    )
    middle_without_10_edges = join_words(words[ten_percent:-ten_percent])

    dataset = [
        AlpacaDataset(
            instruction=(
                "You are given the beginning of a passage. "
                "Continue the passage by generating all remaining text after the provided beginning. "
                "Do not repeat the provided beginning."
            ),
            input=prefix_10,
            output=rest_after_prefix_10,
        ),
        AlpacaDataset(
            instruction=(
                "You are given the first 20% of a passage. "
                "Generate the rest of the passage exactly after the provided text. "
                "Do not repeat the provided text."
            ),
            input=prefix_20,
            output=rest_after_prefix_20,
        ),
        AlpacaDataset(
            instruction=(
                "You are given the beginning and the ending of a passage. "
                "Generate only the missing middle section that connects them. "
                "Do not repeat the provided beginning or ending."
            ),
            input=prefix_10_suffix_10,
            output=middle_without_10_edges,
        ),
    ]

    return [row for row in dataset if row.input and row.output]


def main(chunk: str) -> list[AlpacaDataset]:
    return convert_to_dataset(chunk)


if __name__ == "__main__":
    dataset = main(
        """
The dominant sequence transduction models are based on complex recurrent or convolutional
neural networks in an encoder-decoder configuration. The best performing models also connect
the encoder and decoder through an attention mechanism. We propose a new simple network
architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence
and convolutions entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more parallelizable and requiring significantly less time to train.
        """
    )

    print(len(dataset))
