
from pydantic_core.core_schema import model_ser_schema
from pydantic import BaseModel, Field
from utils import AlpacaDataset, save_dataset
from model import generate_structured_output, augment_data


class Comparisons(BaseModel):
    answer: str = Field(max_length=500)
                               


system_prompt = """
Given 2 passages of text, generate a detailed comparison of the two
    """

def convert_to_dataset(passage, out):
    dataset = []

    dataset.append(
        AlpacaDataset(
            instruction=system_prompt,
            input=passage,
            output=out.answer
        )
    )
    return dataset


def main(passage: str):
    messages = [
        {
            "role": "system", "content": system_prompt
        },
        {
            "role": "user", "content": passage 
        }
    ]

    dataset = [] 
    for _ in range(3):
        out = generate_structured_output(messages, 
                                         Comparisons, 
                                         temp=0.5)
        print(out) 
        dataset = dataset + convert_to_dataset(passage, out)

    return dataset

if __name__ == "__main__":

    dataset = main("""
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. 

Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
             """)

    print(len(dataset))
