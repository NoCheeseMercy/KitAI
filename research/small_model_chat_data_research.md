# Small-Model Chat Data Research

## Context

The existing OASST1 and WikiText run behaves as a plain next-token model rather than a reliable assistant. The observed output continues complete serialized conversations because its runtime prompt did not match the training format and its response loss was not isolated from user/template tokens. A larger model alone will not correct that mismatch.

## Practitioner evidence

A LocalLLaMA fine-tuning discussion recommends training on diverse examples with clear inputs and outputs, rather than dumping every available record into the model. It distinguishes instruction/chat adaptation from adding broad knowledge and specifically cautions that more data is not automatically better. [1]

A second LocalLLaMA thread catalogues high-quality supervised fine-tuning datasets, pointing practitioners toward a curated-dataset approach rather than an unrestricted web scrape. [2]

A reported 130M model trained locally used **billions of pretraining tokens** plus a separate curated instruction-tuning stage. This supports the distinction between base pretraining and assistant post-training, rather than mixing low-quality conversations into general text. [3]

## Candidate sources

| Dataset | Intended use | Size / evidence | License / caveat | Recommendation |
|---|---|---:|---|---|
| `HuggingFaceTB/smol-smoltalk` | Assistant post-training for **135M and 360M** SmolLM2 models | Small-model-specific subset of SmolTalk | Dataset card points to component licenses; inspect the selected subset before redistribution | **Primary SFT candidate** after strict filtering and assistant-only loss masking |
| `HuggingFaceTB/smoltalk` selected subsets | Curated SFT; the recipe was used for SmolLM2-Instruct models | 1.04M training examples in the full collection | New core subsets Apache-2.0; some included public subsets retain their own terms | Use only small-model-appropriate, non-tool, non-chain-of-thought subsets |
| `HuggingFaceH4/ultrachat_200k` | Supervised chat training | 207,865 SFT training conversations; used for Zephyr-7B-beta and linked from TinyLlama-1.1B-Chat | MIT | Reserve a filtered, limited portion for conversational diversity; do not use unfiltered whole set |
| `allenai/tulu-3-sft-mixture` | Broad SFT mix | 939k conversations | ODC-BY; component data has varied terms | Optional research-only source; not the initial local mix due license and breadth |

## Training requirements inferred from the failure

1. Train on a serialized format such as `<|user|> ... <|assistant|> ... <|end|>` **and use the exact same template at inference**.
2. Mask loss for the user, system, and delimiter tokens; optimize only assistant reply tokens.
3. Set `<|end|>` as an explicit stop token in Ollama so a generated answer cannot roll into the next stored conversation.
4. Include explicit greetings and short conversational examples in the SFT set, but do not expect raw web pretraining to teach chat behavior.
5. Keep a held-out “Hello / How are you / Who are you?” test suite and require it to pass before export.

## References

[1] https://www.reddit.com/r/LocalLLaMA/comments/1ilkamr/a_comprehensive_overview_of_everything_i_know/

[2] https://www.reddit.com/r/LocalLLaMA/comments/1cg2ce7/llm_datasets_a_curated_list_of_datasets_for/

[3] https://www.reddit.com/r/LocalLLaMA/comments/1n5j783/i_built_pretrained_and_finetuned_a_small_language/

[4] https://huggingface.co/datasets/HuggingFaceTB/smoltalk

[5] https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

[6] https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
