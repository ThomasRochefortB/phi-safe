
# phi-safe

A "GPU-poor" implementation to scale SAFE-GPT to larger autoregressive models like Phi1_5.

Reference to SAFE-GPT:

```
Noutahi, E., Gabellini, C., Craig, M., Lim, J. S., & Tossou, P. (2024). Gotta be SAFE: a new framework for molecular design. *Digital Discovery, 3*(4), 796-804.
```

## Features

- Supports training and fine-tuning of Phi1_5 with 1.3B parmaeters on limited GPU resources.
- Utilizes LORA with all linear layers of the model for parameter efficient training.
- Utilizes SAFE-GPT's tokenizer which means that the token embeddings are trained 
- Uses only 5% of the original SAFE dataset for training.
- Provides options to resume training from checkpoints.

## WIP notes:
- You can visualize and test the generated molecules in the phi-safe-viz.ipynb
- I have implemented a LangChain agent to recreate a very simple version of LOWE in the notebook langchain_experiment.ipynb
- Currently running on a 3090 with batch size of 32.
- Currently puzzled by the slow training speed and large memory requirement of the data input sequence.

---

## Requirements & Installation

Follow the instructions from the SAFE repo:
[SAFE repository](https://github.com/datamol-io/safe)


Make sure you have PEFT installed for LORA:

```bash
pip install peft
```


## Usage

### Training from Scratch

To train the model from scratch, use the following command:

```bash
python phisafe_train.py --dataset_name datamol-io/safe-gpt --train_split "train[:5%]" --eval_split "test[:5%]" --tokenizer_path "./tokenizer.json" --model_id "microsoft/phi-1_5" --model_path "phi1_5_updated" --output_dir ".saved_model/phi1_5-safemol" --max_seq_length 512 --learning_rate 2.0e-05 --max_steps -1 --num_train_epochs 1 --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --bf16 --seed 42
```

### Resuming Training from a Checkpoint

If you want to resume training from a checkpoint, specify the `--checkpoint_path` argument:

```bash
python phisafe_train.py --dataset_name datamol-io/safe-gpt --train_split "train[:5%]" --eval_split "test[:5%]" --tokenizer_path "./tokenizer.json" --model_id "microsoft/phi-1_5" --model_path "phi1_5_updated" --output_dir ".saved_model/phi1_5-safemol" --max_seq_length 512 --learning_rate 2.0e-05 --max_steps -1 --num_train_epochs 1 --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --bf16 --seed 42 --checkpoint_path "/path/to/checkpoint"
```


## License

The original work, including the training dataset and code base, is licensed under the following:

- The training dataset is licensed under CC BY 4.0. See DATA_LICENSE for details.
- The code base is licensed under the Apache-2.0 license. See LICENSE for details.
- The model weights of SAFE-GPT are licensed for research purposes under CC BY-NC 4.0.

The current work is licensed under the same terms.


