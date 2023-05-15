# vall-e-encodec

## generated result
https://github.com/voidful/vall-e-encodec/assets/10904842/b53ad6b3-998c-4828-b93f-368931372d14



## Baseline text to mhubert unit
https://huggingface.co/voidful/mhubert-unit-tts

To train the mhubert model for generating text to unit, follow the steps below:

1. Install the `tfkit` package using `pip`:
```
pip install tfkit
```


2. Create a unit dataset by running `speech2unit.py`. The `speech2unit.py` script generates `train.csv` and `test.csv` files that will be used for training and testing respectively. You can run the script with the following command:
```css
python speech2unit.py --model mhubert_layer11_code1000 --ds superb --ds_split asr
```
Alternatively, you can create `train.csv` and `test.csv` manually, with `code_str = "".join(f"v_tok_{tok}" for tok in code)`.


3. Train the mhubert model to generate text to unit using the `tfkit-train` command. The `--epoch` parameter specifies the number of epochs to train the model for, `--handle_exceed` removes the exceeded samples, `--train` specifies the path of the training dataset, `--test` specifies the path of the testing dataset, `--no_eval` disables the evaluation of the model during training, `--task` specifies the type of task as "seq2seq", `--config` specifies the configuration of the model to use, `--worker` specifies the number of workers to use, `--grad_accum` specifies the number of gradient accumulation steps, `--batch` specifies the batch size, and `--wandb` logs the training process to Weights and Biases.
```css
tfkit-train \
--epoch 20 \
--handle_exceed remove \
--train ./text_to_mhubert/train.csv \
--test ./text_to_mhubert/test.csv \
--no_eval \
--task seq2seq \
--config voidful/bart-base-unit \
--worker 15 \
--grad_accum 2 \
--batch 3 \
--wandb
```


4. (Optional) Evaluate the generated unit with the ground truth unit using the `tfkit-eval` command. The `--model` parameter specifies the path of the trained model checkpoint, `--valid` specifies the path of the validation dataset, and `--metric` specifies the evaluation metric as "er".
```css
tfkit-eval --model ./checkpoints/ --valid ./text_to_mhubert/test.csv --metric er
```

5. Dump the trained model in Huggingface's format using the `tfkit-dump` command. The `--model` parameter specifies the path of the trained model checkpoint, and `--output` specifies the path of the dumped model.
```bash
tfkit-dump --model ./checkpoints/5.pt --output ./dumped_model
```
