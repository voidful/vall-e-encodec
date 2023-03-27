import argparse
import sys

import jsonlines
import torch
from datasets import load_dataset

from speech2unit_model.hubert import hubert_layer9_code500, hubert_layer6_code50, hubert_layer6_code100, \
    hubert_layer6_code200
from speech2unit_model.mhubert import mhubert_layer11_code1000

ModelMap = {
    'hubert_layer9_code500': hubert_layer9_code500,
    'hubert_layer6_code50': hubert_layer6_code50,
    'hubert_layer6_code100': hubert_layer6_code100,
    'hubert_layer6_code200': hubert_layer6_code200,
    'mhubert_layer11_code1000': mhubert_layer11_code1000,
}


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ModelMap.keys(), required=True, help="model name")
    parser.add_argument("--chunk_sec", type=int, default=30, help="chunk sec, default 30")
    parser.add_argument("--feat_norm", action="store_true", help="normalize feature")
    parser.add_argument("--beamsearch", action="store_true", help="enable beamsearch")
    parser.add_argument("--topk", type=int, default=3, help="topk, default 3")
    parser.add_argument("--beamsize", type=int, default=1, help="beamsize, default 1")
    parser.add_argument("--ds", default='superb', type=str)
    parser.add_argument("--ds_split", default='asr', type=str)
    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    model_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
    return input_arg, model_arg


def main(arg=None):
    input_arg, model_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    dataset = load_dataset(input_arg['ds'], input_arg['ds_split'])
    hc_model = ModelMap[input_arg['model']]()

    def convert_code_fn(data):
        hubert = hc_model(input_values=torch.from_numpy(data['audio']['array'].astype('float32')).unsqueeze(0),
                          feat_norm=input_arg['feat_norm'],
                          beamsearch=input_arg['beamsearch'],
                          top_k=100,
                          beamsize=5)
        data.update(hubert)
        return data

    new_ds = dataset.map(convert_code_fn)

    try:
        new_ds = new_ds.remove_columns(['file'])
    except:
        pass
    try:
        new_ds = new_ds.remove_columns(['filename'])
    except:
        pass

    new_ds = new_ds.remove_columns(['audio'])

    for k, v in new_ds.items():
        json_items = []
        for d in v:
            json_items.append(d)

        with jsonlines.open(
                f'./{input_arg["ds"].replace("/", "_")}_{input_arg["ds_split"]}_{k}_chunk_{input_arg["chunk_sec"]}_{input_arg["model"]}_norm_{input_arg["feat_norm"]}_beam_{input_arg["beamsearch"]}_topk_{input_arg["topk"]}_beamsize_{input_arg["beamsize"]}.jsonl',
                mode='w') as writer:
            writer.write_all(json_items)


if __name__ == "__main__":
    main()
