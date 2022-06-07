import argparse
from probers import *
from embedder import *
from collections import defaultdict
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Location of the dataset. Should be a pandas dataframe with headers",
    )

    parser.add_argument("--device", type=str, help="device num")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = args.device
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device


    pre_t_models = "aajrami/bert-ascii-base,aajrami/bert-mlm-base,aajrami/bert-fc-base,aajrami/bert-rand-base,aajrami/bert-sr-base".split(",")
    debertas = "microsoft/deberta-v3-base,microsoft/deberta-v3-large,microsoft/deberta-v3-xsmall,microsoft/deberta-v3-small".split(",")
    all_models = "nyu-mll/roberta-base-1B-1,nyu-mll/roberta-base-1B-2,nyu-mll/roberta-base-1B-3,nyu-mll/roberta-base-100M-1,nyu-mll/roberta-base-100M-2,nyu-mll/roberta-base-100M-3,nyu-mll/roberta-base-10M-1,nyu-mll/roberta-base-10M-2,nyu-mll/roberta-base-10M-3,nyu-mll/roberta-med-small-1M-1,nyu-mll/roberta-med-small-1M-2,nyu-mll/roberta-med-small-1M-3,roberta-base,roberta-large".split(",")
    all_m = all_models + pre_t_models + debertas


    dataset_location = args.dataset_name
    dataset_name = dataset_location.split("/")[1]

    total = pd.read_csv(dataset_location)

    average_runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with open(f"probes/online_{dataset_name}", "w") as online_filino:

        with open(f"probes/classical_{dataset_name}", "w") as classical_filino:
            for m in all_m:
                embe = Embedder(m)
                embe.create_embeddings(total["text"].values.tolist(),
                                       total["label"].values.tolist(),
                                       list(range(1, embe.model.config.num_hidden_layers+1)),
                                       f"embeddings/test{device}.pkl")

                for r in average_runs:
                    mldprober = MLDProber(embe.model.config.hidden_size)

                    macro_f_dict = mldprober.run(f"embeddings/test{device}.pkl")
                    for layer in macro_f_dict.items():
                        f1 = layer[1]['code_length']
                        loss = layer[1]['sum_of_losses']
                        online_filino.write(f"{m},{r},{layer[0]},{f1},{loss}"+"\n")
                        online_filino.flush()

                for r in average_runs:

                    mldprober = ClassicalProber(embe.model.config.hidden_size)

                    macro_f_dict = mldprober.run(f"embeddings/test{device}.pkl")
                    for layer in macro_f_dict.items():

                        f1 = layer[1]['f1']
                        loss = layer[1]['loss']
                        classical_filino.write(f"{m},{r},{layer[0]},{f1},{loss}\n")
                        classical_filino.flush()

                os.remove(f"embeddings/test{device}.pkl")


if __name__ == "__main__":
    main()
