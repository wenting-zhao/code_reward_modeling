"""Main STaR Loop"""

from datasets import load_dataset
from inference import generate_predictions
from utils import parse_args


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset_name, args.dataset_config_name)
    assert "train" in ds

    model_name = args.model_name_or_path
    # sample
    all_samples = generate_predictions(
        model_name, ds["train"], args.temperature, args.n
    )
    ds["train"].add_column(name="sample", column=all_samples).to_json(
        f"{args.output_dir}/data/samples.json"
    )
    assert len(ds["train"]) == len(all_samples)


if __name__ == "__main__":
    main()


__all__ = []