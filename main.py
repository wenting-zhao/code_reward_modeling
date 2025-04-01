"""Main STaR Loop"""

from datasets import load_dataset
from inference import generate_predictions
from utils import parse_args, extract_code_blocks, execute_code
from tqdm import tqdm

def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset_name, args.dataset_config_name)
    assert "train" in ds

    all_samples = generate_predictions(
        args.model_name_or_path, args.dataset_name, ds["train"], args.temperature, args.n
    )
    assert len(ds["train"]) == len(all_samples)

    # execute code and get return codes
    all_filtered_samples = []
    all_return_codes = []
    for example, samples in tqdm(zip(ds["train"], all_samples), total=len(ds["train"])):
        filtered_samples = [extract_code_blocks(sample) for sample in samples]
        filtered_samples = [sample[0] for sample in filtered_samples if len(sample) > 0]
        if "mbpp" in args.dataset_name:
            test = "\n\n" + "\n".join(example["test_list"])
        else:
            test = "\n\n" + example["test"]
        code_list = [sample + test for sample in filtered_samples]
        errors = execute_code(code_list, timeout=10)
        all_filtered_samples.append(filtered_samples)
        return_codes = [0 if error is None else 1 for error in errors]
        all_return_codes.append(return_codes)
        print(return_codes)

    ds["train"].add_column(name="samples", column=all_filtered_samples).add_column(name="return_codes", column=all_return_codes).to_json(
            f"{args.output_dir}/samples.json"
    )


if __name__ == "__main__":
    main()


__all__ = []
