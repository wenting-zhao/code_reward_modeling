"""Main STaR Loop"""

from datasets import load_dataset
from inference import generate_predictions
from utils import parse_args, extract_code_blocks, execute_code
from tqdm import tqdm
import modal

def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset_name, args.dataset_config_name)
    assert "train" in ds

    all_samples = generate_predictions(
        args.model_name_or_path, args.dataset_name, ds["train"], args.temperature, args.n
    )
    assert len(ds["train"]) == len(all_samples)

    # setup modal
    image = modal.Image.debian_slim(python_version="3.12")
    app = modal.App.lookup("safe-code-execution", create_if_missing=True)
    with modal.enable_output():
        sandbox = modal.Sandbox.create(app=app, image=image)

    # execute code
    all_return_codes = []
    all_filtered_samples = []
    for example, samples in tqdm(zip(ds["train"], all_samples), total=len(ds["train"])):
        filtered_samples = [extract_code_blocks(sample) for sample in samples]
        filtered_samples = [sample[0] for sample in filtered_samples if len(sample) > 0]
        if "mbpp" in args.dataset_name:
            test = "\n\n" + "\n".join(example["test_list"])
        else:
            test = "\n\n" + example["test"]
        code_list = [sample + test for sample in filtered_samples]
        _, _, return_codes = execute_code(sandbox, code_list, timeout=5)
        all_return_codes.append(return_codes)
        all_filtered_samples.append(filtered_samples)

    ds["train"].add_column(name="samples", column=all_filtered_samples).add_column(name="return_codes", column=all_return_codes).to_json(
            f"{args.output_dir}/samples.json"
    )

    sandbox.terminate()


if __name__ == "__main__":
    main()


__all__ = []