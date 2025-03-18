from pathlib import Path
import os
import argparse
import zipfile
import tempfile
import gdown
from tqdm import tqdm
from contextlib import contextmanager


@contextmanager
def temporary_file(suffix=None):
    """Context manager for temporary files that ensures deletion."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield tmp.name
    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def get_zip_size(zip_path):
    """Get the total uncompressed size of all files in the zip."""
    total_size = 0
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            total_size += file_info.file_size
    return total_size


def download_with_gdown(url, output_path):
    """Download a file from Google Drive with progress bar."""
    try:
        # gdown handles progress bar internally
        output = gdown.download(url, output_path, fuzzy=True)
        if output is None:
            print(f"Failed to download from {url}.")
            return False
        return True
    except Exception as e:
        print(f"Error during download: {e}")
        return False


def extract_with_progress(zip_path, extract_path):
    """Extract zip file with progress bar."""
    try:
        total_size = get_zip_size(zip_path)
        extracted_size = 0

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Create progress bar for extraction
            with tqdm(
                    desc="Extracting",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for file_info in zip_ref.infolist():
                    zip_ref.extract(file_info, extract_path)
                    extracted_size += file_info.file_size
                    pbar.update(file_info.file_size)

        print(f"Successfully extracted to {extract_path}")
        return True
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download and extract dataset resources')
    parser.add_argument('--raw-dataset', action='store_true', help='Download raw dataset')
    parser.add_argument('--processed-dataset', action='store_true', help='Download processed dataset')
    parser.add_argument('--pretrained-artifacts', action='store_true', help='Download pretrained artifacts')
    parser.add_argument('--case-studies', action='store_true', help='Download case studies')
    parser.add_argument('-a', '--all', action='store_true', help='Download all resources')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for extracted files')

    args = parser.parse_args()

    resources = {
        "raw-dataset": "https://drive.google.com/file/d/13PjOB2recWbWKf-w7ORSInChcSQcpZhx/view?usp=sharing",
        "processed-dataset": "https://drive.google.com/file/d/1LAjcOhtGn0yiUtkssKf29HOc9Q9lfJLw/view?usp=sharing",
        "pretrained-artifacts": "https://drive.google.com/file/d/1xB5ZEhNrdh-Zm5A9vqvZqV-BQY6nEpjf/view?usp=sharing",
        "case-studies": "https://drive.google.com/file/d/1xuP-7FJ7OzBkA7oSgenbfwBcetXHWfc0/view?usp=sharing",
    }

    # Determine which resources to download
    to_download = []
    if args.all:
        to_download = list(resources.keys())
    else:
        if args.raw_dataset:
            to_download.append("raw-dataset")
        if args.processed_dataset:
            to_download.append("processed-dataset")
        if args.pretrained_artifacts:
            to_download.append("pretrained-artifacts")
        if args.case_studies:
            to_download.append("case-studies")

    if not to_download:
        parser.print_help()
        return

    # Create output directory if it doesn't exist
    script_dir = Path(__file__).resolve().parent
    project_folder = script_dir.parent
    output_dir = project_folder / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract each resource
    for resource in to_download:
        url = resources[resource]
        if not url:  # Skip if URL is empty
            print(f"No URL provided for {resource}, skipping...")
            continue

        print(f"\nProcessing {resource}...")

        # Use context manager for temporary file
        with temporary_file(suffix='.zip') as temp_path:
            if download_with_gdown(url, temp_path):
                extract_with_progress(temp_path, output_dir)


if __name__ == "__main__":
    main()