import tempfile
import glob
import shutil
import os

from cleanfid import fid


def copy_recursive_file_pattern(src_folder: str, fname_pattern: str, dst_folder: str):
    for file in glob.iglob(f'**/{fname_pattern}', root_dir=src_folder, recursive=True):
        from_path = os.path.join(src_folder, file)
        to_path = os.path.join(dst_folder, file)
        os.makedirs(os.path.dirname(to_path), exist_ok=True)
        shutil.copy(from_path, to_path)


def compute_fid(
    first_path: str,
    first_fname: str,
    second_path: str,
    second_fname: str,
):
    with tempfile.TemporaryDirectory(suffix='mmsteer_metrics') as tmp_dir:
        first_temp_path = os.path.join(tmp_dir, 'first')
        second_temp_path = os.path.join(tmp_dir, 'second')
        copy_recursive_file_pattern(first_path, first_fname, first_temp_path)
        copy_recursive_file_pattern(second_path, second_fname, second_temp_path)
        fid_value = fid.compute_fid(first_temp_path, second_temp_path)
        return fid_value
