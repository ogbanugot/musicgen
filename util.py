import os
import shutil


def copy_files_to_root(source_path, dest_path):
    fs = []
    for root, dirs, files in os.walk(source_path):
        for (i, filename) in enumerate(files):
            if os.path.isfile(os.path.join(root, filename)):
                fs.append(os.path.join(root, filename))

    for f in fs:
        print(f"copying {f}")
        shutil.copy(f, dest_path)


def count_files(folder_path):
    file_count = 0

    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count


if __name__ == '__main__':
    source = "/home/pythonuser/project/musicgen/Africa1"
    dest = "/home/pythonuser/project/musicgen/dataset"
    # copy_files_to_root(source, dest)
    count = count_files(source)
    print(count)
