import os
import zipfile

main_directory = os.path.expanduser('../datasets/jester_Dataset')

directories = sorted([d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))])


def zip_directory(directory_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)


for dir_name in directories:
    dir_path = os.path.join(main_directory, dir_name)
    zip_file_name = f"{dir_name}.zip"
    zip_file_path = os.path.join(main_directory, zip_file_name)

    zip_directory(dir_path, zip_file_path)

    print(f"Zipped: {zip_file_name}")

print("Archiving done.")
