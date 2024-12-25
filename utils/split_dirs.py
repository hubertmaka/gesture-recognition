import os
import shutil
import logging

logger = logging.getLogger(__name__)

main_directory = os.path.expanduser('../../../datasets/jester_Dataset')

batch_size = 9000

directories = sorted(
    [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d)) and d.isdigit()],
    key=lambda x: int(x)
)

num_batches = (len(directories) + batch_size - 1) // batch_size

for batch_num in range(num_batches):
    new_dir_name = f"kat{batch_num + 1}"
    new_dir_path = os.path.join(main_directory, new_dir_name)

    os.makedirs(new_dir_path, exist_ok=True)

    batch_directories = directories[batch_num * batch_size:(batch_num + 1) * batch_size]

    for dir_name in batch_directories:
        old_path = os.path.join(main_directory, dir_name)
        new_path = os.path.join(new_dir_path, dir_name)
        shutil.move(old_path, new_path)

logger.info("Splitting done.")
