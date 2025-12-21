import random
import shutil
from pathlib import Path


def rename_images(images_dir: Path, image_ext: str, student_id: str) -> None:
    """
    Rename images in a directory to a unified sequential format.

    New filename format:
        {student_id}_000001.jpg, {student_id}_000002.jpg, ...

    Args:
        images_dir (Path): Directory containing image files.
        image_ext (str): Image file extension, e.g. 'jpg'.
        student_id (str): Student ID used as filename prefix.
    """
    images = sorted(images_dir.glob(f"*.{image_ext}"))

    for idx, img_path in enumerate(images, start=1):
        new_name = f"{student_id}_{idx:06d}.{image_ext}"
        new_path = img_path.with_name(new_name)

        # 使用 move 而不是 rename，避免跨设备错误
        shutil.move(str(img_path), str(new_path))


def expand_images(
    images_dir: Path,
    labels_dir: Path,
    num_copies: int,
    image_ext: str,
    label_ext: str
) -> None:
    """
    Expand dataset by duplicating images and corresponding labels.

    Example:
        image.jpg -> image_0.jpg, image_1.jpg, ...
        label.txt -> image_0.txt, image_1.txt, ...

    Args:
        images_dir (Path): Image directory.
        labels_dir (Path): Label directory.
        num_copies (int): Number of copies per image.
        image_ext (str): Image extension.
        label_ext (str): Label extension.
    """
    images = list(images_dir.glob(f"*.{image_ext}"))

    for img_path in images:
        label_path = labels_dir / img_path.with_suffix(f".{label_ext}").name

        if not label_path.exists():
            print(f"[Warning] Missing label for image: {img_path.name}")
            continue

        for i in range(num_copies):
            img_copy = img_path.with_name(f"{img_path.stem}_{i}.{image_ext}")
            label_copy = label_path.with_name(f"{img_path.stem}_{i}.{label_ext}")

            shutil.copy(img_path, img_copy)
            shutil.copy(label_path, label_copy)


def split_dataset(
    root_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    train_ratio: float,
    val_ratio: float,
    image_ext: str,
    label_ext: str
) -> None:
    """
    Split dataset into train / val / test sets and generate YOLO txt files.

    Args:
        root_dir (Path): Dataset root directory.
        images_dir (Path): Image directory.
        labels_dir (Path): Label directory.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        image_ext (str): Image extension.
        label_ext (str): Label extension.
    """
    images = list(images_dir.glob(f"*.{image_ext}"))
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    # 清空旧文件，避免重复写入
    for split in splits:
        (root_dir / f"{split}.txt").write_text("")

    for split, img_list in splits.items():
        txt_path = root_dir / f"{split}.txt"

        with open(txt_path, "a") as f:
            for img_path in img_list:
                label_path = labels_dir / img_path.with_suffix(f".{label_ext}").name

                if not label_path.exists():
                    print(f"[Warning] Missing label for image: {img_path.name}")
                    continue

                # YOLO 要求写入 image 的绝对或相对路径
                f.write(f"{img_path.as_posix()}\n")


if __name__ == "__main__":
    # ===================== 基本路径配置 =====================
    root_dir = Path("/home/ubuntu/Documents/Project/gulp/yolov5_test/datasets/Campus")
    images_dir = root_dir / "images"
    labels_dir = root_dir / "labels"

    # ===================== 数据集参数 =====================
    image_extension = "jpg"
    label_extension = "txt"
    student_id = "2020123456"

    train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2

    # ===================== 功能开关 =====================
    # rename_images(images_dir, image_extension, student_id)
    # expand_images(images_dir, labels_dir, 5, image_extension, label_extension)
    split_dataset(
        root_dir,
        images_dir,
        labels_dir,
        train_ratio,
        val_ratio,
        image_extension,
        label_extension
    )
