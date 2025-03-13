import pathlib
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_path", type=str,
                        default=f"{pathlib.Path(__file__).parent.resolve()}/train_images")
    parser.add_argument("--test_image_path", type=str,
                        default=f"{pathlib.Path(__file__).parent.resolve()}/test_images")
    args = parser.parse_args()

    train_image_path = pathlib.Path(args.train_image_path)
    test_image_path = pathlib.Path(args.test_image_path)

    logger.debug(f"Train image path: {train_image_path}")
    logger.debug(f"Test image path: {test_image_path}")

    # Get only image files
    train_images = [img for img in train_image_path.iterdir()
                    if img.is_file() and img.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    test_images = [img for img in test_image_path.iterdir()
                   if img.is_file() and img.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    logger.debug(f"Found {len(train_images)} training images")
    logger.debug(f"Found {len(test_images)} test images")

    cur_location = pathlib.Path(__file__).parent.resolve()
    with open(f"{cur_location}/train_list.txt", "w") as f:
        for img in train_images:
            # Write just the filename, not the full path
            f.write(f"{img.name}\n")

    with open(f"{cur_location}/test_list.txt", "w") as f:
        for img in test_images:
            # Write just the filename, not the full path
            f.write(f"{img.name}\n")

    logger.debug("File lists created successfully")