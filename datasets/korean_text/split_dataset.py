import pathlib
import argparse
import logging
import asyncio
import shutil
import concurrent.futures
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SplitDataSet:
    def __init__(self, origin_image_path, origin_label_path,
                 train_image_path, train_label_path,
                 test_image_path, test_label_path):
        self.origin_image_path = origin_image_path
        self.origin_label_path = origin_label_path
        self.train_image_path = train_image_path
        self.train_label_path = train_label_path
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path
        # Create a thread pool executor for async file operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor()

    async def split_dataset(self):
        # origin 이미지 파일 리스트 로드
        image_files = [file for file in pathlib.Path(self.origin_image_path).iterdir() if file.is_file()]
        logger.debug(f"{len(image_files)} origin image files loaded")

        # train/test 비율로 분할 (여기서는 75% train, 25% test)
        train_files, test_files = train_test_split(image_files, test_size=0.25, random_state=42)
        logger.debug(f"train_files: {len(train_files)}, test_files: {len(test_files)}")

        # 동시에 train과 test 폴더로 복사 시작
        await asyncio.gather(
            self.copy_dataset(train_files, self.train_image_path, self.train_label_path),
            self.copy_dataset(test_files, self.test_image_path, self.test_label_path)
        )

    async def run_in_thread(self, func, *args):
        """Run a function in a separate thread as a coroutine"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args)

    async def copy_dataset(self, files, target_img_dir, target_label_dir):
        target_img_dir = pathlib.Path(target_img_dir)
        target_label_dir = pathlib.Path(target_label_dir)
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        for image_file in files:
            # 이미지 파일 복사
            dest_img = target_img_dir / image_file.name
            logger.debug(f"Copying {image_file} to {dest_img}")
            await self.run_in_thread(shutil.copy, image_file, dest_img)

            # 이미지에 해당하는 label 파일(.jpg -> .json)
            label_file = pathlib.Path(self.origin_label_path) / f"{image_file.stem}.json"
            if label_file.exists():
                dest_label = target_label_dir / label_file.name
                logger.debug(f"Copying {label_file} to {dest_label}")
                await self.run_in_thread(shutil.copy, label_file, dest_label)
            else:
                logger.warning(f"Label file {label_file} not found!")

    def __del__(self):
        # Ensure the thread pool is shut down properly
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_image_path", type=str,
                        default=f"{pathlib.Path(__file__).parent.resolve()}/origin_images")
    parser.add_argument("--origin_label_path", type=str,
                        default=f"{pathlib.Path(__file__).parent.resolve()}/origin_labels")
    parser.add_argument("--train_image_path", type=str,
                        default=f"{pathlib.Path(__file__).parent.resolve()}/train_images")
    parser.add_argument("--train_label_path", type=str,
                        default=f"{pathlib.Path(__file__).parent.resolve()}/train_labels")
    parser.add_argument("--test_image_path", type=str, default=f"{pathlib.Path(__file__).parent.resolve()}/test_images")
    parser.add_argument("--test_label_path", type=str, default=f"{pathlib.Path(__file__).parent.resolve()}/test_labels")
    args = parser.parse_args()

    dataset_splitter = SplitDataSet(
        args.origin_image_path,
        args.origin_label_path,
        args.train_image_path,
        args.train_label_path,
        args.test_image_path,
        args.test_label_path
    )
    asyncio.run(dataset_splitter.split_dataset())