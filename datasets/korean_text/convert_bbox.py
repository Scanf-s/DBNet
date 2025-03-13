import json
import os
import pathlib
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def convert_bbox(json_file, output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # info 필드의 name을 기반으로 올바른 파일명을 구성 (예: "책표지_문학_002325.jpg")
    expected_filename = data.get("info", {}).get("name", "") + ".jpg"

    # 여러 이미지가 존재할 경우, expected_filename과 일치하는 이미지 정보만 선택
    images_info = {}
    for img in data['images']:
        if img.get("file_name") == expected_filename:
            images_info[img.get("id")] = img
            break
    # 만약 expected_filename과 일치하는 항목이 없으면 기존 방식으로 fallback
    if not images_info:
        logger.warning(f"Expected image {expected_filename} not found, falling back to first image for each id.")
        for img in data['images']:
            images_info.setdefault(img.get("id"), img)

    # annotation 처리: 각 annotation의 image_id에 해당하는 이미지 정보 사용
    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        text = ann['text']
        image_id = ann['image_id']
        image_info = images_info.get(image_id)
        if image_info is None:
            logger.warning(f"No image info for image_id {image_id}, skipping annotation.")
            continue

        # 네 꼭지점 좌표 계산
        x2 = x + w
        y2 = y
        x3 = x + w
        y3 = y + h
        x4 = x
        y4 = y + h

        # 최종 문자열 생성
        line = f"{x},{y},{x2},{y2},{x3},{y3},{x4},{y4},{text}"

        # txt 파일 이름은 해당 이미지 파일 이름을 기준으로 함
        file_name = image_info.get('file_name')
        txt_filename = os.path.join(output_dir, file_name + '.txt')

        # 여러 bbox가 있을 수 있으므로 append 모드로 기록
        with open(txt_filename, 'a', encoding='utf-8') as fw:
            fw.write(line + "\n")

    # 모든 이미지에 대해, annotation이 없어서 txt 파일이 생성되지 않은 경우 빈 파일 생성
    for img in data['images']:
        file_name = img.get('file_name')
        txt_filename = os.path.join(output_dir, file_name + '.txt')
        if not os.path.exists(txt_filename):
            logger.debug(f"{file_name}에 대한 annotation이 없어 빈 gt 파일을 생성합니다.")
            open(txt_filename, 'w', encoding='utf-8').close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_label_path", type=str,
                        default=f"{pathlib.Path(__file__).parent}/train_labels")
    parser.add_argument("--test_label_path", type=str,
                        default=f"{pathlib.Path(__file__).parent}/test_labels")
    parser.add_argument("--train_output_dir", type=str,
                        default=f"{pathlib.Path(__file__).parent}/train_gts")
    parser.add_argument("--test_output_dir", type=str,
                        default=f"{pathlib.Path(__file__).parent}/test_gts")
    args = parser.parse_args()

    train_json_files = [file for file in pathlib.Path(args.train_label_path).glob('*.json')]
    test_json_files = [file for file in pathlib.Path(args.test_label_path).glob('*.json')]

    logger.debug(f"Train json files: {len(train_json_files)}, Test json files: {len(test_json_files)}")

    # Process train files
    for json_file in train_json_files:
        convert_bbox(json_file, args.train_output_dir)

    # Process test files
    for json_file in test_json_files:
        convert_bbox(json_file, args.test_output_dir)

    logger.debug("Conversion completed!")
