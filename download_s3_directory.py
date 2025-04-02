import argparse
import random
import os
import boto3
from tqdm import tqdm
import math


def get_args():
    parser = argparse.ArgumentParser(description="Fetch a subset of audio files from S3.")
    parser.add_argument("--bucket", type=str, required=True, help="S3 버킷 이름")
    parser.add_argument("--prefix", type=str, default="", help="목록을 받아올 S3 경로(prefix). 일반적으로 'audios/' 등 사용")
    parser.add_argument("--dst", type=str, default="downloaded_audios", help="오디오를 저장할 로컬 디렉터리")
    parser.add_argument("--max_files", type=int, default=30000, help="가져올 파일 개수 (기본값 = 30000)")
    parser.add_argument("--sample_method", type=str, default="efficient", choices=["full", "efficient"],
                        help="샘플링 방식: full(전체 목록), efficient(효율적 샘플링)")
    parser.add_argument("--exclude_subfolders", action="store_true",
                        help="하위 폴더를 제외하고 상위 경로 파일만 다운로드")
    return parser.parse_args()


def is_top_level_file(key, prefix):
    """
    주어진 key가 prefix 바로 아래의 파일인지 확인합니다.
    하위 폴더 내의 파일은 제외합니다.
    """
    # prefix가 '/'로 끝나지 않으면 '/'를 추가
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'

    # prefix를 제거한 나머지 경로
    remaining_path = key[len(prefix):] if key.startswith(prefix) else key

    # 남은 경로에 '/'가 없으면 상위 경로의 파일입니다
    return '/' not in remaining_path


def count_s3_objects(bucket_name, prefix, exclude_subfolders=False):
    """S3 버킷 내 객체 수를 대략적으로 추정합니다."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator('list_objects_v2')

    # 몇 개의 페이지만 확인하여 전체 개수 추정
    page_size = 1000
    total_objects = 0
    pages_checked = 0
    max_pages_to_check = 5

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix,
                                   PaginationConfig={'MaxItems': page_size * max_pages_to_check}):
        if 'Contents' in page:
            if exclude_subfolders:
                filtered_objects = [obj for obj in page['Contents'] if is_top_level_file(obj['Key'], prefix)]
                total_objects += len(filtered_objects)
            else:
                total_objects += len(page['Contents'])
        pages_checked += 1
        if pages_checked >= max_pages_to_check:
            break

    # 샘플링된 페이지의 평균을 사용해 전체 예상치 계산
    if pages_checked > 0:
        avg_objects_per_page = total_objects / pages_checked
        # 정확한 추정은 어렵지만, 우리는 대략적인 수치만 필요함
        return int(avg_objects_per_page * 1000)  # 추정치
    return 0


def efficient_sampling(bucket_name, prefix, max_files, exclude_subfolders=False):
    """
    전체 목록을 가져오지 않고 효율적으로 파일을 샘플링합니다.
    """
    s3 = boto3.client("s3")

    # 총 객체 수 추정
    estimated_total = count_s3_objects(bucket_name, prefix, exclude_subfolders)
    if estimated_total == 0:
        print("파일을 찾을 수 없습니다.")
        return []

    print(f"추정된 총 파일 수: 약 {estimated_total:,}개")

    # 샘플링 비율 계산
    sample_rate = min(1.0, max_files * 1.2 / estimated_total)  # 목표보다 20% 더 가져옴

    paginator = s3.get_paginator('list_objects_v2')
    selected_keys = []

    print("파일 샘플링 중...")
    with tqdm(total=max_files, desc="샘플링 진행률") as pbar:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                # 하위 폴더 제외 옵션이 켜져 있으면 상위 경로 파일만 필터링
                if exclude_subfolders and not is_top_level_file(obj['Key'], prefix):
                    continue

                # 샘플링 비율에 따라 선택
                if random.random() <= sample_rate:
                    selected_keys.append(obj['Key'])
                    pbar.update(min(1, max_files - len(selected_keys) + 1))

                # 충분히 모았으면 중단
                if len(selected_keys) >= max_files:
                    return random.sample(selected_keys, max_files)

    # 목표보다 적게 모았다면 전부 반환
    return selected_keys


def list_s3_files(bucket_name, prefix, exclude_subfolders=False):
    """bucket_name 버킷의 prefix 경로 아래 모든 파일(Key) 목록을 반환합니다."""
    s3 = boto3.client("s3")
    keys = []

    # 진행 상황 표시를 위한 초기 설정
    print("모든 S3 파일 목록 가져오는 중...")

    continuation_token = None
    page_count = 0

    with tqdm(desc="파일 목록 가져오기") as pbar:
        while True:
            if continuation_token:
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, ContinuationToken=continuation_token)
            else:
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

            if "Contents" in response:
                if exclude_subfolders:
                    # 하위 폴더를 제외하고 상위 경로 파일만 필터링
                    filtered_keys = [obj["Key"] for obj in response["Contents"]
                                     if is_top_level_file(obj["Key"], prefix)]
                    keys.extend(filtered_keys)
                    items_count = len(filtered_keys)
                else:
                    # 모든 파일 포함
                    keys.extend([obj["Key"] for obj in response["Contents"]])
                    items_count = len(response["Contents"])

                pbar.update(items_count)
                pbar.set_postfix(files=len(keys))

            page_count += 1
            if page_count % 10 == 0:
                print(f"현재까지 {len(keys):,}개 파일 목록 수집됨...")

            if response.get("IsTruncated"):
                continuation_token = response.get("NextContinuationToken")
            else:
                break

    return keys


def download_files(bucket_name, keys, dst_dir):
    """주어진 keys 목록의 파일을 모두 dst_dir 로 다운로드합니다."""
    s3 = boto3.client("s3")
    os.makedirs(dst_dir, exist_ok=True)

    print(f"총 {len(keys)}개 파일 다운로드 시작...")
    for key in tqdm(keys, desc="다운로드 진행률"):
        filename = os.path.basename(key)
        local_path = os.path.join(dst_dir, filename)
        s3.download_file(bucket_name, key, local_path)


def main():
    args = get_args()

    if args.exclude_subfolders:
        print("하위 폴더를 제외하고 상위 경로 파일만 다운로드합니다.")

    if args.sample_method == "efficient":
        print("효율적인 샘플링 방식으로 파일을 선택합니다...")
        selected_keys = efficient_sampling(args.bucket, args.prefix, args.max_files, args.exclude_subfolders)
    else:
        print("전체 파일 목록에서 무작위 선택합니다...")
        all_keys = list_s3_files(args.bucket, args.prefix, args.exclude_subfolders)

        if len(all_keys) == 0:
            print("해당 버킷/프리픽스에서 파일을 찾을 수 없습니다.")
            return

        print(f"총 {len(all_keys):,}개의 파일이 있습니다.")
        if len(all_keys) <= args.max_files:
            selected_keys = all_keys
            print("모든 파일을 다운로드합니다.")
        else:
            print(f"{args.max_files:,}개 파일을 무작위로 선택합니다...")
            selected_keys = random.sample(all_keys, args.max_files)

    if not selected_keys:
        print("선택된 파일이 없습니다.")
        return

    print(f"{len(selected_keys):,}개 파일을 다운로드합니다...")
    download_files(args.bucket, selected_keys, args.dst)
    print("다운로드가 완료되었습니다.")


if __name__ == "__main__":
    main()