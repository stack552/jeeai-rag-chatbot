"""
Extract lecture scripts from kine_lect{N}_slide{M}.py files,
combine slides per lecture, match to video_id from kinematics_videos.csv,
and save one clean .txt file per lecture.

Folder structure expected:
    C:\\Users\\nsaip\\OneDrive\\Desktop\\Kinematics\\lecture3\\kine_lect{N}_slide{M}.py

Each file contains a variable: script_text = ("...")
"""

import os
import re
import ast
import csv

SCRIPTS_FOLDER = r"C:\Users\nsaip\OneDrive\Desktop\Kinematics\lecture3"
CSV_PATH = r"C:\Users\nsaip\OneDrive\Desktop\Kinematics\kinematics_videos.csv"
OUTPUT_FOLDER = r"C:\Users\nsaip\OneDrive\Desktop\Kinematics\lecture_scripts_combined"

FILENAME_PATTERN = re.compile(r"kine_lect(\d+)_slide(\d+)\.py", re.IGNORECASE)


def extract_script_text(filepath):
    """
    Safely extract the value of `script_text` from a .py file
    without executing the file. Handles both:
        script_text = "a" "b" "c"   (implicit string concatenation)
        script_text = ("a" "b" "c") (parenthesized concatenation)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "script_text":
                    value = ast.literal_eval(node.value)
                    return value
    return None


def collect_lecture_scripts(scripts_folder):
    """
    Returns dict: { lecture_number: { slide_number: text, ... }, ... }
    """
    lectures = {}

    for filename in os.listdir(scripts_folder):
        match = FILENAME_PATTERN.match(filename)
        if not match:
            continue

        lecture_num = int(match.group(1))
        slide_num = int(match.group(2))
        filepath = os.path.join(scripts_folder, filename)

        try:
            text = extract_script_text(filepath)
        except Exception as e:
            print(f"  [WARN] Failed to parse {filename}: {e}")
            continue

        if text is None:
            print(f"  [WARN] No script_text found in {filename}")
            continue

        lectures.setdefault(lecture_num, {})[slide_num] = text

    return lectures


def combine_lecture_text(slide_dict):
    """Concatenate slides in order (slide1, slide2, ...)."""
    ordered_slides = [slide_dict[k] for k in sorted(slide_dict.keys())]
    return "\n\n".join(ordered_slides)


def load_video_csv(csv_path):
    """
    Returns dict: { lecture_number: {"video_id":..., "title":..., "url":...} }
    Matches lecture number by searching for 'Lecture N' in the title text.
    """
    lecture_map = {}
    title_lecture_pattern = re.compile(r"lecture\s*(\d+)", re.IGNORECASE)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["title"]
            match = title_lecture_pattern.search(title)
            if match:
                lecture_num = int(match.group(1))
                lecture_map[lecture_num] = row
            else:
                print(f"  [WARN] Could not find lecture number in title: {title}")

    return lecture_map


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Scanning script files...")
    lectures = collect_lecture_scripts(SCRIPTS_FOLDER)
    print(f"Found scripts for {len(lectures)} lectures.")

    print("Loading video CSV...")
    video_map = load_video_csv(CSV_PATH)
    print(f"Found {len(video_map)} videos in CSV.")

    matched = 0
    unmatched_scripts = []
    unmatched_videos = []

    for lecture_num, slide_dict in sorted(lectures.items()):
        combined_text = combine_lecture_text(slide_dict)

        video_info = video_map.get(lecture_num)
        if video_info is None:
            unmatched_scripts.append(lecture_num)
            # Still save it, just without video_id matched
            out_path = os.path.join(OUTPUT_FOLDER, f"lecture_{lecture_num}_UNMATCHED.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(combined_text)
            continue

        video_id = video_info["video_id"]
        out_filename = f"lecture_{lecture_num}_{video_id}.txt"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Lecture: {lecture_num}\n")
            f.write(f"# Video ID: {video_id}\n")
            f.write(f"# Title: {video_info['title']}\n")
            f.write(f"# URL: {video_info['url']}\n\n")
            f.write(combined_text)

        matched += 1

    # Check which video CSV lectures had no script at all (expected: lectures 1-5)
    for lecture_num in video_map:
        if lecture_num not in lectures:
            unmatched_videos.append(lecture_num)

    print(f"\nDone. Matched and saved {matched} lectures to: {OUTPUT_FOLDER}")

    if unmatched_scripts:
        print(f"\n[WARNING] Scripts found but no matching video in CSV for lecture numbers: {sorted(unmatched_scripts)}")

    if unmatched_videos:
        print(f"\n[INFO] Videos in CSV with no script file found (expected for missing lectures): {sorted(unmatched_videos)}")


if __name__ == "__main__":
    main()