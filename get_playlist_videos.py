"""
Fetch all video IDs + titles from a YouTube playlist.
Requires: pip install google-api-python-client
"""

import csv
from googleapiclient.discovery import build

API_KEY = "AIzaSyBNLPgsVdfbh1js5pFihIXY9k9Utl6Dpak"       # paste your API key
PLAYLIST_ID = "PLC4AeBWYnm8U"    # paste your playlist ID (starts with PL...)
OUTPUT_CSV = "kinematics_videos.csv"

def get_playlist_videos(api_key, playlist_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    videos = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response["items"]:
            video_id = item["contentDetails"]["videoId"]
            title = item["snippet"]["title"]
            position = item["snippet"]["position"]
            if title.strip().lower() in ("private video", "deleted video"):
                continue
            videos.append({
                "position": position,
                "video_id": video_id,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}"
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return videos

if __name__ == "__main__":
    videos = get_playlist_videos(API_KEY, PLAYLIST_ID)
    print(f"Found {len(videos)} videos")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["position", "video_id", "title", "url"])
        writer.writeheader()
        writer.writerows(videos)

    print(f"Saved to {OUTPUT_CSV}")
