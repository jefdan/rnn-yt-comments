import subprocess
import uuid

with open('./links.txt', 'r') as file:
    links = file.readlines()

for link in links:
    link = link.strip()
    if link:
        output_filename = f"./comments-json/{uuid.uuid4()}"
        subprocess.run([
            "./yt-dlp/yt-dlp.exe",
            link,
            "--output", output_filename,
            "--write-comments",
            "--skip-download"
        ])
