import os
import json

comments = []

for filename in os.listdir('./comments-json'):
    if filename.endswith('.json'):
        with open(os.path.join('./comments-json', filename),
                  'r', encoding='utf-8') as f:
            data = json.load(f)
            comments.extend(
                [
                    comment['text']
                    for comment in data.get('comments', [])
                    ]
                    )

with open('comments.txt', 'w', encoding='utf-8') as f:
    for comment in comments:
        f.write(comment + '\n\n\n')
