import openai
import base64
import requests
import json
import time

from dotenv import load_dotenv
import os
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

image_urls = [
    f"https://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/{i}.jpg"
    for i in range(1, 26)
]

def image_url_to_base64(url):
    try:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        else:
            print(f"Error downloading image: {url}")
            return None
    except Exception as e:
        print(f"Error while downloading {url}: {e}")
        return None

captions = {}

for url in image_urls:
    print(f"Processing: {url}")
    base64_img = image_url_to_base64(url)

    if base64_img is None:
        captions[url] = "Download error"
        continue

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this hotel room in detail: type (single/double/triple), "
                                "number of beds, sea/city/garden view, balcony, air conditioning, furniture, and capacity."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        caption = response.choices[0].message.content.strip()
        captions[url] = caption
        time.sleep(1.5)
    except Exception as e:
        print(f"Error processing {url}: {e}")
        captions[url] = "Vision error"

# JSON dosyasına yaz
with open("captions.json", "w", encoding="utf-8") as f:
    json.dump(captions, f, ensure_ascii=False, indent=2)

print("\n✅ Captions saved to captions.json")
