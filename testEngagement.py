import requests

# --- Configuration ---
BASE_URL   = "http://localhost:5000"
ENDPOINT   = "/engagement"
IMAGE_PATH = "gp2.jpg"      # replace with your test image
CLASS_ID   = "AI&DS 104"          # replace with a classid that has an active period

def test_engagement(classid: str, image_path: str):
    """
    Sends a single POST to /engagement with the given classid and image.
    """
    url = BASE_URL + ENDPOINT
    data = {'classid': classid}
    files = {'image': open(image_path, 'rb')}
    try:
        resp = requests.post(url, data=data, files=files)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return

    print(f"Status Code: {resp.status_code}")
    try:
        print("Response JSON:", resp.json())
    except ValueError:
        print("Non-JSON response:", resp.text)

def main():
    # Single test
    print(f"Testing engagement for class '{CLASS_ID}' with image '{IMAGE_PATH}'…")
    test_engagement(CLASS_ID, IMAGE_PATH)

    # Example: loop over multiple images
    # images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    # for img in images:
    #     test_engagement(CLASS_ID, img)

if __name__ == "__main__":
    main()
