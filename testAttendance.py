import requests

# Endpoint URL
URL = "http://localhost:5000/attendance"

# Replace with the classid you want to test
payload = {
    'classid': 'AI&DS 104'
}

# Path to your sample image
files = {
    'image': open('gp2.jpg', 'rb')
}

def main():
    try:
        resp = requests.post(URL, data=payload, files=files)
        resp.raise_for_status()
        print("Status Code:", resp.status_code)
        print("Response JSON:", resp.json())
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

if __name__ == "__main__":
    main()