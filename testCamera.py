import cv2
import time
import requests
from datetime import datetime, timedelta

# ——— CONFIG ———
API_BASE            = "http://localhost:5000"
CLASS_ID            = "AI&DS 104"
TIMETABLE_ENDPOINT  = f"{API_BASE}/timetable"
ATTENDANCE_ENDPOINT = f"{API_BASE}/attendance"
ENGAGEMENT_ENDPOINT = f"{API_BASE}/engagement"

FETCH_INTERVAL      = timedelta(minutes=10)
ATTEND_INTERVAL     = timedelta(minutes=5)
ENGAGE_INTERVAL     = timedelta(minutes=1)

# ——— SETUP ———
session = requests.Session()
cap     = cv2.VideoCapture(0)  # use camera’s native resolution

def find_current_subject(timetable, classid, now):
    day = now.strftime("%A")
    for subj, info in timetable.get(classid, {}).get(day, {}).items():
        start = datetime.strptime(info['start'], "%H:%M").time()
        end   = datetime.strptime(info['end'],   "%H:%M").time()
        if start <= now.time() <= end:
            return subj
    return None

def send_frame(endpoint, frame):
    # ensure BGR color
    if frame.ndim == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # encode & send
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    files = {'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
    data  = {'classid': CLASS_ID}

    try:
        r = session.post(endpoint, data=data, files=files, timeout=10)
        print(f"[{datetime.now():%H:%M:%S}] {endpoint.split('/')[-1]} → {r.json()}")
    except Exception as e:
        print(f"[ERROR] {endpoint}: {e}")

def main():
    if not cap.isOpened():
        print("Cannot open camera"); return

    timetable   = {}
    now         = datetime.now()
    next_fetch  = now
    next_att    = now
    next_engage = now

    try:
        while True:
            now = datetime.now()

            # 1) Fetch timetable
            if now >= next_fetch or not timetable:
                try:
                    resp = session.get(TIMETABLE_ENDPOINT, timeout=5)
                    resp.raise_for_status()
                    timetable = resp.json()
                    print(f"[{now:%H:%M:%S}] timetable updated")
                except Exception as e:
                    print(f"[ERROR] fetch timetable: {e}")
                next_fetch = now + FETCH_INTERVAL

            # 2) Check for active class
            subj = find_current_subject(timetable, CLASS_ID, now)
            if subj:
                # attendance
                if now >= next_att:
                    ret, frame = cap.read()
                    if ret:
                        send_frame(ATTENDANCE_ENDPOINT, frame)
                    next_att = now + ATTEND_INTERVAL

                # engagement
                if now >= next_engage:
                    ret, frame = cap.read()
                    if ret:
                        send_frame(ENGAGEMENT_ENDPOINT, frame)
                    next_engage = now + ENGAGE_INTERVAL
            else:
                # reset timers to fire immediately when class starts
                next_att    = now
                next_engage = now

            # 3) Sleep until the nearest upcoming event
            upcoming = [next_fetch]
            if subj:
                upcoming += [next_att, next_engage]
            wakeup = min(upcoming)
            delay = (wakeup - datetime.now()).total_seconds()
            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print("Shutting down…")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
