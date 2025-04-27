import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import json
import plotly.express as px
import plotly_calplot as calplot

# ——— MongoDB setup ———
client     = MongoClient("mongodb://localhost:27017/")
db         = client["attendance_db"]
collection = db["attendance"]

st.set_page_config(page_title="ClasSync Attendance & Engagement Dashboard", layout="wide")
st.title("📊 ClasSync Attendance & Engagement Dashboard")

# ——— Load timetable.json ———
with open("timetable.json") as f:
    timetable = json.load(f)

# ——— Fetch classes ———
docs      = list(collection.find({}))
class_ids = [doc["_id"] for doc in docs]
if not class_ids:
    st.warning("No class data found."); st.stop()

# ——— Class selector ———
class_id  = st.selectbox("Select Class", class_ids)
class_doc = next(doc for doc in docs if doc["_id"] == class_id)

# ——— Attendance Section ———
attendance_data = class_doc.get("attendance", {})
if not attendance_data:
    st.warning("No attendance records."); st.stop()

available_dates    = sorted(attendance_data.keys())
available_dates_dt = [datetime.strptime(d, "%Y-%m-%d").date() for d in available_dates]
today_date         = datetime.today().date()
default_date       = today_date if today_date in available_dates_dt else available_dates_dt[0]

selected_date = st.date_input(
    "Select Date",
    value=default_date,
    min_value=min(available_dates_dt),
    max_value=max(available_dates_dt)
)
date_str = selected_date.strftime("%Y-%m-%d")
if date_str not in attendance_data:
    st.warning(f"No attendance for {date_str}, showing closest instead.")
    date_str = available_dates[0]

day_data = attendance_data[date_str]
subjects = list(day_data.keys())

# Build attendance DataFrame
rows = []
for student in sorted({s for subj in subjects for s in day_data[subj].keys()}):
    total_count = 0
    row = {"Student": student}
    for subj in subjects:
        info = day_data[subj].get(student, {"presence": False, "count": 0})
        # read either timespent or count
        cnt = info.get("timespent", info.get("count", 0))
        row[subj] = "✅" if info.get("presence", False) else "⛔"
        total_count += cnt
    row["Total Recognitions"] = total_count
    rows.append(row)

df_att = pd.DataFrame(rows)

def color_presence(val):
    if val == "✅": return "background-color: lightgreen"
    if val == "⛔": return "background-color: lightblue"
    return ""

st.subheader(f"Attendance for {class_id} on {date_str}")
st.dataframe(df_att.style.applymap(color_presence, subset=subjects), height=300)
present_count = sum(any(r[s]=="✅" for s in subjects) for r in rows)
st.markdown(f"**Total Present:** {present_count} / {len(rows)}")
st.markdown(f"**Subjects Covered:** {', '.join(subjects)}")

# ——— Engagement Section ———
eng = class_doc.get("engagement", {}).get(date_str, {})
if not eng:
    st.warning("No engagement data for this date."); st.stop()

# Build subject→hour lookup from timetable
weekday = selected_date.strftime("%A")
periods = timetable.get(class_id, {}).get(weekday, {})
subj_time_lookup = {
    subj: datetime.strptime(info["start"], "%H:%M").hour
    for subj, info in periods.items()
}

# Flatten engagement
records = []
for subj, stu_dict in eng.items():
    hour = subj_time_lookup.get(subj)
    if hour is None: continue
    for student, emotions in stu_dict.items():
        for emo in emotions:
            engaged = emo in ["happy","surprise","neutral"]
            # … inside your flatten engagement loop …
            records.append({
                "date":    date_str,
                "subject": subj,
                "hour":    hour,
                "emotion": emo,       # capture raw emotion
                "engaged": engaged
            })
df_eng = pd.DataFrame(records)

if df_eng.empty:
    st.warning("No engagement records matching today’s timetable."); st.stop()

# Aggregate ALL dates (for calendar & multi-day avg)
agg_all = (
    df_eng
    .groupby(["date","subject","hour"])["engaged"]
    .agg(total="count", engaged="sum")
    .reset_index()
    .assign(non_engaged=lambda d: d.total - d.engaged,
            rate=lambda d: d.engaged / d.total)
)

# Filter to selected date
today_eng = agg_all[agg_all["date"] == date_str]

st.subheader(f"Engagement Analysis for {class_id} on {date_str}")

# 1) Engagement Rate by Hour (Today)
# 1) Emotion Distribution by Hour (stacked bar)
hourly_emotions = (
    df_eng
    .groupby(["hour","emotion"])
    .size()
    .reset_index(name="count")
)
fig1 = px.bar(
    hourly_emotions,
    x="hour",
    y="count",
    color="emotion",
    title="Emotion Distribution by Hour",
    labels={"hour":"Hour of Day","count":"Number of Faces"}
)
st.plotly_chart(fig1, use_container_width=True)


# 2) Engaged vs Non-Engaged per Subject (Stacked Bar)
fig2 = px.bar(today_eng, x="subject",
              y=["engaged","non_engaged"],
              title="Engaged vs Non-Engaged per Subject (Today)",
              labels={"value":"Count","subject":"Subject"},
              barmode="stack")
st.plotly_chart(fig2, use_container_width=True)

# 3) Subject × Hour Engagement Heatmap (Today)
pivot = today_eng.pivot(index="subject", columns="hour", values="rate").fillna(0)
fig3 = px.imshow(pivot,
                 labels={'x':'Hour','y':'Subject','color':'Rate'},
                 title="Subject × Hour Engagement Heatmap")
st.plotly_chart(fig3, use_container_width=True)

# 4) Daily Engagement Calendar (All Dates)
daily = agg_all.groupby("date")["rate"].mean().reset_index()
fig4 = calplot.calplot(daily, x="date", y="rate",
                       title="Daily Engagement Rate Calendar")
st.plotly_chart(fig4, use_container_width=True)

# 5) Trends by Hour & Subject (Today)
# prepare counts
emotion_counts = (
    df_eng
    .groupby(["subject","emotion"])
    .size()
    .reset_index(name="count")
)

fig5 = px.sunburst(
    emotion_counts,
    path=["subject","emotion"],
    values="count",
    title="Emotion Breakdown by Subject"
)
st.plotly_chart(fig5, use_container_width=True)
