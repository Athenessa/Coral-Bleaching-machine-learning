# -*- coding: utf-8 -*-
"""
พล็อตจุดสำรวจการฟอกขาวของปะการังบนแผนที่โลกจริง
ใช้ข้อมูล data/coral_clean_global.csv จาก repo ของโปรเจกต์

รันบน Google Colab:
    !pip install -q cartopy plotly
    แล้ว copy โค้ดทั้งไฟล์ไปวางใน cell แล้วรัน

ผลลัพธ์ (บันทึกในโฟลเดอร์เดียวกับที่รัน):
    map_global_bleach_class.png   แผนที่โลก จุดสีตามระดับการฟอกขาว + แถบภูมิภาคตาม longitude
    map_severe_by_longitude.png   สัดส่วน severe ในแต่ละช่วง longitude (ตอบคำถามว่า longitude มีผลไหม)
    map_sea_zoom.png              ซูมเฉพาะ Southeast Asia
    map_interactive.html          แผนที่ interactive (ซูม / hover ดูรายละเอียดได้)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.express as px
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------
# 1) โหลดข้อมูล
#    ใช้ชุดเดียวกับ Workflow B: ตัด Donner, Nuryana, Setiawan ออก
#    (แหล่งที่บันทึกเฉพาะตอนฟอกขาว ไม่มีแถว 0%)
#    ถ้าอยากดูข้อมูลทั้งหมด ให้ตั้ง USE_WORKFLOW_B = False
# ---------------------------------------------------------------
URL = ("https://raw.githubusercontent.com/Athenessa/"
       "Coral-Bleaching-machine-learning/main/data/coral_clean_global.csv")
USE_WORKFLOW_B = True

df = pd.read_csv(URL)
if USE_WORKFLOW_B:
    df = df[~df["data_source"].isin(["Donner", "Nuryana", "Setiawan"])]
df = df.dropna(subset=["latitude_degrees", "longitude_degrees", "bleach_class"])
print(f"จำนวนการสำรวจที่พล็อต: {len(df):,} แถว, {df['site_id'].nunique():,} sites")
print(df["bleach_class"].value_counts())

CLASS_ORDER = ["none", "mild", "severe"]
COLORS = {"none": "#4C9BE8", "mild": "#F2B134", "severe": "#D7263D"}
LABELS = {"none": "none (0%)", "mild": "mild (0–30%)", "severe": "severe (>30%)"}

# แถบ longitude ของแต่ละภูมิภาค (ใช้อธิบายว่าทำไม longitude มีผล)
REGIONS = [
    ("Caribbean /\nFlorida", -100, -55),
    ("Red Sea /\nIndian Ocean", 30, 80),
    ("SEA /\nCoral Triangle", 95, 130),
    ("Great Barrier\nReef", 140, 155),
]

# ---------------------------------------------------------------
# 2) แผนที่โลก: จุดสีตามระดับการฟอกขาว
#    วาด none ก่อน แล้ว mild แล้ว severe ทับด้านบน เพื่อให้เห็นจุดรุนแรงชัด
# ---------------------------------------------------------------
fig = plt.figure(figsize=(16, 8))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=150))
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor="#EDEDED")
ax.add_feature(cfeature.OCEAN, facecolor="#FAFCFF")
ax.coastlines(linewidth=0.4, color="#888888")
ax.gridlines(draw_labels=True, linewidth=0.3, color="#BBBBBB", alpha=0.6)

for name, lon0, lon1 in REGIONS:
    ax.add_patch(Rectangle((lon0, -40), lon1 - lon0, 85, color="#7A5CFA",
                           alpha=0.08, transform=ccrs.PlateCarree()))
    ax.text((lon0 + lon1) / 2, 36, name, ha="center", va="bottom",
            fontsize=9, color="#4B3BA8", transform=ccrs.PlateCarree())

for c in CLASS_ORDER:
    sub = df[df["bleach_class"] == c]
    ax.scatter(sub["longitude_degrees"], sub["latitude_degrees"],
               s=6 if c == "none" else 9, c=COLORS[c], alpha=0.55,
               edgecolors="none", label=f"{LABELS[c]}  n={len(sub):,}",
               transform=ccrs.PlateCarree())

ax.set_extent([-180, 180, -40, 45], crs=ccrs.PlateCarree())
ax.legend(loc="lower left", markerscale=3, fontsize=10, frameon=True)
ax.set_title("Coral bleaching surveys by severity (1980–2020)", fontsize=14)
plt.savefig("map_global_bleach_class.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------
# 3) สัดส่วน severe ในแต่ละช่วง longitude (bin ละ 10 องศา)
#    ถ้ากราฟขึ้นลงตามภูมิภาค แปลว่า longitude แยกพื้นที่ที่เสี่ยงต่างกันได้
# ---------------------------------------------------------------
bins = np.arange(-180, 181, 10)
df["lon_bin"] = pd.cut(df["longitude_degrees"], bins=bins, right=False)
g = (df.groupby("lon_bin", observed=True)
       .agg(n=("bleach_class", "size"),
            severe=("bleach_class", lambda s: (s == "severe").mean()),
            bleached=("bleach_class", lambda s: (s != "none").mean()))
       .reset_index())
g = g[g["n"] >= 30]                      # ตัด bin ที่ข้อมูลน้อยเกินไป
g["mid"] = g["lon_bin"].apply(lambda b: b.left + 5)

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.bar(g["mid"], g["bleached"] * 100, width=9, color=COLORS["mild"],
       alpha=0.5, label="% bleached (mild + severe)")
ax.bar(g["mid"], g["severe"] * 100, width=9, color=COLORS["severe"],
       label="% severe")
for name, lon0, lon1 in REGIONS:
    ax.axvspan(lon0, lon1, color="#7A5CFA", alpha=0.07)
    ax.text((lon0 + lon1) / 2, 98, name.replace("\n", " "), ha="center",
            va="top", fontsize=8, color="#4B3BA8")
for _, r in g.iterrows():
    ax.text(r["mid"], r["bleached"] * 100 + 1, f"n={r['n']:,}", ha="center",
            va="bottom", fontsize=7, rotation=90, color="#555555")
ax.set_xlim(-180, 180)
ax.set_ylim(0, 100)
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("% of surveys")
ax.set_title("Bleaching rate by longitude band (10°, bands with ≥30 surveys)")
ax.legend(loc="upper left")
plt.savefig("map_severe_by_longitude.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------
# 4) ซูม Southeast Asia
# ---------------------------------------------------------------
sea = df[df["is_sea"] == True]
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([92, 135, -12, 22], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#EDEDED")
ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="#AAAAAA")
ax.coastlines(resolution="50m", linewidth=0.4, color="#888888")
ax.gridlines(draw_labels=True, linewidth=0.3, color="#BBBBBB", alpha=0.6)
for c in CLASS_ORDER:
    sub = sea[sea["bleach_class"] == c]
    ax.scatter(sub["longitude_degrees"], sub["latitude_degrees"],
               s=14, c=COLORS[c], alpha=0.6, edgecolors="none",
               label=f"{LABELS[c]}  n={len(sub):,}",
               transform=ccrs.PlateCarree())
ax.legend(loc="lower left", markerscale=2, fontsize=10)
ax.set_title("Southeast Asia: bleaching surveys by severity", fontsize=14)
plt.savefig("map_sea_zoom.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------
# 5) แผนที่ interactive (เปิดใน browser, ซูมและ hover ได้)
# ---------------------------------------------------------------
df_plot = df.sort_values("bleach_class",
                         key=lambda s: s.map({c: i for i, c in enumerate(CLASS_ORDER)}))
fig = px.scatter_geo(
    df_plot, lat="latitude_degrees", lon="longitude_degrees",
    color="bleach_class", color_discrete_map=COLORS,
    category_orders={"bleach_class": CLASS_ORDER},
    hover_data={"country_name": True, "date_year": True,
                "percent_bleaching": True, "tsa_dhw": True, "depth_m": True,
                "data_source": True, "latitude_degrees": False,
                "longitude_degrees": False},
    projection="natural earth", opacity=0.6,
    title="Coral bleaching surveys (hover for details)")
fig.update_traces(marker=dict(size=4))
fig.update_geos(showland=True, landcolor="#EDEDED", showcountries=True,
                countrycolor="#CCCCCC", lataxis_range=[-45, 45])
fig.write_html("map_interactive.html")
fig.show()
