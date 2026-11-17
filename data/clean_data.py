"""
Coral bleaching data cleaning
2107541 ML for Environmental Applications - Project 2

Input : data/raw/global_bleaching_environmental.csv
        (download from https://www.bco-dmo.org/dataset/773466)
Output: data/processed/coral_clean_global.csv
        data/processed/coral_clean_sea.csv
        data/processed/data_dictionary.csv
        figures/*.png

Run   : python src/clean_data.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_PATH = "data/raw/global_bleaching_environmental.csv"
OUT_DIR = "data/processed"
FIG_DIR = "figures"
SEVERE = 30                 # % bleaching threshold for "severe" class
DROP_IF_MISSING_OVER = 50   # drop features with more missing (%) than this
SEA = ["Thailand", "Malaysia", "Indonesia", "Philippines", "Vietnam", "Viet Nam",
       "Cambodia", "Myanmar", "Singapore", "Brunei", "Timor-Leste", "East Timor"]

# feature -> description (units)
FEATURES = {
    "tsa_dhw": "Degree heating weeks, TSA-based, previous 12 weeks (weeks)",
    "ssta_dhw": "Degree heating weeks, SSTA-based (weeks)",
    "tsa": "Weekly SST minus max weekly climatological SST (°C)",
    "ssta": "Weekly SST minus weekly climatological SST (°C)",
    "ssta_frequency": "Weeks in previous 52 with SSTA >= 1 °C (count)",
    "tsa_frequency": "Weeks in previous 52 with TSA >= 1 °C (count)",
    "climsst": "Climatological SST of the site (°C)",
    "temperature_c": "SST at time of survey (°C)",
    "temperature_maximum": "Maximum SST (°C)",
    "turbidity": "Kd490 in 100-km buffer, higher = more turbid (1/m)",
    "depth_m": "Survey depth (m)",
    "distance_to_shore": "Distance to nearest land (m)",
    "exposure": "Wave exposure: Exposed / Sheltered / Sometimes (category)",
    "cyclone_frequency": "Number of cyclones 1964-2014 (count)",
    "windspeed": "Wind speed (units as provided by source)",
    "latitude_degrees": "Latitude (°N)",
    "longitude_degrees": "Longitude (°E)",
    "ocean_name": "Ocean (category)",
    "realm_name": "MEOW realm (category)",
    "date_year": "Survey year",
    "date_month": "Survey month",
}
META = {
    "sample_id": "Survey event ID - not a feature",
    "site_id": "Site ID - use for group split, not a feature",
    "country_name": "Country - for subsetting",
    "data_source": "Source dataset - measurement method differs",
    "date": "Survey date",
    "is_sea": "True if Southeast Asian country",
    "is_thailand": "True if Thailand",
}
TARGETS = {
    "percent_bleaching": "TARGET regression: % corals bleached",
    "bleach_class": f"TARGET classification: none (0) / mild (0-{SEVERE}] / severe (>{SEVERE})",
    "bleached": "TARGET binary: 1 if any bleaching",
}
TEXT_COLS = ["data_source", "ocean_name", "realm_name", "ecoregion_name", "country_name",
             "state_island_province_name", "city_town_name", "site_name", "exposure",
             "substrate_name", "bleaching_level", "reef_id", "date",
             "site_comments", "sample_comments", "bleaching_comments"]
# valid physical ranges; values outside -> NaN
RANGES = {
    "percent_bleaching": (0, 100), "percent_cover": (0, 100),
    "latitude_degrees": (-90, 90), "longitude_degrees": (-180, 180),
    "depth_m": (0, 100), "distance_to_shore": (0, None), "turbidity": (0, None),
    "cyclone_frequency": (0, None), "tsa_dhw": (0, None), "ssta_dhw": (0, None),
    "temperature_c": (15, 40), "climsst": (15, 40), "windspeed": (0, None),
}


def load(path):
    # source uses "nd" for no data
    raw = pd.read_csv(path, na_values=["nd", "ND", "NA", "N/A", "", " "], low_memory=False)
    print(f"[load] raw shape: {raw.shape}")
    return raw


def fix_types(df):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
    for c in df.columns:
        if c in TEXT_COLS:
            df[c] = df[c].astype("string").str.strip()
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.to_datetime(dict(year=df["date_year"], month=df["date_month"],
                                         day=df["date_day"]), errors="coerce")
    # temperature: convert Kelvin -> °C if values look like Kelvin
    for c in ["climsst", "temperature_kelvin", "temperature_mean",
              "temperature_minimum", "temperature_maximum"]:
        if c in df and df[c].median() > 200:
            df[c] = df[c] - 273.15
            print(f"[types] {c}: Kelvin -> °C")
    return df.rename(columns={"temperature_kelvin": "temperature_c"})


def fix_invalid(df):
    # impossible values -> NaN (keep the row)
    for c, (lo, hi) in RANGES.items():
        if c not in df:
            continue
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[c] < lo
        if hi is not None:
            bad |= df[c] > hi
        if bad.any():
            print(f"[invalid] {c}: {int(bad.sum())} values outside [{lo}, {hi}] -> NaN")
        df.loc[bad, c] = np.nan
    return df


def drop_duplicates(df):
    n = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="sample_id", keep="first")
    print(f"[dups] removed {n - len(df)} rows")
    return df


def make_targets(df):
    n = len(df)
    df = df.dropna(subset=["percent_bleaching"])   # never impute the target
    print(f"[target] dropped {n - len(df)} rows without target")
    df["bleach_class"] = pd.cut(df["percent_bleaching"], bins=[-0.01, 0, SEVERE, 100],
                                labels=["none", "mild", "severe"])
    df["bleached"] = (df["percent_bleaching"] > 0).astype(int)
    print("[target] by data source (low n_unique = coded values, not measured %):")
    print(df.groupby("data_source")["percent_bleaching"]
            .agg(n="count", mean="mean", pct_zero=lambda s: (s == 0).mean() * 100,
                 n_unique="nunique").round(1).to_string())
    return df


def add_region(df):
    df["is_sea"] = df["country_name"].isin(SEA)
    df["is_thailand"] = df["country_name"].eq("Thailand")
    print(f"[region] global {len(df)} | SEA {df['is_sea'].sum()} | Thailand {df['is_thailand'].sum()}")
    return df


def select_features(df):
    feats = [c for c in FEATURES if c in df]
    miss = df[feats].isna().mean() * 100
    sparse = miss[miss > DROP_IF_MISSING_OVER].index.tolist()
    if sparse:
        print(f"[features] dropped (>{DROP_IF_MISSING_OVER}% missing): {sparse}")
    return [f for f in feats if f not in sparse]


def save_figures(df, feats):
    os.makedirs(FIG_DIR, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.hist(df["percent_bleaching"], bins=50, color="coral")
    plt.xlabel("Bleaching (%)"); plt.ylabel("Number of surveys")
    plt.title("Distribution of coral bleaching, 1980-2020")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/target_distribution.png", dpi=300, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(14, 6))
    sc = plt.scatter(df["longitude_degrees"], df["latitude_degrees"], c=df["percent_bleaching"],
                     cmap="YlOrRd", s=6, alpha=0.6)
    plt.colorbar(sc, label="Bleaching (%)")
    plt.xlabel("Longitude (°)"); plt.ylabel("Latitude (°)")
    plt.title("Coral bleaching survey sites, 1980-2020")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/site_map.png", dpi=300, bbox_inches="tight"); plt.close()

    num = df[feats + ["percent_bleaching"]].select_dtypes("number")
    corr = num.corr(method="spearman")
    plt.figure(figsize=(11, 9))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman ρ")
    plt.xticks(range(len(corr)), corr.columns, rotation=90); plt.yticks(range(len(corr)), corr.columns)
    plt.title("Spearman correlation - features and bleaching")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_correlation.png", dpi=300, bbox_inches="tight"); plt.close()
    print(f"[figures] saved to {FIG_DIR}/")


def export(df, feats):
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = [c for c in list(META) + feats + list(TARGETS) if c in df]
    clean = df[cols].sort_values("date").reset_index(drop=True)
    clean.to_csv(f"{OUT_DIR}/coral_clean_global.csv", index=False)
    clean[clean["is_sea"]].to_csv(f"{OUT_DIR}/coral_clean_sea.csv", index=False)

    desc = {**META, **FEATURES, **TARGETS}
    pd.DataFrame({
        "attribute": cols,
        "role": ["meta" if c in META else "target" if c in TARGETS else "feature" for c in cols],
        "description": [desc[c] for c in cols],
        "dtype": [str(clean[c].dtype) for c in cols],
        "missing_%": [round(clean[c].isna().mean() * 100, 1) for c in cols],
    }).to_csv(f"{OUT_DIR}/data_dictionary.csv", index=False)
    print(f"[export] {clean.shape} -> {OUT_DIR}/")
    return clean


if __name__ == "__main__":
    raw = load(RAW_PATH)
    df = fix_types(raw.copy())          # raw stays untouched
    df = fix_invalid(df)
    df = drop_duplicates(df)
    df = make_targets(df)
    df = add_region(df)
    feats = select_features(df)
    save_figures(df, feats)
    export(df, feats)
