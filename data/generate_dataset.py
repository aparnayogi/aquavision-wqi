import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000  # number of samples

def compute_wqi(ph, do, turbidity, conductivity, bod, nitrates, coliform):
    """
    Weighted WQI formula (0-100 scale).
    Higher = better water quality.
    """
    ph_score        = np.clip(100 - np.abs(ph - 7.5) * 25,       0, 100)
    do_score        = np.clip(do * 11,                             0, 100)
    turb_score      = np.clip(100 - turbidity * 4.5,              0, 100)
    cond_score      = np.clip(100 - (conductivity - 50) * 0.08,   0, 100)
    bod_score       = np.clip(100 - bod * 12,                      0, 100)
    nitrate_score   = np.clip(100 - nitrates * 6,                  0, 100)
    coliform_score  = np.clip(100 - coliform * 20,                 0, 100)

    wqi = (
        ph_score       * 0.20 +
        do_score       * 0.22 +
        turb_score     * 0.18 +
        cond_score     * 0.10 +
        bod_score      * 0.15 +
        nitrate_score  * 0.08 +
        coliform_score * 0.07
    )
    return np.clip(wqi, 0, 100)

data = {
    "ph":               np.random.uniform(5.0,  9.5,  N),
    "dissolved_oxygen": np.random.uniform(1.0,  14.0, N),
    "turbidity":        np.random.uniform(0.1,  25.0, N),
    "conductivity":     np.random.uniform(50,   1500, N),
    "bod":              np.random.uniform(0.5,  12.0, N),
    "nitrates":         np.random.uniform(0.1,  20.0, N),
    "total_coliform":   np.random.uniform(0,    5.0,  N),
    # random locations around India for map visualization
    "latitude":         np.random.uniform(8.0,  35.0, N),
    "longitude":        np.random.uniform(68.0, 97.0, N),
}

df = pd.DataFrame(data)
df["wqi"] = compute_wqi(
    df["ph"], df["dissolved_oxygen"], df["turbidity"],
    df["conductivity"], df["bod"], df["nitrates"], df["total_coliform"]
)

# Add small noise to make it realistic
df["wqi"] += np.random.normal(0, 1.5, N)
df["wqi"] = df["wqi"].clip(0, 100).round(2)
df = df.round(3)

df.to_csv("water_quality.csv", index=False)
print(f"Dataset saved: {N} rows, WQI range {df['wqi'].min():.1f} – {df['wqi'].max():.1f}")
print(df.describe().round(2))