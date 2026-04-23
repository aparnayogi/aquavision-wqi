# backend/recommender.py
# ── Upgraded Rule-Based Recommendation Engine ────────────────────────────────
# Based on WHO & BIS (Bureau of Indian Standards) water quality guidelines

def get_recommendations(data: dict) -> dict:
    """
    Analyse sensor readings and return structured recommendations.

    Returns:
        {
          "summary":  "Safe" | "Caution" | "Danger",
          "score":    0-100  (overall safety score),
          "items": [
            {
              "parameter": "Turbidity",
              "value":     12.4,
              "unit":      "NTU",
              "status":    "danger",        # ok | warn | danger
              "message":   "High turbidity detected...",
              "action":    "Use coagulation and sand filtration.",
              "icon":      "filter"
            }, ...
          ]
        }
    """
    ph        = float(data.get("ph",               7.0))
    do        = float(data.get("dissolved_oxygen",  7.0))
    turbidity = float(data.get("turbidity",         1.0))
    bod       = float(data.get("bod",               2.0))
    cond      = float(data.get("conductivity",    300.0))
    nitrates  = float(data.get("nitrates",          5.0))
    coliform  = float(data.get("total_coliform",    0.0))

    items = []

    # ── pH ───────────────────────────────────────────────────────────────────
    if ph < 5.5:
        items.append({
            "parameter": "pH",
            "value": ph, "unit": "",
            "status": "danger",
            "message": f"pH {ph} is critically acidic — corrosive to pipes and harmful to aquatic life.",
            "action": "Add lime (Ca(OH)₂) or soda ash to raise pH above 6.5.",
            "icon": "ph",
        })
    elif ph < 6.5:
        items.append({
            "parameter": "pH",
            "value": ph, "unit": "",
            "status": "warn",
            "message": f"pH {ph} is mildly acidic — below WHO safe range (6.5–8.5).",
            "action": "Add alkaline buffer. Monitor closely.",
            "icon": "ph",
        })
    elif ph > 9.0:
        items.append({
            "parameter": "pH",
            "value": ph, "unit": "",
            "status": "danger",
            "message": f"pH {ph} is strongly alkaline — may cause skin/eye irritation.",
            "action": "Inject CO₂ or dilute acid to lower pH below 8.5.",
            "icon": "ph",
        })
    elif ph > 8.5:
        items.append({
            "parameter": "pH",
            "value": ph, "unit": "",
            "status": "warn",
            "message": f"pH {ph} is mildly alkaline — slightly above safe range.",
            "action": "Consider mild acid treatment or dilution.",
            "icon": "ph",
        })
    else:
        items.append({
            "parameter": "pH",
            "value": ph, "unit": "",
            "status": "ok",
            "message": f"pH {ph} is within the safe range (6.5–8.5).",
            "action": "No action needed.",
            "icon": "ph",
        })

    # ── Dissolved Oxygen ─────────────────────────────────────────────────────
    if do < 2.0:
        items.append({
            "parameter": "Dissolved Oxygen",
            "value": do, "unit": "mg/L",
            "status": "danger",
            "message": f"DO {do} mg/L is critically low — hypoxic conditions, fish kills likely.",
            "action": "Install emergency aeration / diffusers immediately.",
            "icon": "aeration",
        })
    elif do < 5.0:
        items.append({
            "parameter": "Dissolved Oxygen",
            "value": do, "unit": "mg/L",
            "status": "warn",
            "message": f"DO {do} mg/L is low — stress conditions for aquatic organisms.",
            "action": "Increase surface aeration or mechanical mixing.",
            "icon": "aeration",
        })
    else:
        items.append({
            "parameter": "Dissolved Oxygen",
            "value": do, "unit": "mg/L",
            "status": "ok",
            "message": f"DO {do} mg/L is healthy (≥5 mg/L).",
            "action": "No action needed.",
            "icon": "aeration",
        })

    # ── Turbidity ─────────────────────────────────────────────────────────────
    if turbidity > 15:
        items.append({
            "parameter": "Turbidity",
            "value": turbidity, "unit": "NTU",
            "status": "danger",
            "message": f"Turbidity {turbidity} NTU is very high — water is visibly cloudy and unsafe.",
            "action": "Apply coagulation (alum), flocculation, and multi-stage sand filtration.",
            "icon": "filter",
        })
    elif turbidity > 4:
        items.append({
            "parameter": "Turbidity",
            "value": turbidity, "unit": "NTU",
            "status": "warn",
            "message": f"Turbidity {turbidity} NTU exceeds WHO drinking limit (4 NTU).",
            "action": "Use pre-filtration or cartridge filter to reduce suspended solids.",
            "icon": "filter",
        })
    else:
        items.append({
            "parameter": "Turbidity",
            "value": turbidity, "unit": "NTU",
            "status": "ok",
            "message": f"Turbidity {turbidity} NTU is within acceptable limits (≤4 NTU).",
            "action": "No action needed.",
            "icon": "filter",
        })

    # ── BOD ───────────────────────────────────────────────────────────────────
    if bod > 6:
        items.append({
            "parameter": "BOD",
            "value": bod, "unit": "mg/L",
            "status": "danger",
            "message": f"BOD {bod} mg/L indicates heavy organic pollution.",
            "action": "Apply biological treatment (activated sludge or trickling filter).",
            "icon": "organic",
        })
    elif bod > 3:
        items.append({
            "parameter": "BOD",
            "value": bod, "unit": "mg/L",
            "status": "warn",
            "message": f"BOD {bod} mg/L shows moderate organic load.",
            "action": "Monitor organic waste discharge. Consider bio-remediation.",
            "icon": "organic",
        })
    else:
        items.append({
            "parameter": "BOD",
            "value": bod, "unit": "mg/L",
            "status": "ok",
            "message": f"BOD {bod} mg/L is within safe limits (≤3 mg/L).",
            "action": "No action needed.",
            "icon": "organic",
        })

    # ── Conductivity ──────────────────────────────────────────────────────────
    if cond > 1000:
        items.append({
            "parameter": "Conductivity",
            "value": cond, "unit": "µS/cm",
            "status": "danger",
            "message": f"Conductivity {cond} µS/cm — excessively high dissolved salts.",
            "action": "Use Reverse Osmosis (RO) or electrodialysis to reduce TDS.",
            "icon": "salt",
        })
    elif cond > 600:
        items.append({
            "parameter": "Conductivity",
            "value": cond, "unit": "µS/cm",
            "status": "warn",
            "message": f"Conductivity {cond} µS/cm is elevated — check for salt intrusion.",
            "action": "Investigate source. Consider partial RO blending.",
            "icon": "salt",
        })
    else:
        items.append({
            "parameter": "Conductivity",
            "value": cond, "unit": "µS/cm",
            "status": "ok",
            "message": f"Conductivity {cond} µS/cm is within normal range.",
            "action": "No action needed.",
            "icon": "salt",
        })

    # ── Nitrates ──────────────────────────────────────────────────────────────
    if nitrates > 10:
        items.append({
            "parameter": "Nitrates",
            "value": nitrates, "unit": "mg/L",
            "status": "danger",
            "message": f"Nitrates {nitrates} mg/L exceed WHO limit (10 mg/L) — risk of methemoglobinemia.",
            "action": "Use ion exchange, biological denitrification, or RO.",
            "icon": "chemical",
        })
    elif nitrates > 5:
        items.append({
            "parameter": "Nitrates",
            "value": nitrates, "unit": "mg/L",
            "status": "warn",
            "message": f"Nitrates {nitrates} mg/L are elevated — possible agricultural runoff.",
            "action": "Investigate upstream sources. Increase monitoring frequency.",
            "icon": "chemical",
        })
    else:
        items.append({
            "parameter": "Nitrates",
            "value": nitrates, "unit": "mg/L",
            "status": "ok",
            "message": f"Nitrates {nitrates} mg/L are within safe limits (≤5 mg/L).",
            "action": "No action needed.",
            "icon": "chemical",
        })

    # ── Total Coliform ────────────────────────────────────────────────────────
    if coliform > 2:
        items.append({
            "parameter": "Total Coliform",
            "value": coliform, "unit": "CFU/100mL",
            "status": "danger",
            "message": f"Coliform {coliform} CFU/100mL — serious microbial contamination detected!",
            "action": "Shock chlorination + UV disinfection required immediately.",
            "icon": "bacteria",
        })
    elif coliform > 0:
        items.append({
            "parameter": "Total Coliform",
            "value": coliform, "unit": "CFU/100mL",
            "status": "warn",
            "message": f"Coliform {coliform} CFU/100mL detected — WHO limit is 0 for drinking water.",
            "action": "Apply chlorination or UV treatment. Retest after 24 hours.",
            "icon": "bacteria",
        })
    else:
        items.append({
            "parameter": "Total Coliform",
            "value": coliform, "unit": "CFU/100mL",
            "status": "ok",
            "message": "No coliform detected — water is microbiologically safe.",
            "action": "Maintain current disinfection regime.",
            "icon": "bacteria",
        })

    # ── Overall summary ───────────────────────────────────────────────────────
    danger_count = sum(1 for i in items if i["status"] == "danger")
    warn_count   = sum(1 for i in items if i["status"] == "warn")
    ok_count     = sum(1 for i in items if i["status"] == "ok")

    if danger_count >= 2:
        summary = "Danger"
        score   = max(0, 30 - danger_count * 8)
    elif danger_count == 1:
        summary = "Danger"
        score   = max(20, 45 - warn_count * 5)
    elif warn_count >= 2:
        summary = "Caution"
        score   = max(40, 65 - warn_count * 5)
    elif warn_count == 1:
        summary = "Caution"
        score   = 72
    else:
        summary = "Safe"
        score   = min(100, 85 + ok_count * 2)

    return {
        "summary": summary,
        "score":   score,
        "counts":  {"danger": danger_count, "warn": warn_count, "ok": ok_count},
        "items":   items,
    }