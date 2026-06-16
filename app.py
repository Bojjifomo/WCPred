"""
Analisa Pertandingan Timnas — Model Dixon-Coles/Poisson (SATU FILE)
===================================================================
Dashboard Streamlit mandiri. Pilih Tim A vs Tim B -> distribusi skor & fair odds
semua pasar (1X2, Asian Handicap, Over/Under, correct score) + cek EV opsional.

Model di-fit pada 32.299 laga internasional asli (1990 - 14 Jun 2026, dataset
martj42). Parameter & rating Elo sudah di-embed di bawah, jadi TIDAK butuh file
lain selain requirements.txt.

  Backtest out-of-sample: log-loss 0.868 vs baseline 1.049 (+17%), terkalibrasi
  baik. Artinya model mendarat di tempat pasar sudah berada -> model bagus =
  bukti tak ada edge mudah, BUKAN jalan ke edge. Ini alat forecasting/edukasi,
  bukan rekomendasi taruhan. Kolom EV jujur & akan bilang NO BET saat memang.

Jalankan:  pip install -r requirements.txt  &&  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from math import exp, factorial

# ===========================================================================
# PARAMETER TER-FIT + RATING ELO  (hasil fit pada data asli, di-embed)
# ===========================================================================
PARAMS = {
    "a": 0.16021951624723013,
    "beta": 0.35033346981818647,
    "hfa": 0.3475100645530499,
    "rho": -0.06730938687501081
}

RATINGS = {
    "Spain": 2207.7,
    "Argentina": 2161.3,
    "France": 2114.1,
    "England": 2099.2,
    "Colombia": 2048.4,
    "Brazil": 2041.6,
    "Japan": 2016.1,
    "Germany": 2013.7,
    "Morocco": 2013.5,
    "Portugal": 2005.5,
    "Norway": 1999.2,
    "Netherlands": 1998.6,
    "Ecuador": 1998.5,
    "Australia": 1984.0,
    "Mexico": 1975.8,
    "Uruguay": 1964.0,
    "Croatia": 1960.8,
    "Belgium": 1946.5,
    "Turkey": 1932.0,
    "Switzerland": 1927.2,
    "South Korea": 1926.7,
    "Senegal": 1911.9,
    "Iran": 1908.3,
    "Denmark": 1904.7,
    "United States": 1903.3,
    "Paraguay": 1896.5,
    "Algeria": 1895.7,
    "Nigeria": 1895.3,
    "Italy": 1888.1,
    "Canada": 1883.5,
    "Scotland": 1883.1,
    "Ivory Coast": 1866.2,
    "Austria": 1863.8,
    "Uzbekistan": 1853.1,
    "Panama": 1845.4,
    "Russia": 1840.9,
    "Jersey": 1828.0,
    "Egypt": 1810.4,
    "Venezuela": 1804.0,
    "Sweden": 1801.9,
    "Jordan": 1801.0,
    "Ukraine": 1798.9,
    "Kosovo": 1789.0,
    "DR Congo": 1786.6,
    "Chile": 1785.4,
    "Iraq": 1771.0,
    "Republic of Ireland": 1769.8,
    "Greece": 1766.6,
    "Czech Republic": 1762.7,
    "Poland": 1761.5,
    "Guernsey": 1757.9,
    "Serbia": 1757.6,
    "Peru": 1754.4,
    "Hungary": 1750.5,
    "Basque Country": 1746.6,
    "New Zealand": 1745.2,
    "Bolivia": 1737.7,
    "Wales": 1733.9,
    "Costa Rica": 1732.5,
    "Isle of Man": 1728.1,
    "Northern Cyprus": 1724.2,
    "Cameroon": 1719.3,
    "Saudi Arabia": 1715.8,
    "Mali": 1709.3,
    "Haiti": 1705.8,
    "Slovakia": 1705.3,
    "Honduras": 1700.1,
    "Slovenia": 1696.8,
    "Padania": 1695.0,
    "Tunisia": 1689.2,
    "United Arab Emirates": 1686.8,
    "Bosnia and Herzegovina": 1676.6,
    "Georgia": 1672.7,
    "Jamaica": 1670.8,
    "Albania": 1667.1,
    "Israel": 1665.3,
    "Ynys Môn": 1661.9,
    "Burkina Faso": 1656.4,
    "Guatemala": 1656.3,
    "Romania": 1655.3,
    "Cape Verde": 1654.1,
    "Northern Ireland": 1650.8,
    "South Africa": 1648.9,
    "Abkhazia": 1645.4,
    "Ghana": 1641.7,
    "Curaçao": 1632.6,
    "North Macedonia": 1628.2,
    "Kárpátalja": 1626.9,
    "Palestine": 1613.6,
    "Suriname": 1612.3,
    "Iceland": 1611.7,
    "New Caledonia": 1605.9,
    "Syria": 1605.7,
    "Qatar": 1603.8,
    "Réunion": 1592.7,
    "Panjab": 1592.0,
    "Tamil Eelam": 1590.6,
    "Oman": 1589.0,
    "Ellan Vannin": 1573.5,
    "Artsakh": 1570.6,
    "Angola": 1570.4,
    "China": 1570.2,
    "Guinea": 1568.5,
    "Finland": 1568.2,
    "Belarus": 1566.4,
    "Chameria": 1559.3,
    "Yorkshire": 1556.9,
    "Cascadia": 1552.0,
    "Benin": 1545.9,
    "Gambia": 1544.4,
    "Thailand": 1542.9,
    "Guadeloupe": 1539.4,
    "Libya": 1539.4,
    "Trinidad and Tobago": 1539.2,
    "Indonesia": 1538.2,
    "Zambia": 1537.8,
    "Niger": 1534.1,
    "Uganda": 1533.5,
    "Gabon": 1531.8,
    "Bahrain": 1531.0,
    "Bulgaria": 1525.1,
    "Madagascar": 1524.9,
    "Western Armenia": 1522.2,
    "Nicaragua": 1519.8,
    "Shetland": 1516.3,
    "Malaysia": 1511.1,
    "Tahiti": 1506.5,
    "Kazakhstan": 1501.5,
    "United Koreans in Japan": 1501.5,
    "Kyrgyzstan": 1499.0,
    "Guyana": 1497.8,
    "Togo": 1494.2,
    "Mozambique": 1493.6,
    "Székely Land": 1493.3,
    "El Salvador": 1493.2,
    "North Korea": 1492.2,
    "Sápmi": 1487.8,
    "Martinique": 1485.6,
    "Matabeleland": 1482.1,
    "Dominican Republic": 1482.0,
    "Kuwait": 1480.9,
    "South Ossetia": 1478.6,
    "Vietnam": 1478.2,
    "Comoros": 1476.1,
    "Sierra Leone": 1473.2,
    "Gozo": 1473.1,
    "Sudan": 1467.7,
    "Montenegro": 1465.4,
    "Mayotte": 1461.4,
    "Zimbabwe": 1457.3,
    "Kenya": 1457.1,
    "Lebanon": 1455.6,
    "East Turkestan": 1454.2,
    "Liberia": 1453.1,
    "Kabylia": 1451.8,
    "Papua New Guinea": 1448.4,
    "Rwanda": 1448.2,
    "Faroe Islands": 1446.3,
    "Mauritania": 1445.0,
    "Tajikistan": 1442.9,
    "French Guiana": 1438.6,
    "Vanuatu": 1436.7,
    "Western Isles": 1434.3,
    "Malawi": 1429.2,
    "Lesotho": 1425.6,
    "Cuba": 1424.4,
    "Luxembourg": 1423.7,
    "Saint Barthélemy": 1422.9,
    "Equatorial Guinea": 1419.3,
    "Armenia": 1411.2,
    "Orkney": 1407.3,
    "Tanzania": 1400.4,
    "Fiji": 1399.6,
    "Turkmenistan": 1396.8,
    "Namibia": 1396.5,
    "Raetia": 1395.3,
    "Estonia": 1389.2,
    "Puerto Rico": 1386.4,
    "Ethiopia": 1380.2,
    "Botswana": 1379.4,
    "Azerbaijan": 1368.6,
    "Barawa": 1362.2,
    "Philippines": 1362.1,
    "Burundi": 1355.5,
    "Solomon Islands": 1348.1,
    "Cyprus": 1346.7,
    "Chagos Islands": 1342.2,
    "Malta": 1341.8,
    "Hitra": 1341.5,
    "Bermuda": 1340.8,
    "Guinea-Bissau": 1339.1,
    "Saint Vincent and the Grenadines": 1338.9,
    "Saint Helena": 1329.0,
    "Central African Republic": 1328.3,
    "Hong Kong": 1323.3,
    "Yemen": 1321.9,
    "Singapore": 1316.4,
    "Latvia": 1311.5,
    "Saint Martin": 1310.6,
    "Sint Maarten": 1309.3,
    "Lithuania": 1307.8,
    "Congo": 1304.7,
    "Grenada": 1298.9,
    "Eswatini": 1293.7,
    "Moldova": 1281.2,
    "Belize": 1280.1,
    "South Sudan": 1274.8,
    "Saint Kitts and Nevis": 1264.0,
    "Tibet": 1252.5,
    "Montserrat": 1244.0,
    "Samoa": 1241.4,
    "Aruba": 1225.6,
    "India": 1222.8,
    "Saint Lucia": 1216.5,
    "Tuvalu": 1211.7,
    "Chad": 1211.3,
    "Mauritius": 1208.8,
    "Frøya": 1169.6,
    "Afghanistan": 1164.3,
    "Dominica": 1163.5,
    "São Tomé and Príncipe": 1163.2,
    "Bangladesh": 1149.1,
    "Myanmar": 1143.7,
    "Alderney": 1140.0,
    "Pakistan": 1134.4,
    "Bonaire": 1128.0,
    "Barbados": 1120.0,
    "Antigua and Barbuda": 1117.5,
    "Andorra": 1112.3,
    "Cook Islands": 1109.5,
    "Cambodia": 1091.3,
    "Nepal": 1089.1,
    "Gibraltar": 1083.1,
    "Tonga": 1074.5,
    "Djibouti": 1062.9,
    "Somalia": 1054.6,
    "Cayman Islands": 1041.2,
    "Sri Lanka": 1033.6,
    "British Virgin Islands": 1033.5,
    "Taiwan": 1031.6,
    "United States Virgin Islands": 998.6,
    "Turks and Caicos Islands": 990.7,
    "Maldives": 985.4,
    "Mongolia": 984.6,
    "Seychelles": 970.1,
    "Guam": 962.5,
    "Northern Mariana Islands": 959.8,
    "Laos": 952.4,
    "Liechtenstein": 932.4,
    "Bahamas": 932.4,
    "American Samoa": 931.0,
    "Brunei": 923.5,
    "Anguilla": 906.3,
    "Timor-Leste": 900.2,
    "Bhutan": 886.5,
    "San Marino": 840.3,
    "Macau": 829.3
}

# ===========================================================================
# MODEL: Dixon-Coles bivariate Poisson  -> distribusi skor -> pasar
# ===========================================================================

def _pois(k, lam):
    return exp(-lam) * lam ** k / factorial(k)

def scoreline_matrix(lh, la, rho, max_goals=10):
    n = max_goals + 1
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = _pois(i, lh) * _pois(j, la)
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()

def matrix_from_params(p, elo_home, elo_away, neutral=True, max_goals=10):
    not_neutral = 0.0 if neutral else 1.0
    s = p["beta"] * (elo_home - elo_away) / 100.0 + p["hfa"] * not_neutral
    lh = exp(p["a"] + 0.5 * s); la = exp(p["a"] - 0.5 * s)
    return scoreline_matrix(lh, la, p["rho"], max_goals), lh, la

def result_probs(M):
    return {"home": float(np.tril(M, -1).sum()),
            "draw": float(np.trace(M)),
            "away": float(np.triu(M, 1).sum())}

def _mass(M, value_fn):
    A = B = 0.0
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            p = M[i, j]
            if p == 0:
                continue
            adj = value_fn(i, j)
            if adj >= 0.5:      A += p
            elif adj == 0.25:   A += 0.5 * p
            elif adj == 0.0:    pass
            elif adj == -0.25:  B += 0.5 * p
            else:               B += p
    return A, B

def ah_fair(M, line, side="home"):
    def vf(i, j):
        margin = (i - j) if side == "home" else (j - i)
        return margin + line if side == "home" else margin - line
    A, B = _mass(M, vf)
    return {"fair_odds": (1 + B / A) if A > 0 else float("inf"),
            "win_prob": A / (A + B) if (A + B) > 0 else 0.0}

def ou_fair(M, line, side="over"):
    def vf(i, j):
        total = i + j
        return (total - line) if side == "over" else (line - total)
    A, B = _mass(M, vf)
    return {"fair_odds": (1 + B / A) if A > 0 else float("inf"),
            "win_prob": A / (A + B) if (A + B) > 0 else 0.0}

def top_scores(M, n=8):
    flat = [((i, j), M[i, j]) for i in range(M.shape[0])
            for j in range(M.shape[1])]
    flat.sort(key=lambda x: x[1], reverse=True)
    return [(f"{i}-{j}", float(p)) for (i, j), p in flat[:n]]

def strip_vig(*odds):
    qs = [1 / o for o in odds]; s = sum(qs)
    return [q / s for q in qs], s - 1.0

def ev(prob, odds):
    return prob * odds - 1.0


# ===========================================================================
# UI STREAMLIT
# ===========================================================================
st.set_page_config(page_title="Analisa Timnas — Dixon-Coles",
                   layout="wide", page_icon="⚽")
TEAMS = list(RATINGS.keys())

st.title("⚽ Analisa Pertandingan Timnas")
st.caption("Model Dixon-Coles/Poisson ter-fit pada 32.299 laga internasional "
           "asli (1990 – 14 Jun 2026). Elo dihitung dari hasil nyata.")
st.info(
    "**Baca dulu.** Tools ini menghitung *fair odds* (peluang model, tanpa "
    "margin bandar) untuk **memahami** laga — bukan rekomendasi taruhan. "
    "Model terkalibrasi baik (backtest +17% vs baseline), yang justru berarti "
    "ia mendarat di tempat pasar sudah berada: model bagus = bukti tak ada edge "
    "mudah, bukan jalan ke edge. Kolom EV dihitung jujur & akan bilang NO BET "
    "saat memang begitu.", icon="ℹ️")

c1, c2, c3 = st.columns([3, 3, 2])
with c1:
    home = st.selectbox("Tim A (home / sisi atas)", TEAMS,
                        index=TEAMS.index("France") if "France" in TEAMS else 0)
with c2:
    away_opts = [t for t in TEAMS if t != home]
    away = st.selectbox("Tim B (away)", away_opts,
                        index=away_opts.index("Senegal")
                        if "Senegal" in away_opts else 0)
with c3:
    neutral = st.toggle("Venue netral", value=True,
                        help="Aktif untuk Piala Dunia / venue netral.")

eh, ea = RATINGS[home], RATINGS[away]
M, lh, la = matrix_from_params(PARAMS, eh, ea, neutral=neutral)
r = result_probs(M)

st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Elo {home}", f"{eh:.0f}")
k2.metric(f"Elo {away}", f"{ea:.0f}", f"{ea-eh:+.0f} vs A")
k3.metric(f"xG {home} (λ)", f"{lh:.2f}")
k4.metric(f"xG {away} (λ)", f"{la:.2f}")

cL, cR = st.columns(2)
with cL:
    st.subheader("Hasil 1X2 (model)")
    df1 = pd.DataFrame({"Outcome": [f"{home} menang", "Seri", f"{away} menang"],
                        "Peluang": [r["home"], r["draw"], r["away"]],
                        "Fair odds": [1/r["home"], 1/r["draw"], 1/r["away"]]})
    st.dataframe(df1.style.format({"Peluang": "{:.1%}", "Fair odds": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
    st.subheader("Correct score teratas")
    ts = top_scores(M, 8)
    dfs = pd.DataFrame({"Skor (A-B)": [s for s, _ in ts],
                        "Peluang": [p for _, p in ts],
                        "Fair odds": [1/p for _, p in ts]})
    st.dataframe(dfs.style.format({"Peluang": "{:.1%}", "Fair odds": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
with cR:
    st.subheader("Distribusi skor (heatmap)")
    mg = 6; sub = M[:mg+1, :mg+1]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(sub, cmap="magma_r", origin="upper")
    ax.set_xlabel(f"{away} (gol)"); ax.set_ylabel(f"{home} (gol)")
    ax.set_xticks(range(mg+1)); ax.set_yticks(range(mg+1))
    for i in range(mg+1):
        for j in range(mg+1):
            if sub[i, j] >= 0.02:
                ax.text(j, i, f"{sub[i,j]*100:.0f}", ha="center", va="center",
                        color="white" if sub[i, j] > sub.max()*0.5 else "black",
                        fontsize=8)
    ax.set_title("Peluang skor (%)"); fig.tight_layout()
    st.pyplot(fig)

st.divider()
cah, cou = st.columns(2)
with cah:
    st.subheader("Asian Handicap — fair odds")
    rows = []
    for L in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
        rows.append({"Line (pada A)": f"{L:+.2f}",
                     home: ah_fair(M, L, "home")["fair_odds"],
                     away: ah_fair(M, -L, "away")["fair_odds"]})
    st.dataframe(pd.DataFrame(rows).style.format({home: "{:.2f}", away: "{:.2f}"}),
                 hide_index=True, use_container_width=True)
with cou:
    st.subheader("Over / Under — fair odds")
    rows = []
    for L in [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]:
        rows.append({"Line": L, "Over": ou_fair(M, L, "over")["fair_odds"],
                     "Under": ou_fair(M, L, "under")["fair_odds"]})
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Line": "{:.2f}", "Over": "{:.2f}", "Under": "{:.2f}"}),
        hide_index=True, use_container_width=True)

st.divider()
st.subheader("Cek EV vs harga pasar (opsional)")
st.caption("Masukkan odds desimal dari papan. Vig dibuang otomatis, EV dihitung "
           "per sisi. Verdict jujur — bukan ajakan taruhan.")
t1, t2, t3 = st.tabs(["1X2", "Asian Handicap", "Over/Under"])
with t1:
    a, b, c = st.columns(3)
    o1 = a.number_input(f"{home} menang", 1.01, 99.0, round(1/r["home"], 2), 0.01)
    ox = b.number_input("Seri", 1.01, 99.0, round(1/r["draw"], 2), 0.01)
    o2 = c.number_input(f"{away} menang", 1.01, 99.0, round(1/r["away"], 2), 0.01)
    if st.button("Hitung EV 1X2"):
        (f1, fx, f2), vig = strip_vig(o1, ox, o2)
        evs = [ev(r["home"], o1), ev(r["draw"], ox), ev(r["away"], o2)]
        out = pd.DataFrame({"Sisi": [home, "Seri", away],
                            "Model %": [r["home"], r["draw"], r["away"]],
                            "Pasar fair %": [f1, fx, f2], "EV": evs})
        st.dataframe(out.style.format({"Model %": "{:.1%}",
                     "Pasar fair %": "{:.1%}", "EV": "{:+.1%}"}),
                     hide_index=True, use_container_width=True)
        st.write(f"Overround (vig) pasar: **{vig:.1%}**")
        st.success("✅ NO BET — tidak ada sisi EV+. Model setuju dengan pasar.") \
            if max(evs) <= 0 else \
            st.warning(f"Sisa EV+ {max(evs):+.1%}. Curigai line stale/in-play, "
                       "model meleset, atau selisih < presisi model dulu.")
with t2:
    L = st.selectbox("Line (handicap pada Tim A)",
                     [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], index=2)
    d1, d2 = st.columns(2)
    oh = d1.number_input(f"Odds {home} (des)", 1.01, 99.0,
                         round(ah_fair(M, L, "home")["fair_odds"], 2), 0.01)
    oa = d2.number_input(f"Odds {away} (des)", 1.01, 99.0,
                         round(ah_fair(M, -L, "away")["fair_odds"], 2), 0.01)
    if st.button("Hitung EV Handicap"):
        ph = ah_fair(M, L, "home")["win_prob"]; pa = ah_fair(M, -L, "away")["win_prob"]
        (_, _), vig = strip_vig(oh, oa)
        evs = [ev(ph, oh), ev(pa, oa)]
        out = pd.DataFrame({"Sisi": [f"{home} {L:+.2f}", f"{away} {-L:+.2f}"],
                            "Model win%": [ph, pa], "EV": evs})
        st.dataframe(out.style.format({"Model win%": "{:.1%}", "EV": "{:+.1%}"}),
                     hide_index=True, use_container_width=True)
        st.write(f"Vig pasar: **{vig:.1%}**")
        st.success("✅ NO BET — tidak ada sisi EV+.") if max(evs) <= 0 else \
            st.warning(f"Sisa EV+ {max(evs):+.1%}. Cek higiene data dulu.")
with t3:
    L = st.selectbox("Line OU", [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5], index=4)
    d1, d2 = st.columns(2)
    oo = d1.number_input("Odds Over (des)", 1.01, 99.0,
                         round(ou_fair(M, L, "over")["fair_odds"], 2), 0.01)
    ou_ = d2.number_input("Odds Under (des)", 1.01, 99.0,
                          round(ou_fair(M, L, "under")["fair_odds"], 2), 0.01)
    if st.button("Hitung EV Over/Under"):
        po = ou_fair(M, L, "over")["win_prob"]; pu = ou_fair(M, L, "under")["win_prob"]
        (_, _), vig = strip_vig(oo, ou_)
        evs = [ev(po, oo), ev(pu, ou_)]
        out = pd.DataFrame({"Sisi": [f"Over {L}", f"Under {L}"],
                            "Model %": [po, pu], "EV": evs})
        st.dataframe(out.style.format({"Model %": "{:.1%}", "EV": "{:+.1%}"}),
                     hide_index=True, use_container_width=True)
        st.write(f"Vig pasar: **{vig:.1%}**")
        st.success("✅ NO BET — tidak ada sisi EV+.") if max(evs) <= 0 else \
            st.warning(f"Sisa EV+ {max(evs):+.1%}. Cek higiene data dulu.")

st.divider()
st.caption("Parameter ter-fit: μ_total≈2.35 gol · home adv 0.35 gol · "
           "beta 0.35/100 Elo · rho −0.067 · backtest OOS log-loss 0.868 "
           "(+17% vs baseline). Edge sejati hanya teruji vs closing-odds "
           "historis. Alat forecasting/edukasi, bukan layanan taruhan.")
