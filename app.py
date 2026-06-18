"""
Analisa Timnas — Mode WIZARD (tanya-jawab bertahap)
====================================================
Flow: pilih match -> pilih pasar -> input taruhan + odds -> verdict.
Semua tabel/heatmap/data internal DISEMBUNYIKAN. Output akhir = verdict GAS/SKIP.
Model Dixon-Coles sama persis (terkalibrasi +17% vs baseline). Tidak memaksa
memilih 'pemenang' dari opsi EV-negatif: kalau semua jelek, dia bilang SKIP semua.
"""
import numpy as np
import streamlit as st
from math import exp, factorial

PARAMS = {"a": 0.16021951624723013, "beta": 0.35033346981818647,
          "hfa": 0.3475100645530499, "rho": -0.06730938687501081}
LAMBDA_CAP = 4.0

# (RATINGS dipersingkat di komentar — pakai dict lengkap dari app utama)
RATINGS = {
    "Spain": 2207.7, "Argentina": 2161.3, "France": 2114.1, "England": 2099.2,
    "Colombia": 2048.4, "Brazil": 2041.6, "Japan": 2016.1, "Germany": 2013.7,
    "Morocco": 2013.5, "Portugal": 2005.5, "Norway": 1999.2, "Netherlands": 1998.6,
    "Ecuador": 1998.5, "Australia": 1984.0, "Mexico": 1975.8, "Uruguay": 1964.0,
    "Croatia": 1960.8, "Belgium": 1946.5, "Turkey": 1932.0, "Switzerland": 1927.2,
    "South Korea": 1926.7, "Senegal": 1911.9, "Iran": 1908.3, "Denmark": 1904.7,
    "United States": 1903.3, "Paraguay": 1896.5, "Algeria": 1895.7, "Nigeria": 1895.3,
    "Italy": 1888.1, "Canada": 1883.5, "Scotland": 1883.1, "Ivory Coast": 1866.2,
    "Austria": 1863.8, "Uzbekistan": 1853.1, "Panama": 1845.4, "Egypt": 1810.4,
    "Venezuela": 1804.0, "Sweden": 1801.9, "Jordan": 1801.0, "Ukraine": 1798.9,
    "DR Congo": 1786.6, "Chile": 1785.4, "Iraq": 1771.0, "Greece": 1766.6,
    "Czech Republic": 1762.7, "Poland": 1761.5, "Serbia": 1757.6, "Peru": 1754.4,
    "Hungary": 1750.5, "Wales": 1733.9, "Costa Rica": 1732.5, "Cameroon": 1719.3,
    "Saudi Arabia": 1715.8, "Slovakia": 1705.3, "Slovenia": 1696.8, "Tunisia": 1689.2,
    "Jamaica": 1670.8, "Albania": 1667.1, "Romania": 1655.3, "Cape Verde": 1654.1,
    "South Africa": 1648.9, "Ghana": 1641.7, "Iceland": 1611.7, "Qatar": 1603.8,
    "China": 1570.2, "Finland": 1568.2, "Indonesia": 1538.2, "Bulgaria": 1525.1,
    "Malaysia": 1511.1, "Kazakhstan": 1501.5, "El Salvador": 1493.2,
    "North Korea": 1492.2, "Vietnam": 1478.2, "Latvia": 1311.5, "Thailand": 1542.9,
    "India": 1222.8, "San Marino": 840.3,
}

def _pois(k, lam): return exp(-lam) * lam ** k / factorial(k)
def _soft_cap(lam, cap=LAMBDA_CAP):
    return lam if lam <= cap else cap + np.log1p(lam - cap)

def scoreline_matrix(lh, la, rho, max_goals=10):
    n = max_goals + 1; M = np.zeros((n, n))
    for i in range(n):
        for j in range(n): M[i, j] = _pois(i, lh) * _pois(j, la)
    M[0, 0] *= 1 - lh*la*rho; M[0, 1] *= 1 + lh*rho
    M[1, 0] *= 1 + la*rho; M[1, 1] *= 1 - rho
    return M / M.sum()

def matrix_from_params(p, eh, ea, adj_h=0.0, adj_a=0.0, neutral=True, mg=10):
    nn = 0.0 if neutral else 1.0
    s = p["beta"]*((eh+adj_h)-(ea+adj_a))/100.0 + p["hfa"]*nn
    lh = _soft_cap(exp(p["a"]+0.5*s)); la = _soft_cap(exp(p["a"]-0.5*s))
    return scoreline_matrix(lh, la, p["rho"], mg)

def _mass(M, vf):
    A = B = 0.0; n = M.shape[0]
    for i in range(n):
        for j in range(n):
            p = M[i, j]
            if p == 0: continue
            adj = vf(i, j)
            if adj >= 0.5: A += p
            elif adj == 0.25: A += 0.5*p
            elif adj == 0.0: pass
            elif adj == -0.25: B += 0.5*p
            else: B += p
    return A, B

def ah_fair(M, line, side="home"):
    def vf(i, j):
        m = (i-j) if side == "home" else (j-i)
        return m+line if side == "home" else m-line
    A, B = _mass(M, vf)
    return {"fair_odds": (1+B/A) if A > 0 else float("inf"),
            "win_prob": A/(A+B) if (A+B) > 0 else 0.0}

def ou_fair(M, line, side="over"):
    def vf(i, j):
        t = i+j
        return (t-line) if side == "over" else (line-t)
    A, B = _mass(M, vf)
    return {"fair_odds": (1+B/A) if A > 0 else float("inf"),
            "win_prob": A/(A+B) if (A+B) > 0 else 0.0}

def ev(prob, odds): return prob*odds - 1.0

# ============ WIZARD UI ============
st.set_page_config(page_title="Cek Bet — Wizard", layout="centered", page_icon="🎯")
TEAMS = sorted(RATINGS.keys())

if "step" not in st.session_state: st.session_state.step = 1
def goto(n): st.session_state.step = n

st.title("🎯 Cek Bet")
st.caption("Jawab bertahap. Model hitung fair value & kasih verdict GAS / SKIP. "
           "Kalau tak ada value, dia bilang SKIP — tak memaksa nunjuk pemenang.")

# progress
steps = ["Match", "Pasar", "Taruhan", "Verdict"]
st.progress(st.session_state.step / 4)
st.write(f"**Langkah {st.session_state.step}/4 — {steps[st.session_state.step-1]}**")
st.divider()

# ---- STEP 1: MATCH ----
if st.session_state.step == 1:
    st.subheader("Match apa?")
    c1, c2 = st.columns(2)
    home = c1.selectbox("Tim A", TEAMS, index=TEAMS.index("Argentina"))
    away = c2.selectbox("Tim B", [t for t in TEAMS if t != home],
                        index=0)
    neutral = st.toggle("Venue netral (Piala Dunia)", value=True)
    with st.expander("Penyesuaian Elo (opsional — suspensi/cedera)"):
        cc1, cc2 = st.columns(2)
        adj_h = cc1.number_input(f"Adj {home}", -200.0, 200.0, 0.0, 5.0)
        adj_a = cc2.number_input(f"Adj {away}", -200.0, 200.0, 0.0, 5.0)
    if st.button("Lanjut →", type="primary"):
        st.session_state.update(home=home, away=away, neutral=neutral,
                                adj_h=adj_h, adj_a=adj_a)
        goto(2); st.rerun()

# ---- STEP 2: PASAR ----
elif st.session_state.step == 2:
    st.subheader(f"{st.session_state.home} vs {st.session_state.away}")
    st.write("Pasar apa yang mau dicek?")
    c1, c2 = st.columns(2)
    if c1.button("⚖️ Handicap", use_container_width=True):
        st.session_state.market = "ah"; goto(3); st.rerun()
    if c2.button("📊 Over / Under", use_container_width=True):
        st.session_state.market = "ou"; goto(3); st.rerun()
    st.divider()
    if st.button("← Kembali"): goto(1); st.rerun()

# ---- STEP 3: INPUT TARUHAN ----
elif st.session_state.step == 3:
    st.subheader("Taruhan yang kamu pertimbangkan")
    st.caption("Boleh isi beberapa untuk dibandingkan. Odds = dari papan bandar.")
    market = st.session_state.market
    n = st.number_input("Berapa taruhan?", 1, 6, 1, 1)
    bets = []
    if market == "ah":
        AH_LINES = [-3.0,-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,
                    0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]
        for i in range(int(n)):
            st.markdown(f"**Taruhan #{i+1}**")
            a, b, c = st.columns(3)
            team = a.selectbox("Tim", [st.session_state.home, st.session_state.away],
                               key=f"t{i}")
            line = b.selectbox("Garis (pada tim itu)", AH_LINES, index=8, key=f"l{i}")
            odds = c.number_input("Odds bandar", 1.01, 100000.0, 1.90, 0.01, key=f"o{i}")
            bets.append(("ah", team, line, odds))
    else:
        OU_LINES = [0.5,1.0,1.5,2.0,2.25,2.5,2.75,3.0,3.5,4.0]
        for i in range(int(n)):
            st.markdown(f"**Taruhan #{i+1}**")
            a, b, c = st.columns(3)
            line = a.selectbox("Garis total", OU_LINES, index=5, key=f"l{i}")
            side = b.selectbox("Sisi", ["Over", "Under"], key=f"s{i}")
            odds = c.number_input("Odds bandar", 1.01, 100000.0, 1.90, 0.01, key=f"o{i}")
            bets.append(("ou", side, line, odds))
    cc1, cc2 = st.columns(2)
    if cc1.button("← Kembali"): goto(2); st.rerun()
    if cc2.button("Lihat verdict →", type="primary"):
        st.session_state.bets = bets; goto(4); st.rerun()

# ---- STEP 4: VERDICT ----
elif st.session_state.step == 4:
    s = st.session_state
    M = matrix_from_params(PARAMS, RATINGS[s.home], RATINGS[s.away],
                           s.adj_h, s.adj_a, neutral=s.neutral)
    st.subheader(f"Verdict — {s.home} vs {s.away}")

    results = []
    for bet in s.bets:
        if bet[0] == "ah":
            _, team, line, odds = bet
            if team == s.home:
                res = ah_fair(M, line, "home")
            else:
                res = ah_fair(M, -line, "away")  # konvensi away dibalik
            label = f"{team} {line:+.2f}"
        else:
            _, side, line, odds = bet
            res = ou_fair(M, line, side.lower())
            label = f"{side} {line}"
        wp, fair = res["win_prob"], res["fair_odds"]
        e = ev(wp, odds)
        results.append({"label": label, "odds": odds, "fair": fair,
                        "wp": wp, "ev": e})

    # urut dari EV tertinggi
    results.sort(key=lambda x: x["ev"], reverse=True)

    any_positive = any(r["ev"] > 0 for r in results)

    for idx, r in enumerate(results):
        is_best = idx == 0
        verdict = "✅ GAS" if r["ev"] > 0 else "⛔ SKIP"
        box = st.success if r["ev"] > 0 else st.error
        tag = " — paling favourable" if is_best and r["ev"] > 0 else ""
        box(f"**{verdict}** · {r['label']}{tag}\n\n"
            f"Odds kamu **{r['odds']:.2f}** vs fair **{r['fair']:.2f}** · "
            f"peluang model {r['wp']:.1%} · EV **{r['ev']:+.1%}**")

    st.divider()
    if not any_positive:
        st.info("**Kesimpulan: SKIP SEMUA.** Tak ada taruhan EV positif — harga "
                "bandar adil atau di bawah fair value. Tak ada value adalah "
                "jawaban yang sah, bukan kegagalan. Di laga populer ini normal.")
    else:
        st.warning("**Ada EV+, tapi hati-hati.** EV positif di laga besar sering "
                   "berarti: line stale/in-play, atau model buta variabel yang "
                   "bandar tahu (lineup, cedera). Curigai dulu — EV+ bukan "
                   "otomatis sinyal gas. Verifikasi sebelum pasang.")

    if st.button("🔄 Cek taruhan lain"):
        goto(1); st.rerun()
