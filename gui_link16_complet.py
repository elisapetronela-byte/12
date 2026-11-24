# ============================================================
#  SIMULATOR LINK 16 - INTERFATA GRAFICA (tkinter)
#  Saptamana 1: schita interfetei + planul de frecvente
#  Saptamana 2: generarea semnalelor (CCSK + MSK) si saltul de frecventa
#  Saptamana 3: bruiaj AWGN pe subbenzi aplicat asupra semnalului
#
#  Mesajul scris de utilizator: biti -> simboluri de 5 biti -> CCSK -> MSK
#  -> salt de frecventa -> CANAL (zgomot termic + bruiaj pe canalele bruiate)
#  -> filtru adaptat MSK -> corelare CCSK -> mesajul receptionat.
# ============================================================
import re
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ------------------------------------------------------------
# 1. PARAMETRI LINK 16 (saptamana 1)
# ------------------------------------------------------------
FRECVENTE_MHZ = np.concatenate([np.arange(969, 1009, 3),     # 14 canale
                                np.arange(1053, 1066, 3),    #  5 canale
                                np.arange(1113, 1207, 3)]    # 32 canale
                               ).astype(float)
NR_CANALE = len(FRECVENTE_MHZ)                               # 51
FRECVENTA_CENTRALA = (FRECVENTE_MHZ.min() + FRECVENTE_MHZ.max()) / 2
SUBBENZI = {"Inferioara (969-1008)": list(range(0, 14)),
            "Mediana (1053-1065)": list(range(14, 19)),
            "Superioara (1113-1206)": list(range(19, 51))}
TIPURI_BRUIAJ = ["Fara bruiaj", "Banda completa", "Banda partiala (rho)",
                 "Subbanda", "Canale personalizate"]

SALT_MINIM_MHZ = 30.0
RATA_CHIP = 5e6
CHIPURI_PE_SIMBOL = 32
BITI_PE_SIMBOL = 5
PERIOADA_IMPULS = 13e-6
IMPULSURI_PE_SLOT = 109
SPS = 4                                                      # esantioane pe chip

MESAJ_IMPLICIT = "Link 16 - mesaj de test. Tinta la azimut 045, distanta 12 km."


# ------------------------------------------------------------
# 2. BRUIAJ AWGN PE SUBBENZI (saptamana 3)
# ------------------------------------------------------------
def parseaza_canale(text):
    """'0-9, 30, 40-45' -> [0..9, 30, 40..45];  '969-1008' (MHz) -> indicii canalelor."""
    canale = set()
    for parte in re.split(r"[,; ]+", text.strip()):
        if not parte:
            continue
        try:
            a, _, b = parte.partition("-")
            a, b = float(a), float(b or a)
        except ValueError:
            raise ValueError(f"Nu pot interpreta '{parte}'. Exemplu corect: 0-9, 30, 40-45")
        if max(a, b) > 100:                                  # valori in MHz
            idx = np.flatnonzero((FRECVENTE_MHZ >= min(a, b)) & (FRECVENTE_MHZ <= max(a, b)))
            canale.update(idx.tolist())
        else:
            canale.update(c for c in range(int(min(a, b)), int(max(a, b)) + 1) if 0 <= c < NR_CANALE)
    return sorted(canale)


def canale_bruiate(tip, rho=0.25, subbanda="Mediana (1053-1065)", canale_text="", seed=7):
    """Lista canalelor bruiate pentru fiecare tip de bruiaj."""
    if tip == "Banda completa":
        return list(range(NR_CANALE))
    if tip == "Subbanda":
        return SUBBENZI[subbanda]
    if tip == "Banda partiala (rho)":
        n = max(1, int(round(rho * NR_CANALE)))
        return sorted(np.random.default_rng(seed).choice(NR_CANALE, n, replace=False).tolist())
    if tip == "Canale personalizate":
        return parseaza_canale(canale_text)
    return []


def js_pe_canal(js_db, nr_bruiate, putere_totala_constanta=True):
    """
    J/S = puterea bruiajului / puterea semnalului.
    Cu putere TOTALA constanta, bruiajul isi imparte puterea pe canalele bruiate:
    cu cat bruiaza mai putine canale, cu atat fiecare este bruiat mai puternic.
    """
    if nr_bruiate == 0:
        return 0.0
    js = 10 ** (js_db / 10)
    return js / nr_bruiate if putere_totala_constanta else js


# ------------------------------------------------------------
# 3. SEMNALE SI SALT DE FRECVENTA (saptamana 2)
# ------------------------------------------------------------
def text_in_biti(text):
    """Text -> biti (UTF-8), completati cu 0 pana la un multiplu de 5."""
    biti = np.unpackbits(np.frombuffer(text.encode("utf-8"), dtype=np.uint8))
    rest = (-len(biti)) % BITI_PE_SIMBOL
    return np.concatenate([biti, np.zeros(rest, dtype=np.uint8)]).astype(int), len(biti)


def biti_in_text(biti, nr_biti):
    octeti = np.packbits(np.asarray(biti[:nr_biti], dtype=np.uint8)).tobytes()
    return octeti.decode("utf-8", errors="replace")


def biti_in_simboluri(biti):
    return np.asarray(biti).reshape(-1, BITI_PE_SIMBOL) @ np.array([16, 8, 4, 2, 1])


def simboluri_in_biti(simboluri):
    return ((np.asarray(simboluri)[:, None] >> np.arange(4, -1, -1)) & 1).ravel()


# CCSK: o secventa de baza de 32 de chipuri; simbolul k = secventa rotita cu k pozitii
S0_HEX = 0x7CE90AEC
S0 = np.array([1 - 2 * ((S0_HEX >> (31 - i)) & 1) for i in range(32)])
TABEL_CCSK = np.array([np.roll(S0, -k) for k in range(32)])


def ccsk_moduleaza(simboluri):
    return TABEL_CCSK[np.asarray(simboluri)]


def ccsk_demoduleaza(chipuri):
    corelatii = chipuri @ TABEL_CCSK.T
    return np.argmax(corelatii, axis=1), corelatii


# MSK: chipurile pare pe I, cele impare pe Q (decalat cu Tc), forma semi-sinusoidala
def semisinus(sps):
    return np.sin(np.pi * (np.arange(2 * sps) + 0.5) / (2 * sps))


def msk_moduleaza(chipuri, sps=SPS):
    chipuri = np.atleast_2d(chipuri)
    p = semisinus(sps)
    L = (CHIPURI_PE_SIMBOL + 1) * sps
    I = np.zeros((chipuri.shape[0], L))
    Q = np.zeros((chipuri.shape[0], L))
    for k in range(CHIPURI_PE_SIMBOL):
        s = k * sps
        if k % 2 == 0:
            I[:, s:s + 2 * sps] += chipuri[:, k:k + 1] * p
        else:
            Q[:, s:s + 2 * sps] += chipuri[:, k:k + 1] * p
    return I + 1j * Q


def msk_demoduleaza(semnal, sps=SPS):
    semnal = np.atleast_2d(semnal)
    p = semisinus(sps)
    chipuri = np.zeros((semnal.shape[0], CHIPURI_PE_SIMBOL))
    for k in range(CHIPURI_PE_SIMBOL):
        f = semnal[:, k * sps:k * sps + 2 * sps]
        chipuri[:, k] = (f.real if k % 2 == 0 else f.imag) @ p / np.sum(p ** 2)
    return chipuri


# Salt de frecventa: aceeasi cheie + acelasi slot -> aceeasi secventa; minim 30 MHz intre salturi
CANALE_PERMISE = [np.flatnonzero(np.abs(FRECVENTE_MHZ - f) >= SALT_MINIM_MHZ)
                  for f in FRECVENTE_MHZ]


def secventa_salt(nr_slot, nr_impulsuri, cheie, salt_activ=True, canal_fix=25):
    if not salt_activ:
        return np.full(nr_impulsuri, canal_fix)
    rng = np.random.default_rng([cheie, nr_slot])
    canale = np.empty(nr_impulsuri, dtype=int)
    canale[0] = rng.integers(NR_CANALE)
    for i in range(1, nr_impulsuri):
        permise = CANALE_PERMISE[canale[i - 1]]
        canale[i] = permise[rng.integers(len(permise))]
    return canale


def secventa_mesaj(n, cheie, salt_activ, canal_fix):
    """Canalele pentru n impulsuri, impartite in sloturi de cate 109."""
    return np.concatenate([
        secventa_salt(s, min(IMPULSURI_PE_SLOT, n - s * IMPULSURI_PE_SLOT), cheie, salt_activ, canal_fix)
        for s in range(int(np.ceil(n / IMPULSURI_PE_SLOT)))])


# ------------------------------------------------------------
# 4. CANALUL: zgomot termic + bruiaj (saptamana 3)
# ------------------------------------------------------------
def zgomot_pe_impuls(canale, bruiate, js_db, putere_totala, zgomot_activ, ebn0_db):
    """
    Densitatea de zgomot pentru fiecare impuls (energia unui chip = 1):
        N0          pe canalele libere
        N0 + NJ     pe canalele bruiate  (NJ = J/S pe canal)
    """
    masca = np.zeros(NR_CANALE, dtype=bool)
    masca[list(bruiate)] = True
    pe_bruiat = masca[canale]
    ecn0 = 10 ** (ebn0_db / 10) * BITI_PE_SIMBOL / CHIPURI_PE_SIMBOL   # 5 biti pe 32 de chipuri
    n0 = 1 / ecn0 if zgomot_activ else 0.0
    nj = js_pe_canal(js_db, masca.sum(), putere_totala)
    return n0 + pe_bruiat * nj, pe_bruiat, nj


def transmite_mesaj(text, cheie=2026, salt_activ=True, canal_fix=25, zgomot_activ=True,
                    ebn0_db=10.0, bruiate=(), js_db=20.0, putere_totala=True, seed=1):
    """Trece mesajul prin tot lantul (la nivel de forma de unda) si intoarce datele intermediare."""
    rng = np.random.default_rng(seed)
    biti_tx, nr_biti = text_in_biti(text)
    simb_tx = biti_in_simboluri(biti_tx)
    canale = secventa_mesaj(len(simb_tx), cheie, salt_activ, canal_fix)
    n_total, pe_bruiat, nj = zgomot_pe_impuls(canale, bruiate, js_db, putere_totala, zgomot_activ, ebn0_db)

    tx = msk_moduleaza(ccsk_moduleaza(simb_tx))
    sigma = np.sqrt(SPS * n_total / 2)[:, None]
    rx = tx + sigma * (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape))
    simb_rx, corelatii = ccsk_demoduleaza(msk_demoduleaza(rx))
    biti_rx = simboluri_in_biti(simb_rx)
    err = simb_rx != simb_tx
    return {
        "text": text, "text_rx": biti_in_text(biti_rx, nr_biti), "nr_biti": nr_biti,
        "simb_tx": simb_tx, "simb_rx": simb_rx, "canale": canale, "jam": pe_bruiat, "err": err,
        "tx": tx, "rx": rx, "corelatii": corelatii,
        "erori_biti": int(np.sum(biti_rx[:nr_biti] != biti_tx[:nr_biti])),
        "erori_simboluri": int(err.sum()),
        "zgomot_activ": zgomot_activ, "ebn0_db": ebn0_db, "salt_activ": salt_activ,
        "bruiate": list(bruiate), "js_db": js_db, "nj": nj,
    }


def statistica(bruiate, js_db, putere_totala, salt_activ, canal_fix, zgomot_activ, ebn0_db,
               cheie=2026, nr_sloturi=40, seed=1):
    """
    Simulare rapida (nivel chip, echivalent cu forma de unda) cu simboluri aleatoare.
    Folosita pentru statistica pe canale si pentru curbe.
    """
    rng = np.random.default_rng(seed)
    n = nr_sloturi * IMPULSURI_PE_SLOT
    simb = rng.integers(0, 32, n)
    canale = np.concatenate([secventa_salt(s, IMPULSURI_PE_SLOT, cheie, salt_activ, canal_fix)
                             for s in range(nr_sloturi)])
    n_total, pe_bruiat, _ = zgomot_pe_impuls(canale, bruiate, js_db, putere_totala, zgomot_activ, ebn0_db)
    chipuri = ccsk_moduleaza(simb)
    soft = chipuri + np.sqrt(n_total / 2)[:, None] * rng.standard_normal(chipuri.shape)
    err = ccsk_demoduleaza(soft)[0] != simb
    return canale, pe_bruiat, err


# ------------------------------------------------------------
# 5. FUNCTIILE DE DESENARE
# ------------------------------------------------------------
def deseneaza_plan(fig, r, bruiate, tip_bruiaj):
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    bruiate = set(bruiate)
    for i, fr in enumerate(FRECVENTE_MHZ):
        ax1.vlines(fr, 0, 1, color="red" if i in bruiate else "tab:blue", lw=2.5)
    for ax in (ax1, ax2):
        for fiff in (1030, 1090):
            ax.axvspan(fiff - 7, fiff + 7, color="gray", alpha=0.15)
    for fiff in (1030, 1090):
        ax1.text(fiff, 1.05, f"IFF {fiff}", ha="center", fontsize=8, color="dimgray")
    ax1.set_ylim(0, 1.35)
    ax1.set_yticks([])
    ax1.set_title(f"Cele 51 de canale - {tip_bruiaj}: {len(bruiate)} canale bruiate (rosu)", fontsize=10)
    if r is not None:
        utilizari = np.bincount(r["canale"], minlength=NR_CANALE)
        culori = ["red" if i in bruiate else "tab:blue" for i in range(NR_CANALE)]
        ax2.bar(FRECVENTE_MHZ, utilizari, width=2.2, color=culori)
        ax2.set_title(f"Impulsurile mesajului pe fiecare canal - {100 * r['jam'].mean():.0f}% "
                      f"au cazut pe canale bruiate", fontsize=10)
    ax2.set_xlim(955, 1220)
    ax2.set_xlabel("Frecventa [MHz]")
    ax2.set_ylabel("Nr. impulsuri")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()


def deseneaza_salt(fig, r, idx):
    fig.clear()
    ax = fig.add_subplot(111)
    for c in r["bruiate"]:
        ax.axhspan(FRECVENTE_MHZ[c] - 1.5, FRECVENTE_MHZ[c] + 1.5, color="#f4c7c7", lw=0)
    n = len(r["canale"])
    t = np.arange(n) * PERIOADA_IMPULS * 1e6
    f = FRECVENTE_MHZ[r["canale"]]
    jam, err = r["jam"], r["err"]
    ax.plot(t, f, color="#bbbbbb", lw=0.5)
    ax.scatter(t[~jam], f[~jam], s=14, color="tab:blue", label="impuls pe canal liber")
    if jam.any():
        ax.scatter(t[jam], f[jam], s=22, color="tab:red", label="impuls pe canal bruiat")
    if err.any():
        ax.scatter(t[err], f[err], s=75, facecolors="none", edgecolors="black", label="simbol gresit")
    ax.scatter([t[idx]], [f[idx]], s=170, marker="*", color="orange", zorder=5, label=f"impulsul {idx}")
    for k in range(IMPULSURI_PE_SLOT, n, IMPULSURI_PE_SLOT):
        ax.axvline(k * PERIOADA_IMPULS * 1e6, color="gray", ls="--", lw=0.8)
    ax.set_ylim(960, 1215)
    ax.set_xlabel("Timp [us]  (linia punctata = un nou slot; benzile roz = canale bruiate)")
    ax.set_ylabel("Frecventa [MHz]")
    ax.set_title("Salt de frecventa pe 51 de canale sub bruiaj" if r["salt_activ"]
                 else "FARA salt: toate impulsurile pe acelasi canal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()


def deseneaza_semnal(fig, r, idx):
    fig.clear()
    simbol = int(r["simb_tx"][idx])
    tc = np.arange(r["tx"].shape[1]) / (SPS * RATA_CHIP) * 1e6
    axe = fig.subplots(3, 1, sharex=True)
    axe[0].step(np.arange(33) * 0.2, np.append(TABEL_CCSK[simbol], TABEL_CCSK[simbol][-1]),
                where="post", color="black")
    axe[0].set_title(f"Simbolul {simbol} -> secventa CCSK (32 chipuri)", fontsize=10)
    for ax, comp, nume, cul in ((axe[1], np.real, "I (chipuri pare)", "tab:blue"),
                                (axe[2], np.imag, "Q (chipuri impare, decalat cu Tc)", "tab:orange")):
        ax.plot(tc, comp(r["rx"][idx]), color="#bbbbbb", lw=0.7, label="receptionat")
        ax.plot(tc, comp(r["tx"][idx]), color=cul, lw=1.4, label="transmis")
        ax.set_title("Componenta " + nume, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    axe[2].set_xlabel("Timp [us]")
    for ax in axe:
        ax.grid(alpha=0.3)
    c = r["canale"][idx]
    bruiat = bool(r["jam"][idx])
    fig.suptitle(f"Impulsul {idx} - canal {c} ({FRECVENTE_MHZ[c]:.0f} MHz)"
                 + ("  -  CANAL BRUIAT" if bruiat else "  -  canal liber"),
                 color="red" if bruiat else "black", fontsize=11)
    fig.tight_layout()


def deseneaza_corelare(fig, r, idx):
    fig.clear()
    ax = fig.add_subplot(111)
    trimis, decis = int(r["simb_tx"][idx]), int(r["simb_rx"][idx])
    culori = ["lightgray"] * 32
    culori[trimis] = "tab:blue"
    if decis != trimis:
        culori[decis] = "tab:red"
    ax.bar(np.arange(32), r["corelatii"][idx], color=culori)
    stare = "CORECT" if decis == trimis else "GRESIT"
    loc = "canal BRUIAT" if r["jam"][idx] else "canal liber"
    ax.set_title(f"Corelare CCSK - impulsul {idx} ({loc}): trimis {trimis}, decis {decis} ({stare})")
    ax.set_xlabel("Simbol candidat (0..31)")
    ax.set_ylabel("Corelatie")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def deseneaza_spectrograma(fig, r, nr=24):
    fig.clear()
    ax = fig.add_subplot(111)
    nr = min(nr, len(r["simb_tx"]))
    fs = 300e6
    rng = np.random.default_rng(0)
    forme = msk_moduleaza(ccsk_moduleaza(r["simb_tx"][:nr]), int(fs / RATA_CHIP))
    perioada = int(PERIOADA_IMPULS * fs)
    N = perioada * nr
    rf = np.zeros(N, dtype=complex)
    for i in range(nr):
        df = (FRECVENTE_MHZ[r["canale"][i]] - FRECVENTA_CENTRALA) * 1e6
        t = (i * perioada + np.arange(forme.shape[1])) / fs
        rf[i * perioada:i * perioada + forme.shape[1]] = forme[i] * np.exp(2j * np.pi * df * t)
    if r["bruiate"] and r["nj"] > 0:
        # zgomot de bruiaj filtrat doar pe canalele bruiate (latime 3 MHz fiecare)
        Z = np.fft.fft(rng.standard_normal(N) + 1j * rng.standard_normal(N))
        frecv = np.fft.fftfreq(N, 1 / fs)
        masca = np.zeros(N, dtype=bool)
        for c in r["bruiate"]:
            masca |= np.abs(frecv - (FRECVENTE_MHZ[c] - FRECVENTA_CENTRALA) * 1e6) <= 1.5e6
        Z[~masca] = 0
        jam = np.fft.ifft(Z)
        putere = r["nj"] * len(r["bruiate"]) * 3 / 5      # puterea totala a bruiajului (semnal = 1)
        rf += jam * np.sqrt(putere / max(np.mean(np.abs(jam) ** 2), 1e-30))
    rf += 0.03 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    _, _, _, im = ax.specgram(rf, NFFT=512, Fs=fs / 1e6, Fc=FRECVENTA_CENTRALA, noverlap=256, cmap="viridis")
    vmax = np.percentile(np.ma.getdata(im.get_array()), 99.5)
    im.set_clim(vmax - 50, vmax)
    ax.set_ylim(960, 1215)
    ax.set_xlabel("Timp [us]")
    ax.set_ylabel("Frecventa [MHz]")
    ax.set_title(f"Spectrograma: primele {nr} impulsuri + bruiajul continuu pe canalele bruiate")
    fig.tight_layout()


def deseneaza_erori_canal(fig, canale, jam, err, bruiate):
    fig.clear()
    ax = fig.add_subplot(111)
    utilizari = np.bincount(canale, minlength=NR_CANALE)
    erori = np.bincount(canale, weights=err, minlength=NR_CANALE)
    rata = np.divide(erori, utilizari, out=np.zeros(NR_CANALE), where=utilizari > 0)
    bruiate = set(bruiate)
    ax.bar(range(NR_CANALE), 100 * rata,
           color=["tab:red" if c in bruiate else "tab:blue" for c in range(NR_CANALE)])
    ser_b = 100 * err[jam].mean() if jam.any() else 0
    ser_l = 100 * err[~jam].mean() if (~jam).any() else 0
    ax.set_title(f"Simboluri gresite pe fiecare canal ({len(canale)} simboluri aleatoare)\n"
                 f"pe canalele bruiate: {ser_b:.1f}%   |   pe canalele libere: {ser_l:.1f}%", fontsize=10)
    ax.set_xlabel("Indice canal (0 = 969 MHz ... 50 = 1206 MHz); rosu = canal bruiat")
    ax.set_ylabel("Simboluri gresite [%]")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()


def deseneaza_curbe(fig, js, curbe, ebn0_db):
    fig.clear()
    ax = fig.add_subplot(111)
    for (nume, ser), mk in zip(curbe.items(), "osd^x"):
        ax.semilogy(js, np.maximum(ser, 1e-4), marker=mk, label=nume)
    ax.axhline(8 / 31, color="gray", ls="--")
    ax.text(js[0], 8 / 31 * 1.15, "limita de corectie Reed-Solomon (8 din 31) - saptamana 4",
            fontsize=8, color="gray")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel("J/S total [dB]")
    ax.set_ylabel("Rata de eroare a simbolurilor (SER)")
    ax.set_title(f"SER in functie de puterea bruiajului (Eb/N0 = {ebn0_db} dB, fara codare)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()


def text_rezultate(r, tip_bruiaj, putere_totala):
    n = len(r["simb_tx"])
    jam, err = r["jam"], r["err"]
    salt = np.abs(np.diff(FRECVENTE_MHZ[r["canale"]]))
    jsc = f"{10 * np.log10(r['nj']):.1f} dB" if r["nj"] > 0 else "-"
    linii = [
        "REZULTATE",
        "=" * 56,
        f"Caractere in mesaj          : {len(r['text'])}",
        f"Biti transmisi              : {r['nr_biti']}",
        f"Simboluri (impulsuri)       : {n}",
        f"Sloturi TDMA folosite       : {int(np.ceil(n / IMPULSURI_PE_SLOT))}",
        f"Canale distincte folosite   : {len(np.unique(r['canale']))} din 51",
        (f"Salt minim intre impulsuri  : {salt.min():.0f} MHz" if r["salt_activ"] and len(salt)
         else "Salt de frecventa          : DEZACTIVAT"),
        "Zgomot termic               : " + (f"DA, Eb/N0 = {r['ebn0_db']} dB" if r["zgomot_activ"] else "NU"),
        "",
        "BRUIAJ",
        f"Tip                         : {tip_bruiaj}",
        f"Canale bruiate              : {len(r['bruiate'])} din 51",
        f"J/S total / pe canal        : {r['js_db']} dB / {jsc}"
        + ("  (putere impartita pe canale)" if putere_totala else "  (putere constanta pe canal)"),
        f"Impulsuri pe canale bruiate : {int(jam.sum())} din {n} ({100 * jam.mean():.1f}%)",
        "",
        "ERORI (fara codare)",
        f"Simboluri gresite - total   : {int(err.sum())} din {n} ({100 * err.mean():.1f}%)",
        f"   pe canale bruiate        : {int(err[jam].sum())} din {int(jam.sum())}"
        + (f" ({100 * err[jam].mean():.1f}%)" if jam.any() else ""),
        f"   pe canale libere         : {int(err[~jam].sum())} din {int((~jam).sum())}"
        + (f" ({100 * err[~jam].mean():.1f}%)" if (~jam).any() else ""),
        f"Biti gresiti                : {r['erori_biti']} din {r['nr_biti']}",
        f"BER                         : {r['erori_biti'] / max(r['nr_biti'], 1):.2e}",
        "",
        "Mesaj trimis:", r["text"], "",
        "Mesaj receptionat:", r["text_rx"], "",
        "Mesajul a ajuns INTACT." if r["text_rx"] == r["text"] else
        "Mesajul a ajuns CU ERORI. (Reed-Solomon, in saptamana 4, va corecta aceste erori.)",
    ]
    return "\n".join(linii)


# ------------------------------------------------------------
# 6. INTERFATA GRAFICA
# ------------------------------------------------------------
class Aplicatie(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulator Link 16")
        self.geometry("1380x840")
        self.rezultat = None

        # ---- variabile ----
        self.v_zgomot = tk.BooleanVar(value=True)
        self.v_ebn0 = tk.DoubleVar(value=10)
        self.v_tip = tk.StringVar(value=TIPURI_BRUIAJ[2])
        self.v_js = tk.DoubleVar(value=20)
        self.v_rho = tk.DoubleVar(value=0.25)
        self.v_sub = tk.StringVar(value=list(SUBBENZI)[1])
        self.v_canale = tk.StringVar(value="10-20, 35")
        self.v_putere = tk.BooleanVar(value=True)
        self.v_salt = tk.BooleanVar(value=True)
        self.v_canal = tk.IntVar(value=25)
        self.v_cheie = tk.IntVar(value=2026)
        self.v_rs = tk.BooleanVar(value=True)
        self.v_idx = tk.IntVar(value=0)
        self.v_info = tk.StringVar()
        self.v_stare = tk.StringVar(value="Scrieti un mesaj si apasati Transmite.")

        ttk.Label(self, textvariable=self.v_stare, anchor="w").pack(side="bottom", fill="x", padx=6, pady=3)

        # ---- panoul din stanga ----
        stanga = ttk.Frame(self, padding=8)
        stanga.pack(side="left", fill="y")

        g = ttk.LabelFrame(stanga, text="1. Trafic de date", padding=6)
        g.pack(fill="x", pady=3)
        self.txt_mesaj = tk.Text(g, height=4, width=38, wrap="word")
        self.txt_mesaj.insert("1.0", MESAJ_IMPLICIT)
        self.txt_mesaj.pack(fill="x")

        g = ttk.LabelFrame(stanga, text="2. Canal (zgomot termic)", padding=6)
        g.pack(fill="x", pady=3)
        ttk.Checkbutton(g, text="Zgomot termic AWGN", variable=self.v_zgomot,
                        command=self._stari).pack(anchor="w")
        self.sp_ebn0 = ttk.Spinbox(g, from_=-10, to=30, increment=1, textvariable=self.v_ebn0, width=8)
        self._rand(g, "Eb/N0 [dB]:", self.sp_ebn0)

        g = ttk.LabelFrame(stanga, text="3. Bruiaj AWGN pe subbenzi", padding=6)
        g.pack(fill="x", pady=3)
        self._rand(g, "Tip bruiaj:", ttk.Combobox(g, textvariable=self.v_tip, values=TIPURI_BRUIAJ,
                                                 state="readonly", width=22))
        self._rand(g, "J/S total [dB]:", ttk.Spinbox(g, from_=-20, to=70, textvariable=self.v_js, width=8))
        self.sp_rho = ttk.Spinbox(g, from_=0.02, to=1, increment=0.05, textvariable=self.v_rho, width=8)
        self._rand(g, "Fractia rho:", self.sp_rho)
        self.cb_sub = ttk.Combobox(g, textvariable=self.v_sub, values=list(SUBBENZI), state="readonly", width=22)
        self._rand(g, "Subbanda:", self.cb_sub)
        self.en_canale = ttk.Entry(g, textvariable=self.v_canale, width=24)
        self._rand(g, "Canale:", self.en_canale)
        ttk.Label(g, text="ex. 0-9, 30, 40-45  sau  969-1008 (MHz)", foreground="gray").pack(anchor="w")
        ttk.Checkbutton(g, text="Putere totala constanta (impartita pe canale)",
                        variable=self.v_putere, command=self._actualizeaza_bruiaj).pack(anchor="w")
        ttk.Label(g, textvariable=self.v_info, foreground="#b03030", wraplength=280,
                  justify="left").pack(anchor="w", pady=(4, 0))

        g = ttk.LabelFrame(stanga, text="4. Sistem Link 16", padding=6)
        g.pack(fill="x", pady=3)
        ttk.Checkbutton(g, text="Salt de frecventa (51 canale)", variable=self.v_salt,
                        command=self._actualizeaza_bruiaj).pack(anchor="w")
        self.sp_canal = ttk.Spinbox(g, from_=0, to=50, textvariable=self.v_canal, width=8)
        self._rand(g, "Canal fix:", self.sp_canal)
        self._rand(g, "Cheie retea:", ttk.Spinbox(g, from_=0, to=999999, textvariable=self.v_cheie, width=8))
        ttk.Checkbutton(g, text="Reed-Solomon (saptamana 4)", variable=self.v_rs,
                        state="disabled").pack(anchor="w")

        g = ttk.LabelFrame(stanga, text="5. Impulsul afisat", padding=6)
        g.pack(fill="x", pady=3)
        self.sp_idx = ttk.Spinbox(g, from_=0, to=0, textvariable=self.v_idx, width=8,
                                  command=self._redeseneaza_impuls)
        self._rand(g, "Nr. impuls:", self.sp_idx)
        ttk.Button(g, text="Urmatorul impuls bruiat ->",
                   command=lambda: self._urmator("jam")).pack(fill="x", pady=1)
        ttk.Button(g, text="Urmatorul simbol gresit ->",
                   command=lambda: self._urmator("err")).pack(fill="x", pady=1)

        ttk.Button(stanga, text="Transmite", command=self.transmite).pack(fill="x", pady=8)

        # ---- tab-urile din dreapta ----
        self.taburi = ttk.Notebook(self)
        self.taburi.pack(side="right", fill="both", expand=True)
        self.figuri = {}
        for nume in ("Plan de frecvente", "Salt de frecventa", "Semnal impuls",
                     "Corelare CCSK", "Spectrograma", "Erori pe canal"):
            self.figuri[nume] = self._tab_grafic(nume)
        f = ttk.Frame(self.taburi)
        self.txt_rez = tk.Text(f, wrap="word", font=("Courier New", 10))
        self.txt_rez.pack(fill="both", expand=True)
        self.taburi.add(f, text="Rezultate")
        self._tab_curbe()
        f = ttk.Frame(self.taburi)
        ttk.Label(f, text="Pachete\n(se completeaza in saptamana 4: pachete, Reed-Solomon)",
                  justify="center").place(relx=0.5, rely=0.5, anchor="center")
        self.taburi.add(f, text="Pachete")

        # parametrii de bruiaj actualizeaza imediat planul de frecvente
        for v in (self.v_tip, self.v_rho, self.v_sub, self.v_canale, self.v_js):
            v.trace_add("write", lambda *_: self._actualizeaza_bruiaj())

        self._stari()
        self.transmite()

    # -------------------------------------------------------- construire
    @staticmethod
    def _rand(parinte, text, widget):
        r = ttk.Frame(parinte)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text=text, width=13).pack(side="left")
        widget.pack(in_=r, side="left")     # campul pe acelasi rand cu eticheta
        widget.lift(r)                      # vizibil deasupra cadrului r

    def _tab_grafic(self, nume, parinte=None):
        f = parinte or ttk.Frame(self.taburi)
        fig = Figure(figsize=(9, 6))
        canvas = FigureCanvasTkAgg(fig, master=f)
        bara = NavigationToolbar2Tk(canvas, f, pack_toolbar=False)   # zoom + salvare PNG
        bara.update()
        bara.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        if parinte is None:
            self.taburi.add(f, text=nume)
        return fig, canvas

    def _tab_curbe(self):
        f = ttk.Frame(self.taburi)
        sus = ttk.Frame(f)
        sus.pack(fill="x", padx=6, pady=4)
        ttk.Label(sus, text="Curbe SER pentru configuratia din stanga, comparata cu: fara salt de "
                            "frecventa si bruiaj pe banda completa.").pack(side="left")
        ttk.Button(sus, text="Calculeaza curbe", command=self.calculeaza_curbe).pack(side="right")
        jos = ttk.Frame(f)
        jos.pack(fill="both", expand=True)
        self.figuri["Curbe SER"] = self._tab_grafic("Curbe SER", parinte=jos)
        self.taburi.add(f, text="Curbe SER")

    # -------------------------------------------------------- utilitare
    def _deseneaza(self, nume, functie, *args):
        fig, canvas = self.figuri[nume]
        functie(fig, *args)
        canvas.draw_idle()

    def _stari(self):
        self.sp_canal.configure(state="disabled" if self.v_salt.get() else "normal")
        self.sp_ebn0.configure(state="normal" if self.v_zgomot.get() else "disabled")
        tip = self.v_tip.get()
        self.sp_rho.configure(state="normal" if tip == "Banda partiala (rho)" else "disabled")
        self.cb_sub.configure(state="readonly" if tip == "Subbanda" else "disabled")
        self.en_canale.configure(state="normal" if tip == "Canale personalizate" else "disabled")

    def _bruiate(self):
        """Canalele bruiate din parametrii curenti (ridica ValueError daca textul e gresit)."""
        try:
            rho = float(self.v_rho.get())
        except (tk.TclError, ValueError):
            rho = 0.25
        return canale_bruiate(self.v_tip.get(), min(max(rho, 0.0), 1.0), self.v_sub.get(), self.v_canale.get())

    def _actualizeaza_bruiaj(self):
        self._stari()
        try:
            bruiate = self._bruiate()
            js = float(self.v_js.get())
        except (ValueError, tk.TclError) as e:
            self.v_info.set(f"Eroare: {e}" if isinstance(e, ValueError) else "J/S invalid")
            return
        nj = js_pe_canal(js, len(bruiate), self.v_putere.get())
        if bruiate:
            text = (f"{len(bruiate)} canale bruiate ({100 * len(bruiate) / NR_CANALE:.0f}% din banda), "
                    f"J/S pe canal = {10 * np.log10(nj):.1f} dB")
        else:
            text = "Niciun canal bruiat"
        if not self.v_salt.get():
            try:
                c = int(self.v_canal.get())
                text += f"\nCanal fix {c}: " + ("BRUIAT" if c in bruiate else "liber")
            except (tk.TclError, ValueError):
                pass
        self.v_info.set(text + "\n(apasati Transmite pentru a aplica)")
        self._deseneaza("Plan de frecvente", deseneaza_plan, self.rezultat, bruiate, self.v_tip.get())

    def _parametri(self):
        """Citeste si valideaza toti parametrii. Intoarce un dict sau None."""
        try:
            p = dict(cheie=int(self.v_cheie.get()), canal_fix=int(self.v_canal.get()),
                     ebn0_db=float(self.v_ebn0.get()), js_db=float(self.v_js.get()))
        except (tk.TclError, ValueError):
            messagebox.showwarning("Valoare invalida", "Verificati campurile numerice.")
            return None
        if not 0 <= p["canal_fix"] < NR_CANALE:
            messagebox.showwarning("Valoare invalida", "Canalul fix trebuie sa fie intre 0 si 50.")
            return None
        try:
            p["bruiate"] = self._bruiate()
        except ValueError as e:
            messagebox.showwarning("Canale invalide", str(e))
            return None
        p.update(salt_activ=self.v_salt.get(), zgomot_activ=self.v_zgomot.get(),
                 putere_totala=self.v_putere.get())
        return p

    # -------------------------------------------------------- actiuni
    def transmite(self):
        text = self.txt_mesaj.get("1.0", "end-1c")
        if not text:
            messagebox.showwarning("Mesaj gol", "Scrieti un mesaj de transmis.")
            return
        p = self._parametri()
        if p is None:
            return
        self.rezultat = r = transmite_mesaj(text, **p)
        self.sp_idx.configure(to=len(r["simb_tx"]) - 1)
        if self.v_idx.get() >= len(r["simb_tx"]):
            self.v_idx.set(0)
        self._actualizeaza_bruiaj()
        self.v_info.set(self.v_info.get().replace("\n(apasati Transmite pentru a aplica)", ""))
        self._deseneaza("Spectrograma", deseneaza_spectrograma, r)
        stat = statistica(p["bruiate"], p["js_db"], p["putere_totala"], p["salt_activ"], p["canal_fix"],
                          p["zgomot_activ"], p["ebn0_db"], p["cheie"])
        self._deseneaza("Erori pe canal", deseneaza_erori_canal, *stat, p["bruiate"])
        self._redeseneaza_impuls()
        self.txt_rez.delete("1.0", "end")
        self.txt_rez.insert("1.0", text_rezultate(r, self.v_tip.get(), p["putere_totala"]))
        self.v_stare.set(f"Transmis: {len(r['simb_tx'])} impulsuri, {100 * r['jam'].mean():.0f}% pe canale "
                         f"bruiate, {r['erori_simboluri']} simboluri gresite. "
                         + ("Mesaj intact." if r["text_rx"] == text else "Mesaj cu erori."))

    def _redeseneaza_impuls(self):
        if self.rezultat is None:
            return
        try:
            idx = int(self.v_idx.get())
        except (tk.TclError, ValueError):
            idx = 0
        idx = int(np.clip(idx, 0, len(self.rezultat["simb_tx"]) - 1))
        self._deseneaza("Salt de frecventa", deseneaza_salt, self.rezultat, idx)
        self._deseneaza("Semnal impuls", deseneaza_semnal, self.rezultat, idx)
        self._deseneaza("Corelare CCSK", deseneaza_corelare, self.rezultat, idx)

    def _urmator(self, cheie):
        if self.rezultat is None:
            return
        pozitii = np.flatnonzero(self.rezultat[cheie])
        if len(pozitii) == 0:
            self.v_stare.set("Nu exista impulsuri bruiate." if cheie == "jam"
                             else "Nu exista simboluri gresite. Cresteti J/S sau scadeti Eb/N0.")
            return
        try:
            curent = int(self.v_idx.get())
        except (tk.TclError, ValueError):
            curent = -1
        urm = pozitii[pozitii > curent]
        self.v_idx.set(int(urm[0] if len(urm) else pozitii[0]))
        self._redeseneaza_impuls()

    def calculeaza_curbe(self):
        p = self._parametri()
        if p is None:
            return
        if not p["bruiate"]:
            messagebox.showinfo("Fara bruiaj", "Alegeti un tip de bruiaj pentru a calcula curbele.")
            return
        self.v_stare.set("Se calculeaza curbele...")
        self.update_idletasks()
        js = np.arange(0, 41, 2.5)
        canal_bruiat = p["bruiate"][0]
        scenarii = {
            f"configuratia curenta ({self.v_tip.get()})": (p["bruiate"], p["salt_activ"], p["canal_fix"]),
            f"fara salt (canal fix {canal_bruiat}, bruiat)": (p["bruiate"], False, canal_bruiat),
            "bruiaj pe banda completa": (list(range(NR_CANALE)), p["salt_activ"], p["canal_fix"]),
        }
        curbe = {}
        for nume, (bruiate, salt, canal) in scenarii.items():
            curbe[nume] = [statistica(bruiate, j, p["putere_totala"], salt, canal, p["zgomot_activ"],
                                      p["ebn0_db"], p["cheie"], nr_sloturi=30, seed=i)[2].mean()
                           for i, j in enumerate(js)]
        self._deseneaza("Curbe SER", deseneaza_curbe, js, curbe, p["ebn0_db"])
        self.v_stare.set("Curbe calculate.")


if __name__ == "__main__":
    Aplicatie().mainloop()
