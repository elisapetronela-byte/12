# ============================================================
#  SIMULATOR LINK 16 - SAPTAMANA 2: INTERFATA GRAFICA (tkinter)
#  Generarea semnalelor (CCSK + MSK) si saltul de frecventa
#
#  Ce face: mesajul scris de utilizator este transformat in biti,
#  apoi in simboluri de 5 biti -> CCSK -> MSK -> salt de frecventa
#  -> (optional zgomot) -> receptor -> mesajul receptionat.
# ============================================================
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ------------------------------------------------------------
# 1. PARAMETRI
# ------------------------------------------------------------
FRECVENTE_MHZ = np.concatenate([np.arange(969, 1009, 3),
                                np.arange(1053, 1066, 3),
                                np.arange(1113, 1207, 3)]).astype(float)
NR_CANALE = len(FRECVENTE_MHZ)                         # 51
FRECVENTA_CENTRALA = (FRECVENTE_MHZ.min() + FRECVENTE_MHZ.max()) / 2
SALT_MINIM_MHZ = 30.0
RATA_CHIP = 5e6
CHIPURI_PE_SIMBOL = 32
BITI_PE_SIMBOL = 5
PERIOADA_IMPULS = 13e-6
IMPULSURI_PE_SLOT = 109
SPS = 4                                                # esantioane pe chip

MESAJ_IMPLICIT = "Link 16 - mesaj de test. Tinta la azimut 045, distanta 12 km."


# ------------------------------------------------------------
# 2. FUNCTIILE DIN SAPTAMANA 2
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


S0_HEX = 0x7CE90AEC
S0 = np.array([1 - 2 * ((S0_HEX >> (31 - i)) & 1) for i in range(32)])
TABEL_CCSK = np.array([np.roll(S0, -k) for k in range(32)])


def ccsk_moduleaza(simboluri):
    return TABEL_CCSK[np.asarray(simboluri)]


def ccsk_demoduleaza(chipuri):
    corelatii = chipuri @ TABEL_CCSK.T
    return np.argmax(corelatii, axis=1), corelatii


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


# ------------------------------------------------------------
# 3. SIMULAREA (fara interfata - poate fi testata separat)
# ------------------------------------------------------------
def transmite_mesaj(text, cheie=2026, salt_activ=True, canal_fix=25,
                    zgomot_activ=False, ecn0_db=-3.0, seed=1):
    """Trece mesajul prin tot lantul si intoarce toate datele intermediare."""
    rng = np.random.default_rng(seed)
    biti_tx, nr_biti = text_in_biti(text)
    simb_tx = biti_in_simboluri(biti_tx)
    n = len(simb_tx)

    # saltul de frecventa: cate un slot de 109 impulsuri
    canale = np.concatenate([
        secventa_salt(s, min(IMPULSURI_PE_SLOT, n - s * IMPULSURI_PE_SLOT), cheie, salt_activ, canal_fix)
        for s in range(int(np.ceil(n / IMPULSURI_PE_SLOT)))])

    tx = msk_moduleaza(ccsk_moduleaza(simb_tx))
    rx = tx.copy()
    if zgomot_activ:
        sigma2 = SPS / 10 ** (ecn0_db / 10)
        rx = tx + np.sqrt(sigma2 / 2) * (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape))
    chipuri_rx = msk_demoduleaza(rx)
    simb_rx, corelatii = ccsk_demoduleaza(chipuri_rx)
    biti_rx = simboluri_in_biti(simb_rx)

    return {
        "text": text, "text_rx": biti_in_text(biti_rx, nr_biti),
        "nr_biti": nr_biti, "biti_tx": biti_tx, "biti_rx": biti_rx,
        "simb_tx": simb_tx, "simb_rx": simb_rx, "canale": canale,
        "tx": tx, "rx": rx, "chipuri_rx": chipuri_rx, "corelatii": corelatii,
        "erori_biti": int(np.sum(biti_rx[:nr_biti] != biti_tx[:nr_biti])),
        "erori_simboluri": int(np.sum(simb_rx != simb_tx)),
        "zgomot_activ": zgomot_activ, "ecn0_db": ecn0_db, "salt_activ": salt_activ,
    }


# ------------------------------------------------------------
# 4. FUNCTIILE DE DESENARE (fiecare deseneaza pe o figura)
# ------------------------------------------------------------
def deseneaza_plan(fig, r):
    fig.clear()
    ax = fig.add_subplot(111)
    utilizari = np.bincount(r["canale"], minlength=NR_CANALE)
    ax.bar(FRECVENTE_MHZ, utilizari, width=2.2, color="tab:blue")
    for fiff in (1030, 1090):
        ax.axvspan(fiff - 7, fiff + 7, color="red", alpha=0.2)
        ax.text(fiff, ax.get_ylim()[1] * 0.95, f"IFF {fiff}", ha="center", fontsize=8, color="darkred")
    ax.set_xlim(955, 1220)
    ax.set_xlabel("Frecventa [MHz]")
    ax.set_ylabel("Nr. impulsuri")
    ax.set_title(f"Cate impulsuri au fost transmise pe fiecare din cele 51 de canale "
                 f"({len(np.unique(r['canale']))} canale folosite)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()


def deseneaza_salt(fig, r, idx):
    fig.clear()
    ax = fig.add_subplot(111)
    n = len(r["canale"])
    t = np.arange(n) * PERIOADA_IMPULS * 1e6
    f = FRECVENTE_MHZ[r["canale"]]
    err = r["simb_rx"] != r["simb_tx"]
    ax.plot(t, f, "-o", color="tab:blue", ms=3, lw=0.4, label="impuls")
    if err.any():
        ax.scatter(t[err], f[err], s=70, facecolors="none", edgecolors="red", label="simbol gresit")
    ax.scatter([t[idx]], [f[idx]], s=160, marker="*", color="orange", zorder=5, label=f"impulsul {idx}")
    for k in range(IMPULSURI_PE_SLOT, n, IMPULSURI_PE_SLOT):
        ax.axvline(k * PERIOADA_IMPULS * 1e6, color="gray", ls="--", lw=0.8)
    ax.set_ylim(960, 1215)
    ax.set_xlabel("Timp [us]  (linia punctata = inceputul unui nou slot)")
    ax.set_ylabel("Frecventa [MHz]")
    ax.set_title("Salt de frecventa" if r["salt_activ"] else "FARA salt: toate impulsurile pe acelasi canal")
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
    axe[0].set_title(f"Impulsul {idx}: simbolul {simbol} -> secventa CCSK (32 chipuri)", fontsize=10)
    for ax, comp, nume, cul in ((axe[1], np.real, "I (chipuri pare)", "tab:blue"),
                                (axe[2], np.imag, "Q (chipuri impare, decalat cu Tc)", "tab:orange")):
        if r["zgomot_activ"]:
            ax.plot(tc, comp(r["rx"][idx]), color="#bbbbbb", lw=0.7, label="receptionat")
        ax.plot(tc, comp(r["tx"][idx]), color=cul, lw=1.4, label="transmis")
        ax.set_title("Componenta " + nume, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    axe[2].set_xlabel("Timp [us]")
    for ax in axe:
        ax.grid(alpha=0.3)
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
    ax.set_title(f"Corelare CCSK la receptie - impulsul {idx}: trimis {trimis}, decis {decis} ({stare})")
    ax.set_xlabel("Simbol candidat (0..31)")
    ax.set_ylabel("Corelatie")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def deseneaza_spectrograma(fig, r, nr=20):
    fig.clear()
    ax = fig.add_subplot(111)
    nr = min(nr, len(r["simb_tx"]))
    fs = 300e6
    sps_rf = int(fs / RATA_CHIP)
    forme = msk_moduleaza(ccsk_moduleaza(r["simb_tx"][:nr]), sps_rf)
    perioada = int(PERIOADA_IMPULS * fs)
    rf = np.zeros(perioada * nr, dtype=complex)
    for i in range(nr):
        df = (FRECVENTE_MHZ[r["canale"][i]] - FRECVENTA_CENTRALA) * 1e6
        t = (i * perioada + np.arange(forme.shape[1])) / fs
        rf[i * perioada:i * perioada + forme.shape[1]] = forme[i] * np.exp(2j * np.pi * df * t)
    rng = np.random.default_rng(0)
    rf += 0.05 * (rng.standard_normal(len(rf)) + 1j * rng.standard_normal(len(rf)))
    ax.specgram(rf, NFFT=512, Fs=fs / 1e6, Fc=FRECVENTA_CENTRALA, noverlap=256, cmap="viridis")
    ax.set_ylim(960, 1215)
    ax.set_xlabel("Timp [us]")
    ax.set_ylabel("Frecventa [MHz]")
    ax.set_title(f"Spectrograma primelor {nr} impulsuri")
    fig.tight_layout()


def text_rezultate(r):
    n = len(r["simb_tx"])
    salt = np.abs(np.diff(FRECVENTE_MHZ[r["canale"]]))
    ber = r["erori_biti"] / max(r["nr_biti"], 1)
    linii = [
        "REZULTATE",
        "=" * 50,
        f"Caractere in mesaj        : {len(r['text'])}",
        f"Biti transmisi            : {r['nr_biti']}",
        f"Simboluri (impulsuri)     : {n}",
        f"Sloturi TDMA folosite     : {int(np.ceil(n / IMPULSURI_PE_SLOT))}",
        f"Canale distincte folosite : {len(np.unique(r['canale']))} din 51",
        f"Salt minim intre impulsuri: {salt.min():.0f} MHz" if len(salt) else "Salt minim: -",
        f"Zgomot                    : "
        + (f"DA, Ec/N0 = {r['ecn0_db']} dB" if r["zgomot_activ"] else "NU"),
        "",
        f"Simboluri gresite         : {r['erori_simboluri']} din {n}",
        f"Biti gresiti              : {r['erori_biti']} din {r['nr_biti']}",
        f"BER                       : {ber:.2e}",
        "",
        "Mesaj trimis:",
        r["text"],
        "",
        "Mesaj receptionat:",
        r["text_rx"],
        "",
        "Mesajul a ajuns INTACT." if r["text_rx"] == r["text"] else "Mesajul a ajuns CU ERORI.",
    ]
    return "\n".join(linii)


# ------------------------------------------------------------
# 5. INTERFATA GRAFICA
# ------------------------------------------------------------
class Aplicatie(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulator Link 16 - Saptamana 2: semnale CCSK/MSK si salt de frecventa")
        self.geometry("1300x780")
        self.rezultat = None

        # ---- variabile ----
        self.v_cheie = tk.IntVar(value=2026)
        self.v_salt = tk.BooleanVar(value=True)
        self.v_canal = tk.IntVar(value=25)
        self.v_zgomot = tk.BooleanVar(value=False)
        self.v_ecn0 = tk.DoubleVar(value=-3.0)
        self.v_idx = tk.IntVar(value=0)
        self.v_stare = tk.StringVar(value="Scrieti un mesaj si apasati Transmite.")

        # ---- bara de stare (jos) ----
        ttk.Label(self, textvariable=self.v_stare, anchor="w").pack(side="bottom", fill="x", padx=6, pady=3)

        # ---- panoul din stanga ----
        stanga = ttk.Frame(self, padding=8)
        stanga.pack(side="left", fill="y")

        g = ttk.LabelFrame(stanga, text="1. Mesaj de transmis", padding=6)
        g.pack(fill="x", pady=4)
        self.txt_mesaj = tk.Text(g, height=5, width=34, wrap="word")
        self.txt_mesaj.insert("1.0", MESAJ_IMPLICIT)
        self.txt_mesaj.pack(fill="x")

        g = ttk.LabelFrame(stanga, text="2. Salt de frecventa", padding=6)
        g.pack(fill="x", pady=4)
        ttk.Checkbutton(g, text="Salt activ (51 de canale)", variable=self.v_salt,
                        command=self._stari).pack(anchor="w")
        self._rand(g, "Cheie retea:", ttk.Spinbox(g, from_=0, to=999999, textvariable=self.v_cheie, width=9))
        self.sp_canal = ttk.Spinbox(g, from_=0, to=50, textvariable=self.v_canal, width=9)
        self._rand(g, "Canal fix:", self.sp_canal)

        g = ttk.LabelFrame(stanga, text="3. Zgomot (AWGN)", padding=6)
        g.pack(fill="x", pady=4)
        ttk.Checkbutton(g, text="Adauga zgomot", variable=self.v_zgomot,
                        command=self._stari).pack(anchor="w")
        self.sp_ecn0 = ttk.Spinbox(g, from_=-15, to=20, increment=1, textvariable=self.v_ecn0, width=9)
        self._rand(g, "Ec/N0 [dB]:", self.sp_ecn0)

        g = ttk.LabelFrame(stanga, text="4. Impulsul afisat", padding=6)
        g.pack(fill="x", pady=4)
        self.sp_idx = ttk.Spinbox(g, from_=0, to=0, textvariable=self.v_idx, width=9,
                                  command=self._redeseneaza_impuls)
        self._rand(g, "Nr. impuls:", self.sp_idx)
        ttk.Button(g, text="Urmatorul simbol gresit ->", command=self._urmatoarea_eroare).pack(fill="x", pady=2)

        ttk.Button(stanga, text="Transmite", command=self.transmite).pack(fill="x", pady=10)

        # ---- tab-urile din dreapta ----
        self.taburi = ttk.Notebook(self)
        self.taburi.pack(side="right", fill="both", expand=True)
        self.figuri = {}
        for nume in ("Plan de frecvente", "Salt de frecventa", "Semnal impuls",
                     "Corelare CCSK", "Spectrograma"):
            self.figuri[nume] = self._tab_grafic(nume)
        f = ttk.Frame(self.taburi)
        self.txt_rez = tk.Text(f, wrap="word", font=("Courier New", 10))
        self.txt_rez.pack(fill="both", expand=True)
        self.taburi.add(f, text="Rezultate")

        self._stari()
        self.transmite()

    # -------------------------------------------------------- utilitare
    @staticmethod
    def _rand(parinte, text, widget):
        r = ttk.Frame(parinte)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text=text, width=12).pack(side="left")
        widget.pack(side="left")

    def _tab_grafic(self, nume):
        f = ttk.Frame(self.taburi)
        fig = Figure(figsize=(9, 6))
        canvas = FigureCanvasTkAgg(fig, master=f)
        bara = NavigationToolbar2Tk(canvas, f, pack_toolbar=False)    # zoom + salvare PNG
        bara.update()
        bara.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.taburi.add(f, text=nume)
        return fig, canvas

    def _deseneaza(self, nume, functie, *args):
        fig, canvas = self.figuri[nume]
        functie(fig, *args)
        canvas.draw_idle()

    def _stari(self):
        self.sp_canal.configure(state="disabled" if self.v_salt.get() else "normal")
        self.sp_ecn0.configure(state="normal" if self.v_zgomot.get() else "disabled")

    # -------------------------------------------------------- actiuni
    def transmite(self):
        text = self.txt_mesaj.get("1.0", "end-1c")
        if not text:
            messagebox.showwarning("Mesaj gol", "Scrieti un mesaj de transmis.")
            return
        try:
            cheie, canal = int(self.v_cheie.get()), int(self.v_canal.get())
            ecn0 = float(self.v_ecn0.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Valoare invalida", "Verificati campurile numerice.")
            return
        if not 0 <= canal < NR_CANALE:
            messagebox.showwarning("Valoare invalida", "Canalul fix trebuie sa fie intre 0 si 50.")
            return

        self.rezultat = r = transmite_mesaj(text, cheie, self.v_salt.get(), canal,
                                            self.v_zgomot.get(), ecn0)
        self.sp_idx.configure(to=len(r["simb_tx"]) - 1)
        if self.v_idx.get() >= len(r["simb_tx"]):
            self.v_idx.set(0)
        self._deseneaza("Plan de frecvente", deseneaza_plan, r)
        self._deseneaza("Spectrograma", deseneaza_spectrograma, r)
        self._redeseneaza_impuls()
        self.txt_rez.delete("1.0", "end")
        self.txt_rez.insert("1.0", text_rezultate(r))
        self.v_stare.set(f"Transmis: {r['nr_biti']} biti, {len(r['simb_tx'])} impulsuri, "
                         f"{r['erori_biti']} biti gresiti. "
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

    def _urmatoarea_eroare(self):
        if self.rezultat is None:
            return
        pozitii = np.flatnonzero(self.rezultat["simb_rx"] != self.rezultat["simb_tx"])
        if len(pozitii) == 0:
            self.v_stare.set("Nu exista simboluri gresite. Activati zgomotul si scadeti Ec/N0.")
            return
        urm = pozitii[pozitii > self.v_idx.get()]
        self.v_idx.set(int(urm[0] if len(urm) else pozitii[0]))
        self._redeseneaza_impuls()


if __name__ == "__main__":
    Aplicatie().mainloop()