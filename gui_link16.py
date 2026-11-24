import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

FRECVENTE_MHZ = np.concatenate([np.arange(969, 1009, 3),     # 14 canale
                                np.arange(1053, 1066, 3),    #  5 canale
                                np.arange(1113, 1207, 3)]    # 32 canale
                               ).astype(float)
NR_CANALE = len(FRECVENTE_MHZ)                               # 51
FRECVENTA_CENTRALA = (FRECVENTE_MHZ.min() + FRECVENTE_MHZ.max()) / 2
SUBBENZI = {"Inferioara (969-1008)": list(range(0, 14)),
            "Mediana (1053-1065)": list(range(14, 19)),
            "Superioara (1113-1206)": list(range(19, 51))}
TIPURI_BRUIAJ = ["Fara bruiaj", "Banda completa", "Banda partiala (rho)", "Subbanda"]

SALT_MINIM_MHZ = 30.0
RATA_CHIP = 5e6
CHIPURI_PE_SIMBOL = 32
BITI_PE_SIMBOL = 5
PERIOADA_IMPULS = 13e-6
IMPULSURI_PE_SLOT = 109
SPS = 4                                                      # esantioane pe chip

MESAJ_IMPLICIT = "Link 16 - mesaj de test. Tinta la azimut 045, distanta 12 km."


def canale_bruiate(tip, rho=0.25, subbanda="Mediana (1053-1065)", seed=7):
    if tip == "Banda completa":
        return list(range(NR_CANALE))
    if tip == "Subbanda":
        return SUBBENZI[subbanda]
    if tip == "Banda partiala (rho)":
        n = max(1, int(round(rho * NR_CANALE)))
        return sorted(np.random.default_rng(seed).choice(NR_CANALE, n, replace=False).tolist())
    return []


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


def transmite_mesaj(text, cheie=2026, salt_activ=True, canal_fix=25,
                    zgomot_activ=False, ebn0_db=10.0, seed=1):
    """Trece mesajul prin tot lantul si intoarce toate datele intermediare."""
    rng = np.random.default_rng(seed)
    biti_tx, nr_biti = text_in_biti(text)
    simb_tx = biti_in_simboluri(biti_tx)
    n = len(simb_tx)
    canale = np.concatenate([
        secventa_salt(s, min(IMPULSURI_PE_SLOT, n - s * IMPULSURI_PE_SLOT), cheie, salt_activ, canal_fix)
        for s in range(int(np.ceil(n / IMPULSURI_PE_SLOT)))])

    tx = msk_moduleaza(ccsk_moduleaza(simb_tx))
    rx = tx.copy()
    # Eb/N0 pe bit -> Ec/N0 pe chip: 5 biti sunt intinsi pe 32 de chipuri
    ecn0_db = ebn0_db + 10 * np.log10(BITI_PE_SIMBOL / CHIPURI_PE_SIMBOL)
    if zgomot_activ:
        sigma2 = SPS / 10 ** (ecn0_db / 10)
        rx = tx + np.sqrt(sigma2 / 2) * (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape))
    simb_rx, corelatii = ccsk_demoduleaza(msk_demoduleaza(rx))
    biti_rx = simboluri_in_biti(simb_rx)
    return {
        "text": text, "text_rx": biti_in_text(biti_rx, nr_biti), "nr_biti": nr_biti,
        "simb_tx": simb_tx, "simb_rx": simb_rx, "canale": canale,
        "tx": tx, "rx": rx, "corelatii": corelatii,
        "erori_biti": int(np.sum(biti_rx[:nr_biti] != biti_tx[:nr_biti])),
        "erori_simboluri": int(np.sum(simb_rx != simb_tx)),
        "zgomot_activ": zgomot_activ, "ebn0_db": ebn0_db, "ecn0_db": ecn0_db,
        "salt_activ": salt_activ,
    }


def deseneaza_plan(fig, r, bruiate, tip_bruiaj):
    """Saptamana 1 + 2: cele 51 de canale, canalele bruiate (rosu), cate impulsuri pe fiecare."""
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
        pe_bruiate = np.isin(r["canale"], list(bruiate)).mean() * 100 if bruiate else 0
        ax2.set_title(f"Impulsuri transmise pe fiecare canal - {pe_bruiate:.0f}% ar cadea pe canale "
                      f"bruiate (bruiajul se aplica in saptamana 3)", fontsize=10)
    ax2.set_xlim(955, 1220)
    ax2.set_xlabel("Frecventa [MHz]")
    ax2.set_ylabel("Nr. impulsuri")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()


def deseneaza_salt(fig, r, idx, bruiate):
    fig.clear()
    ax = fig.add_subplot(111)
    for c in bruiate:
        ax.axhspan(FRECVENTE_MHZ[c] - 1.5, FRECVENTE_MHZ[c] + 1.5, color="#f4c7c7", lw=0)
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
    ax.set_xlabel("Timp [us]  (linia punctata = un nou slot; benzile roz = canale bruiate)")
    ax.set_ylabel("Frecventa [MHz]")
    ax.set_title("Salt de frecventa pe 51 de canale" if r["salt_activ"]
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
    ax.set_title(f"Corelare CCSK - impulsul {idx}: trimis {trimis}, decis {decis} ({stare})")
    ax.set_xlabel("Simbol candidat (0..31)")
    ax.set_ylabel("Corelatie")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def deseneaza_spectrograma(fig, r, nr=20):
    fig.clear()
    ax = fig.add_subplot(111)
    nr = min(nr, len(r["simb_tx"]))
    fs = 300e6
    forme = msk_moduleaza(ccsk_moduleaza(r["simb_tx"][:nr]), int(fs / RATA_CHIP))
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
    ax.set_title(f"Spectrograma primelor {nr} impulsuri - fiecare pe alta frecventa")
    fig.tight_layout()


def text_rezultate(r, bruiate, tip_bruiaj, js_db):
    n = len(r["simb_tx"])
    salt = np.abs(np.diff(FRECVENTE_MHZ[r["canale"]]))
    linii = [
        "REZULTATE",
        "=" * 52,
        f"Caractere in mesaj        : {len(r['text'])}",
        f"Biti transmisi            : {r['nr_biti']}",
        f"Simboluri (impulsuri)     : {n}",
        f"Sloturi TDMA folosite     : {int(np.ceil(n / IMPULSURI_PE_SLOT))}",
        f"Canale distincte folosite : {len(np.unique(r['canale']))} din 51",
        (f"Salt minim intre impulsuri: {salt.min():.0f} MHz" if r["salt_activ"] and len(salt)
         else "Salt de frecventa        : DEZACTIVAT"),
        "Zgomot                    : " + (f"DA, Eb/N0 = {r['ebn0_db']} dB (Ec/N0 = {r['ecn0_db']:.1f} dB)"
                                         if r["zgomot_activ"] else "NU"),
        f"Bruiaj (previzualizare)   : {tip_bruiaj}, {len(bruiate)} canale, J/S = {js_db} dB",
        "",
        f"Simboluri gresite         : {r['erori_simboluri']} din {n}",
        f"Biti gresiti              : {r['erori_biti']} din {r['nr_biti']}",
        f"BER                       : {r['erori_biti'] / max(r['nr_biti'], 1):.2e}",
        "",
        "Mesaj trimis:", r["text"], "",
        "Mesaj receptionat:", r["text_rx"], "",
        "Mesajul a ajuns INTACT." if r["text_rx"] == r["text"] else "Mesajul a ajuns CU ERORI.",
    ]
    return "\n".join(linii)


# ------------------------------------------------------------
# 5. INTERFATA GRAFICA
# ------------------------------------------------------------
class Aplicatie(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulator Link 16")
        self.geometry("1350x800")
        self.rezultat = None

        # ---- variabile ----
        self.v_zgomot = tk.BooleanVar(value=False)
        self.v_ebn0 = tk.DoubleVar(value=10)
        self.v_tip = tk.StringVar(value=TIPURI_BRUIAJ[2])
        self.v_js = tk.DoubleVar(value=20)
        self.v_rho = tk.DoubleVar(value=0.25)
        self.v_sub = tk.StringVar(value=list(SUBBENZI)[1])
        self.v_salt = tk.BooleanVar(value=True)
        self.v_canal = tk.IntVar(value=25)
        self.v_cheie = tk.IntVar(value=2026)
        self.v_rs = tk.BooleanVar(value=True)
        self.v_idx = tk.IntVar(value=0)
        self.v_stare = tk.StringVar(value="Scrieti un mesaj si apasati Transmite.")

        ttk.Label(self, textvariable=self.v_stare, anchor="w").pack(side="bottom", fill="x", padx=6, pady=3)

        # ---- panoul din stanga ----
        stanga = ttk.Frame(self, padding=8)
        stanga.pack(side="left", fill="y")

        g = ttk.LabelFrame(stanga, text="1. Trafic de date", padding=6)
        g.pack(fill="x", pady=3)
        self.txt_mesaj = tk.Text(g, height=4, width=36, wrap="word")
        self.txt_mesaj.insert("1.0", MESAJ_IMPLICIT)
        self.txt_mesaj.pack(fill="x")

        g = ttk.LabelFrame(stanga, text="2. Canal (zgomot termic)", padding=6)
        g.pack(fill="x", pady=3)
        ttk.Checkbutton(g, text="Adauga zgomot AWGN", variable=self.v_zgomot,
                        command=self._stari).pack(anchor="w")
        self.sp_ebn0 = ttk.Spinbox(g, from_=-10, to=20, increment=1, textvariable=self.v_ebn0, width=8)
        self._rand(g, "Eb/N0 [dB]:", self.sp_ebn0)

        g = ttk.LabelFrame(stanga, text="3. Bruiaj AWGN (previzualizare - S3)", padding=6)
        g.pack(fill="x", pady=3)
        self._rand(g, "Tip bruiaj:", ttk.Combobox(g, textvariable=self.v_tip, values=TIPURI_BRUIAJ,
                                                 state="readonly", width=22))
        self._rand(g, "J/S [dB]:", ttk.Spinbox(g, from_=-20, to=70, textvariable=self.v_js, width=8))
        self.sp_rho = ttk.Spinbox(g, from_=0.02, to=1, increment=0.05, textvariable=self.v_rho, width=8)
        self._rand(g, "Fractia rho:", self.sp_rho)
        self.cb_sub = ttk.Combobox(g, textvariable=self.v_sub, values=list(SUBBENZI), state="readonly", width=22)
        self._rand(g, "Subbanda:", self.cb_sub)

        g = ttk.LabelFrame(stanga, text="4. Sistem Link 16", padding=6)
        g.pack(fill="x", pady=3)
        ttk.Checkbutton(g, text="Salt de frecventa (51 canale)", variable=self.v_salt,
                        command=self._stari).pack(anchor="w")
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
        ttk.Button(g, text="Urmatorul simbol gresit ->", command=self._urmatoarea_eroare).pack(fill="x", pady=2)

        ttk.Button(stanga, text="Transmite", command=self.transmite).pack(fill="x", pady=8)

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
        for nume, sapt in (("Pachete", "saptamana 4: pachete, Reed-Solomon"),
                           ("Curbe BER/PER", "saptamana 5: curbe de performanta")):
            f = ttk.Frame(self.taburi)
            ttk.Label(f, text=f"{nume}\n(se completeaza in {sapt})",
                      justify="center").place(relx=0.5, rely=0.5, anchor="center")
            self.taburi.add(f, text=nume)

        # parametrii de bruiaj actualizeaza imediat planul de frecvente
        for v in (self.v_tip, self.v_rho, self.v_sub):
            v.trace_add("write", lambda *_: self._actualizeaza_bruiaj())

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
        bara = NavigationToolbar2Tk(canvas, f, pack_toolbar=False)   # zoom + salvare PNG
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
        self.sp_ebn0.configure(state="normal" if self.v_zgomot.get() else "disabled")
        tip = self.v_tip.get()
        self.sp_rho.configure(state="normal" if tip == "Banda partiala (rho)" else "disabled")
        self.cb_sub.configure(state="readonly" if tip == "Subbanda" else "disabled")

    def _bruiate(self):
        try:
            rho = float(self.v_rho.get())
        except (tk.TclError, ValueError):
            rho = 0.25
        return canale_bruiate(self.v_tip.get(), min(max(rho, 0.0), 1.0), self.v_sub.get())

    def _actualizeaza_bruiaj(self):
        self._stari()
        self._deseneaza("Plan de frecvente", deseneaza_plan, self.rezultat, self._bruiate(), self.v_tip.get())
        self._redeseneaza_impuls()

    # -------------------------------------------------------- actiuni
    def transmite(self):
        text = self.txt_mesaj.get("1.0", "end-1c")
        if not text:
            messagebox.showwarning("Mesaj gol", "Scrieti un mesaj de transmis.")
            return
        try:
            cheie, canal = int(self.v_cheie.get()), int(self.v_canal.get())
            ebn0, js = float(self.v_ebn0.get()), float(self.v_js.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Valoare invalida", "Verificati campurile numerice.")
            return
        if not 0 <= canal < NR_CANALE:
            messagebox.showwarning("Valoare invalida", "Canalul fix trebuie sa fie intre 0 si 50.")
            return

        self.rezultat = r = transmite_mesaj(text, cheie, self.v_salt.get(), canal,
                                            self.v_zgomot.get(), ebn0)
        bruiate = self._bruiate()
        self.sp_idx.configure(to=len(r["simb_tx"]) - 1)
        if self.v_idx.get() >= len(r["simb_tx"]):
            self.v_idx.set(0)
        self._deseneaza("Plan de frecvente", deseneaza_plan, r, bruiate, self.v_tip.get())
        self._deseneaza("Spectrograma", deseneaza_spectrograma, r)
        self._redeseneaza_impuls()
        self.txt_rez.delete("1.0", "end")
        self.txt_rez.insert("1.0", text_rezultate(r, bruiate, self.v_tip.get(), js))
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
        self._deseneaza("Salt de frecventa", deseneaza_salt, self.rezultat, idx, self._bruiate())
        self._deseneaza("Semnal impuls", deseneaza_semnal, self.rezultat, idx)
        self._deseneaza("Corelare CCSK", deseneaza_corelare, self.rezultat, idx)

    def _urmatoarea_eroare(self):
        if self.rezultat is None:
            return
        pozitii = np.flatnonzero(self.rezultat["simb_rx"] != self.rezultat["simb_tx"])
        if len(pozitii) == 0:
            self.v_stare.set("Nu exista simboluri gresite. Activati zgomotul si scadeti Eb/N0 (ex. 0 dB).")
            return
        urm = pozitii[pozitii > self.v_idx.get()]
        self.v_idx.set(int(urm[0] if len(urm) else pozitii[0]))
        self._redeseneaza_impuls()


if __name__ == "__main__":
    Aplicatie().mainloop()