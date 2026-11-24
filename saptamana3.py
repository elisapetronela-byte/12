import re
import numpy as np
import matplotlib.pyplot as plt

FRECVENTE_MHZ = np.concatenate([
    np.arange(969, 1008 + 1, 3),     # 14 canale
    np.arange(1053, 1065 + 1, 3),    #  5 canale
    np.arange(1113, 1206 + 1, 3),    # 32 canale
]).astype(float)
NR_CANALE = len(FRECVENTE_MHZ)       # 51
FRECVENTA_CENTRALA = (FRECVENTE_MHZ.min() + FRECVENTE_MHZ.max()) / 2
SUBBENZI = {
    "inferioara": list(range(0, 14)),     # 969-1008 MHz
    "mediana": list(range(14, 19)),       # 1053-1065 MHz
    "superioara": list(range(19, 51)),    # 1113-1206 MHz
}
SALT_MINIM_MHZ = 30.0
RATA_CHIP = 5e6
CHIPURI_PE_SIMBOL = 32
BITI_PE_SIMBOL = 5
PERIOADA_IMPULS = 13e-6
SPS = 4                              # esantioane pe chip
CHEIE_RETEA = 2026
IMPULSURI_PE_SLOT = 109

EBN0_DB = 10.0                       # zgomot termic (raport semnal / zgomot pe bit)


def biti_in_simboluri(biti):
    return np.asarray(biti).reshape(-1, BITI_PE_SIMBOL) @ np.array([16, 8, 4, 2, 1])


def simboluri_in_biti(simboluri):
    return ((np.asarray(simboluri)[:, None] >> np.arange(4, -1, -1)) & 1).ravel()


S0_HEX = 0x7CE90AEC
S0 = np.array([1 - 2 * ((S0_HEX >> (31 - i)) & 1) for i in range(32)])
TABEL_CCSK = np.array([np.roll(S0, -k) for k in range(32)])


def ccsk_moduleaza(simboluri):
    return TABEL_CCSK[np.asarray(simboluri)]


def ccsk_demoduleaza(chipuri_receptionate):
    corelatii = chipuri_receptionate @ TABEL_CCSK.T
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


def secventa_salt(nr_slot, nr_impulsuri, cheie=CHEIE_RETEA, salt_activ=True, canal_fix=25):
    """salt_activ=False -> toate impulsurile pe canal_fix (pentru comparatie)."""
    if not salt_activ:
        return np.full(nr_impulsuri, canal_fix)
    rng = np.random.default_rng([cheie, nr_slot])
    canale = np.empty(nr_impulsuri, dtype=int)
    canale[0] = rng.integers(NR_CANALE)
    for i in range(1, nr_impulsuri):
        permise = CANALE_PERMISE[canale[i - 1]]
        canale[i] = permise[rng.integers(len(permise))]
    return canale


def parseaza_canale(text):
    """'0-9, 30, 40-45' -> [0..9, 30, 40..45];  '969-1008' (MHz) -> indicii canalelor."""
    canale = set()
    for parte in re.split(r"[,; ]+", text.strip()):
        if not parte:
            continue
        a, _, b = parte.partition("-")
        a, b = float(a), float(b or a)
        if max(a, b) > 100:                                  # valori in MHz
            idx = np.flatnonzero((FRECVENTE_MHZ >= min(a, b)) & (FRECVENTE_MHZ <= max(a, b)))
            canale.update(idx.tolist())
        else:
            canale.update(c for c in range(int(min(a, b)), int(max(a, b)) + 1) if c < NR_CANALE)
    return sorted(canale)


def canale_bruiate(mod, rho=0.25, subbanda="mediana", canale_text="", seed=7):
    """
    Intoarce lista canalelor bruiate pentru fiecare tip de bruiaj:
      'fara'           - niciun canal
      'banda_completa' - toate cele 51 de canale (barrage)
      'banda_partiala' - o fractie rho din canale, alese aleator
      'subbanda'       - una din cele 3 subbenzi ale planului de frecvente
      'personalizat'   - canale scrise de utilizator, ex. '10-20, 35'
    """
    if mod == "fara":
        return []
    if mod == "banda_completa":
        return list(range(NR_CANALE))
    if mod == "banda_partiala":
        n = max(1, int(round(rho * NR_CANALE)))
        return sorted(np.random.default_rng(seed).choice(NR_CANALE, n, replace=False).tolist())
    if mod == "subbanda":
        return SUBBENZI[subbanda]
    if mod == "personalizat":
        return parseaza_canale(canale_text)
    raise ValueError("Mod de bruiaj necunoscut: " + mod)


def js_pe_canal(js_db, nr_bruiate, putere_totala_constanta=True):
    """
    J/S = puterea bruiajului / puterea semnalului.
    Cu putere TOTALA constanta (realist), bruiajul isi imparte puterea pe
    canalele bruiate: cu cat bruiaza mai putine canale, cu atat fiecare
    este bruiat mai puternic.
    """
    if nr_bruiate == 0:
        return 0.0
    js = 10 ** (js_db / 10)
    return js / nr_bruiate if putere_totala_constanta else js


def ec_n0(ebn0_db):
    """Energia pe chip / N0: un simbol de 5 biti este intins pe 32 de chipuri."""
    return 10 ** (ebn0_db / 10) * BITI_PE_SIMBOL / CHIPURI_PE_SIMBOL


def transmite(simboluri, canale, bruiate, js_db, ebn0_db=EBN0_DB, forma_de_unda=False,
              rng=None, putere_totala_constanta=True):
    """
    Trece simbolurile prin lantul complet. Pentru fiecare impuls zgomotul este
        N0           daca impulsul cade pe un canal liber
        N0 + NJ      daca impulsul cade pe un canal bruiat   (NJ = J/S pe canal)
    forma_de_unda=True  -> se genereaza efectiv semnalul MSK (lent, exact)
    forma_de_unda=False -> nivel chip, echivalent matematic (rapid, pentru curbe)
    """
    rng = rng or np.random.default_rng(0)
    masca = np.zeros(NR_CANALE, dtype=bool)
    masca[list(bruiate)] = True
    pe_canal_bruiat = masca[canale]
    nj = js_pe_canal(js_db, masca.sum(), putere_totala_constanta)
    n_total = 1 / ec_n0(ebn0_db) + pe_canal_bruiat * nj          # zgomot pe fiecare impuls

    chipuri = ccsk_moduleaza(simboluri)
    if forma_de_unda:
        tx = msk_moduleaza(chipuri)
        sigma = np.sqrt(SPS * n_total / 2)[:, None]
        rx = tx + sigma * (rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape))
        soft = msk_demoduleaza(rx)
    else:
        soft = chipuri + np.sqrt(n_total / 2)[:, None] * rng.standard_normal(chipuri.shape)
    simb_rx, corelatii = ccsk_demoduleaza(soft)
    return simb_rx, pe_canal_bruiat, corelatii


def simuleaza(mod, js_db, rho=0.25, subbanda="mediana", canale_text="", nr_sloturi=20,
              salt_activ=True, canal_fix=None, forma_de_unda=False, ebn0_db=EBN0_DB, seed=1):
    """Transmite nr_sloturi x 109 simboluri aleatoare si intoarce statisticile."""
    rng = np.random.default_rng(seed)
    bruiate = canale_bruiate(mod, rho, subbanda, canale_text)
    if canal_fix is None:
        canal_fix = bruiate[0] if bruiate else 25        # fara salt: cazul cel mai rau
    simb_tx, simb_rx, jam, can = [], [], [], []
    for slot in range(nr_sloturi):
        s = rng.integers(0, 32, IMPULSURI_PE_SLOT)
        c = secventa_salt(slot, IMPULSURI_PE_SLOT, salt_activ=salt_activ, canal_fix=canal_fix)
        r, j, _ = transmite(s, c, bruiate, js_db, ebn0_db, forma_de_unda, rng)
        simb_tx.append(s); simb_rx.append(r); jam.append(j); can.append(c)
    simb_tx, simb_rx = np.concatenate(simb_tx), np.concatenate(simb_rx)
    jam, can = np.concatenate(jam), np.concatenate(can)
    err = simb_rx != simb_tx
    ber = np.mean(simboluri_in_biti(simb_rx) != simboluri_in_biti(simb_tx))
    return {
        "bruiate": bruiate, "canale": can, "jam": jam, "err": err,
        "ser": err.mean(), "ber": ber,
        "impulsuri_bruiate": jam.mean(),
        "ser_bruiat": err[jam].mean() if jam.any() else 0.0,
        "ser_liber": err[~jam].mean() if (~jam).any() else 0.0,
    }


print("=" * 72)
print(" SAPTAMANA 3 - TESTE")
print("=" * 72)


assert canale_bruiate("fara") == []
assert len(canale_bruiate("banda_completa")) == 51
assert len(canale_bruiate("banda_partiala", rho=0.2)) == 10
assert canale_bruiate("subbanda", subbanda="mediana") == [14, 15, 16, 17, 18]
assert canale_bruiate("personalizat", canale_text="0-3, 10") == [0, 1, 2, 3, 10]
assert canale_bruiate("personalizat", canale_text="969-1008") == list(range(14))
print("Moduri de bruiaj: OK (fara / banda completa / partiala / subbanda / personalizat)")

r = simuleaza("fara", 0, ebn0_db=60, nr_sloturi=5, forma_de_unda=True)
print(f"Fara zgomot, fara bruiaj: SER = {r['ser']}")
assert r["ser"] == 0

a = simuleaza("banda_partiala", 25, rho=0.25, nr_sloturi=30, forma_de_unda=True)
b = simuleaza("banda_partiala", 25, rho=0.25, nr_sloturi=30, forma_de_unda=False)
print(f"Verificare: SER forma de unda = {a['ser']:.3f}, SER nivel chip = {b['ser']:.3f} (trebuie apropiate)")
assert abs(a["ser"] - b["ser"]) < 0.03

JS_DEMO = 25
print(f"\nComparatie la J/S total = {JS_DEMO} dB, Eb/N0 = {EBN0_DB} dB, cu salt de frecventa:")
print(f"{'Mod bruiaj':28s} {'canale':>6s} {'J/S/canal':>10s} {'imp.bruiate':>12s} "
      f"{'SER bruiat':>11s} {'SER liber':>10s} {'SER total':>10s}")
for nume, mod, extra in (("Fara bruiaj", "fara", {}),
                         ("Banda completa", "banda_completa", {}),
                         ("Banda partiala rho=0.25", "banda_partiala", {"rho": 0.25}),
                         ("Subbanda mediana", "subbanda", {"subbanda": "mediana"}),
                         ("Personalizat 10-20, 35", "personalizat", {"canale_text": "10-20, 35"})):
    r = simuleaza(mod, JS_DEMO, nr_sloturi=40, **extra)
    n = len(r["bruiate"])
    jsc = f"{10 * np.log10(js_pe_canal(JS_DEMO, n)):.1f} dB" if n else "-"
    print(f"{nume:28s} {n:6d} {jsc:>10s} {100 * r['impulsuri_bruiate']:11.1f}% "
          f"{100 * r['ser_bruiat']:10.1f}% {100 * r['ser_liber']:9.1f}% {100 * r['ser']:9.1f}%")

cu = simuleaza("banda_partiala", 25, rho=0.1, nr_sloturi=40)
fara = simuleaza("banda_partiala", 25, rho=0.1, nr_sloturi=40, salt_activ=False)
print(f"\nBruiaj pe 10% din banda, J/S = 25 dB:")
print(f"  cu salt de frecventa : SER = {100 * cu['ser']:5.1f} %  (doar impulsurile de pe canalele bruiate)")
print(f"  fara salt (canal fix bruiat): SER = {100 * fara['ser']:5.1f} %")
print("Toate testele au trecut.")

r = simuleaza("banda_partiala", JS_DEMO, rho=0.25, nr_sloturi=3)
n = len(r["canale"])
t = np.arange(n) * PERIOADA_IMPULS * 1e6
f = FRECVENTE_MHZ[r["canale"]]
fig1, ax = plt.subplots(figsize=(12, 5.5))
for c in r["bruiate"]:
    ax.axhspan(FRECVENTE_MHZ[c] - 1.5, FRECVENTE_MHZ[c] + 1.5, color="#f4c7c7", lw=0)
ax.plot(t, f, color="#bbbbbb", lw=0.5)
ax.scatter(t[~r["jam"]], f[~r["jam"]], s=12, color="tab:blue", label="impuls pe canal liber")
ax.scatter(t[r["jam"]], f[r["jam"]], s=20, color="tab:red", label="impuls pe canal bruiat")
ax.scatter(t[r["err"]], f[r["err"]], s=70, facecolors="none", edgecolors="black",
           label="simbol gresit")
ax.set_xlabel("Timp [us]")
ax.set_ylabel("Frecventa [MHz]")
ax.set_title(f"Salt de frecventa sub bruiaj pe 25% din banda (benzile rosii), J/S = {JS_DEMO} dB")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig("fig_s3_harta_bruiaj.png", dpi=140)

rng = np.random.default_rng(3)
FS = 300e6
sps_rf = int(FS / RATA_CHIP)
nr = 24
bruiate_demo = canale_bruiate("personalizat", canale_text="14-18, 35-40")
canale_demo = secventa_salt(0, nr)
forme = msk_moduleaza(ccsk_moduleaza(rng.integers(0, 32, nr)), sps_rf)
perioada = int(PERIOADA_IMPULS * FS)
N = perioada * nr
rf = np.zeros(N, dtype=complex)
for i in range(nr):
    df = (FRECVENTE_MHZ[canale_demo[i]] - FRECVENTA_CENTRALA) * 1e6
    tt = (i * perioada + np.arange(forme.shape[1])) / FS
    rf[i * perioada:i * perioada + forme.shape[1]] = forme[i] * np.exp(2j * np.pi * df * tt)
Z = np.fft.fft(rng.standard_normal(N) + 1j * rng.standard_normal(N))     # zgomot de bruiaj
frecv = np.fft.fftfreq(N, 1 / FS)
masca_f = np.zeros(N, dtype=bool)
for c in bruiate_demo:
    masca_f |= np.abs(frecv - (FRECVENTE_MHZ[c] - FRECVENTA_CENTRALA) * 1e6) <= 1.5e6
Z[~masca_f] = 0
jam = np.fft.ifft(Z)
rf += jam / np.sqrt(np.mean(np.abs(jam) ** 2)) * 0.6
rf += 0.03 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
fig2, ax = plt.subplots(figsize=(12, 5))
ax.specgram(rf, NFFT=512, Fs=FS / 1e6, Fc=FRECVENTA_CENTRALA, noverlap=256, cmap="viridis")
ax.set_xlabel("Timp [us]")
ax.set_ylabel("Frecventa [MHz]")
ax.set_ylim(960, 1215)
ax.set_title("Spectrograma: impulsuri cu salt + bruiaj AWGN continuu pe canalele 14-18 si 35-40")
fig2.tight_layout()
fig2.savefig("fig_s3_spectrograma_bruiaj.png", dpi=140)

r = simuleaza("banda_partiala", JS_DEMO, rho=0.25, nr_sloturi=60)
utilizari = np.bincount(r["canale"], minlength=NR_CANALE)
erori = np.bincount(r["canale"], weights=r["err"], minlength=NR_CANALE)
rata = np.divide(erori, utilizari, out=np.zeros(NR_CANALE), where=utilizari > 0)
culori = ["tab:red" if c in r["bruiate"] else "tab:blue" for c in range(NR_CANALE)]
fig3, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(NR_CANALE), 100 * rata, color=culori)
ax.set_xlabel("Indice canal (0 = 969 MHz ... 50 = 1206 MHz)")
ax.set_ylabel("Simboluri gresite [%]")
ax.set_title("Erorile apar doar pe canalele bruiate (rosu); canalele libere (albastru) raman curate")
ax.grid(alpha=0.3, axis="y")
fig3.tight_layout()
fig3.savefig("fig_s3_erori_pe_canal.png", dpi=140)

print("\nSe calculeaza curbele (cateva secunde)...")
js = np.arange(0, 41, 2.5)
fig4, ax = plt.subplots(figsize=(10, 5.5))
scenarii = [("banda partiala rho = 0.1", dict(mod="banda_partiala", rho=0.1)),
            ("banda partiala rho = 0.25", dict(mod="banda_partiala", rho=0.25)),
            ("banda partiala rho = 0.5", dict(mod="banda_partiala", rho=0.5)),
            ("banda completa", dict(mod="banda_completa")),
            ("FARA salt, rho = 0.1 (canal bruiat)", dict(mod="banda_partiala", rho=0.1, salt_activ=False))]
for (nume, p), mk in zip(scenarii, "osd^x"):
    ser = [simuleaza(js_db=j, nr_sloturi=40, seed=i, **p)["ser"] for i, j in enumerate(js)]
    ax.semilogy(js, np.maximum(ser, 1e-4), marker=mk, label=nume)
ax.axhline(8 / 31, color="gray", ls="--")
ax.text(1, 8 / 31 * 1.15, "limita de corectie Reed-Solomon (8 din 31) - saptamana 4", fontsize=8, color="gray")
ax.set_xlabel("J/S total [dB]")
ax.set_ylabel("Rata de eroare a simbolurilor (SER)")
ax.set_ylim(1e-4, 1.5)
ax.set_title(f"SER in functie de puterea bruiajului (Eb/N0 = {EBN0_DB} dB, fara codare)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig4.tight_layout()
fig4.savefig("fig_s3_curbe_ser.png", dpi=140)

print("Figuri salvate: fig_s3_harta_bruiaj.png, fig_s3_spectrograma_bruiaj.png, "
      "fig_s3_erori_pe_canal.png, fig_s3_curbe_ser.png")
plt.show()