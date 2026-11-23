import numpy as np
import matplotlib.pyplot as plt


FRECVENTE_MHZ = np.concatenate([
    np.arange(969, 1008 + 1, 3),     # 14 canale
    np.arange(1053, 1065 + 1, 3),    #  5 canale
    np.arange(1113, 1206 + 1, 3),    # 32 canale
]).astype(float)
NR_CANALE = len(FRECVENTE_MHZ)       # 51
FRECVENTA_CENTRALA = (FRECVENTE_MHZ.min() + FRECVENTE_MHZ.max()) / 2

SALT_MINIM_MHZ = 30.0                # distanta minima intre doua salturi
RATA_CHIP = 5e6                      # 5 Mchip/s
CHIPURI_PE_SIMBOL = 32
BITI_PE_SIMBOL = 5
PERIOADA_IMPULS = 13e-6              # un impuls la fiecare 13 us
SPS = 4                              # esantioane pe chip (simulare forma de unda)
CHEIE_RETEA = 2026                   # "cheia" comuna emitator - receptor



def biti_in_simboluri(biti):
    """[1,0,1,1,0, ...] -> simboluri 0..31 (primul bit = cel mai semnificativ)."""
    biti = np.asarray(biti).reshape(-1, BITI_PE_SIMBOL)
    return biti @ np.array([16, 8, 4, 2, 1])


def simboluri_in_biti(simboluri):
    simboluri = np.asarray(simboluri)
    return ((simboluri[:, None] >> np.arange(4, -1, -1)) & 1).ravel()

S0_HEX = 0x7CE90AEC
S0 = np.array([1 - 2 * ((S0_HEX >> (31 - i)) & 1) for i in range(32)])   # bit 0 -> +1, bit 1 -> -1
TABEL_CCSK = np.array([np.roll(S0, -k) for k in range(32)])              # randul k = simbolul k


def ccsk_moduleaza(simboluri):
    """simboluri (N,) -> chipuri (N, 32) cu valori +1 / -1."""
    return TABEL_CCSK[np.asarray(simboluri)]


def ccsk_demoduleaza(chipuri_receptionate):
    """Coreleaza cu toate cele 32 de secvente si alege maximul."""
    corelatii = chipuri_receptionate @ TABEL_CCSK.T        # (N, 32)
    return np.argmax(corelatii, axis=1), corelatii


SEMISINUS = np.sin(np.pi * (np.arange(2 * SPS) + 0.5) / (2 * SPS))


def msk_moduleaza(chipuri):
    """chipuri (N, 32) -> forme de unda complexe (N, 33*SPS)."""
    chipuri = np.atleast_2d(chipuri)
    N = chipuri.shape[0]
    L = (CHIPURI_PE_SIMBOL + 1) * SPS
    I = np.zeros((N, L))
    Q = np.zeros((N, L))
    for k in range(CHIPURI_PE_SIMBOL):
        start = k * SPS
        if k % 2 == 0:
            I[:, start:start + 2 * SPS] += chipuri[:, k:k + 1] * SEMISINUS
        else:
            Q[:, start:start + 2 * SPS] += chipuri[:, k:k + 1] * SEMISINUS
    return I + 1j * Q


def msk_demoduleaza(semnal):
    """Filtru adaptat: corelare cu semi-sinusoida pe fiecare chip -> valori ~ +1/-1."""
    semnal = np.atleast_2d(semnal)
    chipuri = np.zeros((semnal.shape[0], CHIPURI_PE_SIMBOL))
    for k in range(CHIPURI_PE_SIMBOL):
        fereastra = semnal[:, k * SPS:k * SPS + 2 * SPS]
        componenta = fereastra.real if k % 2 == 0 else fereastra.imag
        chipuri[:, k] = componenta @ SEMISINUS / np.sum(SEMISINUS ** 2)
    return chipuri


CANALE_PERMISE = [np.flatnonzero(np.abs(FRECVENTE_MHZ - f) >= SALT_MINIM_MHZ)
                  for f in FRECVENTE_MHZ]


def secventa_salt(nr_slot, nr_impulsuri, cheie=CHEIE_RETEA):
    rng = np.random.default_rng([cheie, nr_slot])
    canale = np.empty(nr_impulsuri, dtype=int)
    canale[0] = rng.integers(NR_CANALE)
    for i in range(1, nr_impulsuri):
        permise = CANALE_PERMISE[canale[i - 1]]
        canale[i] = permise[rng.integers(len(permise))]
    return canale


print("=" * 60)
print(" SAPTAMANA 2 - TESTE")
print("=" * 60)

cor = TABEL_CCSK @ TABEL_CCSK.T
print("CCSK: corelatie cu propria secventa =", cor[0, 0],
      "| maxima cu alta secventa =", np.max(cor[~np.eye(32, dtype=bool)]))


forma = msk_moduleaza(ccsk_moduleaza(np.arange(32)))
anvelopa = np.abs(forma[:, SPS:-SPS])
print(f"MSK: anvelopa intre {anvelopa.min():.3f} si {anvelopa.max():.3f} (trebuie ~1)")

c1 = secventa_salt(0, 2000)
assert np.array_equal(c1, secventa_salt(0, 2000)), "Secventa trebuie sa fie aceeasi"
assert not np.array_equal(c1[:109], secventa_salt(1, 109)), "Sloturi diferite -> secvente diferite"
salt_min = np.min(np.abs(np.diff(FRECVENTE_MHZ[c1])))
print(f"Salt: distanta minima intre salturi = {salt_min:.0f} MHz, "
      f"canale folosite = {len(np.unique(c1))}/51")

rng = np.random.default_rng(1)
biti_tx = rng.integers(0, 2, 5 * 1000)                 # 5000 de biti = 1000 de simboluri
simb_tx = biti_in_simboluri(biti_tx)
canale = secventa_salt(0, len(simb_tx))
semnal = msk_moduleaza(ccsk_moduleaza(simb_tx))
simb_rx, _ = ccsk_demoduleaza(msk_demoduleaza(semnal))
biti_rx = simboluri_in_biti(simb_rx)
ber = np.mean(biti_rx != biti_tx)
print(f"Lant fara zgomot: {len(biti_tx)} biti, erori = {np.sum(biti_rx != biti_tx)}, BER = {ber}")
assert ber == 0, "Fara zgomot BER trebuie sa fie 0!"

print("\nCu zgomot alb (AWGN) adaugat pe forma de unda:")
for snr_db in (-10, -6, -3, 0):
    sigma2 = SPS / 10 ** (snr_db / 10)                 # zgomot raportat la energia unui chip
    zgomot = np.sqrt(sigma2 / 2) * (rng.standard_normal(semnal.shape)
                                    + 1j * rng.standard_normal(semnal.shape))
    s_rx, _ = ccsk_demoduleaza(msk_demoduleaza(semnal + zgomot))
    print(f"  Ec/N0 = {snr_db:4d} dB -> simboluri gresite: {np.mean(s_rx != simb_tx) * 100:5.1f} %")
print("Toate testele au trecut.")

fig1, ax = plt.subplots(figsize=(12, 5))
for slot, culoare in zip(range(3), ["tab:blue", "tab:orange", "tab:green"]):
    c = secventa_salt(slot, 109)
    t = (slot * 109 + np.arange(109)) * PERIOADA_IMPULS * 1e6
    ax.plot(t, FRECVENTE_MHZ[c], "-o", ms=3, lw=0.5, color=culoare, label=f"slot {slot}")
for fiff in (1030, 1090):
    ax.axhspan(fiff - 7, fiff + 7, color="red", alpha=0.15)
ax.set_xlabel("Timp [us]")
ax.set_ylabel("Frecventa [MHz]")
ax.set_title("Salt de frecventa pe 51 de canale - fiecare impuls pe alta frecventa")
ax.legend()
ax.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig("fig_s2_harta_salturi.png", dpi=140)

simbol = int(simb_tx[0])
tc = np.arange(semnal.shape[1]) / (SPS * RATA_CHIP) * 1e6
fig2, axe = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
axe[0].step(np.arange(33) * 0.2, np.append(TABEL_CCSK[simbol], TABEL_CCSK[simbol][-1]),
            where="post", color="black")
axe[0].set_title(f"Secventa CCSK pentru simbolul {simbol} (32 de chipuri, 200 ns fiecare)")
axe[1].plot(tc, semnal[0].real, color="tab:blue")
axe[1].set_title("Componenta I (chipurile pare)")
axe[2].plot(tc, semnal[0].imag, color="tab:orange")
axe[2].set_title("Componenta Q (chipurile impare, decalata cu Tc)")
axe[2].set_xlabel("Timp [us]")
for a in axe:
    a.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("fig_s2_semnal_msk.png", dpi=140)

sigma2 = SPS / 10 ** (-3 / 10)
cu_zgomot = semnal[:1] + np.sqrt(sigma2 / 2) * (rng.standard_normal(semnal[:1].shape)
                                                + 1j * rng.standard_normal(semnal[:1].shape))
fig3, axe = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, s, titlu in ((axe[0], semnal[:1], "fara zgomot"), (axe[1], cu_zgomot, "cu zgomot (Ec/N0 = -3 dB)")):
    decizie, cor = ccsk_demoduleaza(msk_demoduleaza(s))
    culori = ["lightgray"] * 32
    culori[simbol] = "tab:blue"
    ax.bar(np.arange(32), cor[0], color=culori)
    ax.set_title(f"Corelare CCSK {titlu}: trimis {simbol}, decis {decizie[0]}")
    ax.set_xlabel("Simbol candidat (0..31)")
    ax.grid(alpha=0.3)
axe[0].set_ylabel("Corelatie")
fig3.tight_layout()
fig3.savefig("fig_s2_corelare_ccsk.png", dpi=140)

FS = 300e6                                            # esantionare pe toata banda
sps_rf = int(FS / RATA_CHIP)
nr = 20
SPS_vechi = SPS
SPS = sps_rf                                          # refolosim modulatorul la alta rata
SEMISINUS = np.sin(np.pi * (np.arange(2 * SPS) + 0.5) / (2 * SPS))
forme_rf = msk_moduleaza(ccsk_moduleaza(simb_tx[:nr]))
perioada = int(PERIOADA_IMPULS * FS)
rf = np.zeros(perioada * nr, dtype=complex)
for i in range(nr):
    df = (FRECVENTE_MHZ[canale[i]] - FRECVENTA_CENTRALA) * 1e6
    t = (i * perioada + np.arange(forme_rf.shape[1])) / FS
    rf[i * perioada:i * perioada + forme_rf.shape[1]] = forme_rf[i] * np.exp(2j * np.pi * df * t)
rf += 0.05 * (rng.standard_normal(len(rf)) + 1j * rng.standard_normal(len(rf)))
SPS, SEMISINUS = SPS_vechi, np.sin(np.pi * (np.arange(2 * SPS_vechi) + 0.5) / (2 * SPS_vechi))

fig4, ax = plt.subplots(figsize=(12, 5))
ax.specgram(rf, NFFT=512, Fs=FS / 1e6, Fc=FRECVENTA_CENTRALA, noverlap=256, cmap="viridis")
ax.set_xlabel("Timp [us]")
ax.set_ylabel("Frecventa [MHz]")
ax.set_ylim(960, 1215)
ax.set_title("Spectrograma: 20 de impulsuri, fiecare pe alta frecventa")
fig4.tight_layout()
fig4.savefig("fig_s2_spectrograma.png", dpi=140)

print("\nFiguri salvate: fig_s2_harta_salturi.png, fig_s2_semnal_msk.png, "
      "fig_s2_corelare_ccsk.png, fig_s2_spectrograma.png")
plt.show()