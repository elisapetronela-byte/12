import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch


FRECVENTE_MHZ = np.concatenate([
    np.arange(969, 1008 + 1, 3),     
    np.arange(1053, 1065 + 1, 3),    
    np.arange(1113, 1206 + 1, 3),    
]).astype(float)
NR_CANALE = len(FRECVENTE_MHZ)

SUBBENZI = {
    "Subbanda inferioara (969-1008 MHz)": list(range(0, 14)),
    "Subbanda mediana (1053-1065 MHz)": list(range(14, 19)),
    "Subbanda superioara (1113-1206 MHz)": list(range(19, 51)),
}
SALT_MINIM_MHZ = 30.0            

RATA_CHIP = 5e6                  
T_CHIP = 1 / RATA_CHIP          
CHIPURI_PE_SIMBOL = 32           
BITI_PE_SIMBOL = 5
DURATA_IMPULS = CHIPURI_PE_SIMBOL * T_CHIP   
PERIOADA_IMPULS = 13e-6          
DURATA_SLOT = 7.8125e-3          

BITI_DATE_CUVANT = 70            
RS_DATE_N, RS_DATE_K = 31, 15    
RS_ANTET_N, RS_ANTET_K = 16, 7   
CUVINTE_PE_PACHET = 3
IMPULSURI_PE_PACHET = RS_ANTET_N + CUVINTE_PE_PACHET * RS_DATE_N   
OCTETI_PE_PACHET = CUVINTE_PE_PACHET * BITI_DATE_CUVANT // 8       

def subbanda(i):
    for nume, canale in SUBBENZI.items():
        if i in canale:
            return nume

print("Canal  Frecventa [MHz]  Subbanda")
for i, f in enumerate(FRECVENTE_MHZ):
    print(f"{i:5d}  {f:15.0f}  {subbanda(i)}")
print("Total canale:", NR_CANALE)

print("\nParametri:")
print("  Durata impuls       :", round(DURATA_IMPULS * 1e6, 1), "us")
print("  Salturi pe secunda  :", round(1 / PERIOADA_IMPULS))
print("  Sloturi in 12 s     :", round(12 / DURATA_SLOT))
print("  Impulsuri / pachet  :", IMPULSURI_PE_PACHET)
print("  Octeti utili/pachet :", OCTETI_PE_PACHET)
print("  Castig CCSK         :", round(10 * np.log10(CHIPURI_PE_SIMBOL / BITI_PE_SIMBOL), 1), "dB")
print("  Castig salt (51 can):", round(10 * np.log10(NR_CANALE), 1), "dB")


assert NR_CANALE == 51
assert not np.any(np.abs(FRECVENTE_MHZ - 1030) < 20)
assert not np.any(np.abs(FRECVENTE_MHZ - 1090) < 20)
assert IMPULSURI_PE_PACHET * PERIOADA_IMPULS < DURATA_SLOT
permise = np.array([np.sum(np.abs(FRECVENTE_MHZ - f) >= SALT_MINIM_MHZ)
                    for f in FRECVENTE_MHZ])
print("\nVerificari OK. Canale permise dupa un salt: minim",
      permise.min(), ", maxim", permise.max())


culori = ["tab:blue", "tab:orange", "tab:green"]
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for (nume, canale), c in zip(SUBBENZI.items(), culori):
    ax1.vlines(FRECVENTE_MHZ[canale], 0, 1, color=c, lw=2,
               label=f"{nume} - {len(canale)} canale")
for fiff in (1030, 1090):
    ax1.axvspan(fiff - 7, fiff + 7, color="red", alpha=0.25)
    ax1.text(fiff, 1.05, f"IFF {fiff}", ha="center", fontsize=8, color="darkred")
ax1.set_ylim(0, 1.4)
ax1.set_yticks([])
ax1.legend(fontsize=8)
ax1.set_title("Planul de frecvente Link 16: 51 de canale, pas 3 MHz")
culoare_canal = [culori[0] if i < 14 else culori[1] if i < 19 else culori[2]
                 for i in range(NR_CANALE)]
ax2.bar(FRECVENTE_MHZ, permise, width=2.2, color=culoare_canal)
ax2.set_ylabel("Canale permise\nla saltul urmator")
ax2.set_xlabel("Frecventa [MHz]")
ax2.set_title("Regula de salt: minim 30 MHz intre impulsuri consecutive")
fig1.tight_layout()
fig1.savefig("fig_plan_frecvente.png", dpi=140)


fig2, axe = plt.subplots(3, 1, figsize=(12, 6))
for k in range(6):                                   
    axe[0].add_patch(Rectangle((k * 13, 0), 6.4, 1, color="tab:blue"))
    axe[0].text(k * 13 + 3.2, 0.5, f"f{k + 1}", color="white", ha="center", va="center")
axe[0].set_xlim(-1, 80)
axe[0].set_ylim(-0.3, 1.3)
axe[0].set_yticks([])
axe[0].set_xlabel("Timp [us]")
axe[0].set_title("(a) Impulsuri de 6,4 us la fiecare 13 us, fiecare pe alta frecventa")

durata = IMPULSURI_PE_PACHET * PERIOADA_IMPULS * 1e3  
axe[1].add_patch(Rectangle((0, 0), 0.4, 1, color="gray"))
axe[1].add_patch(Rectangle((0.4, 0), durata, 1, color="tab:blue"))
axe[1].text(0.4 + durata / 2, 0.5, "109 impulsuri", color="white", ha="center", va="center")
axe[1].text(5, 0.5, "garda / propagare", ha="center", va="center")
axe[1].set_xlim(0, DURATA_SLOT * 1e3)
axe[1].set_yticks([])
axe[1].set_xlabel("Timp [ms]")
axe[1].set_title("(b) Slot TDMA de 7,8125 ms")

for s in range(48):                                  
    axe[2].add_patch(Rectangle((s, 0), 0.9, 1, color=f"C{s % 4}"))
axe[2].text(49, 0.5, "... 1536 sloturi = 12 s", va="center")
axe[2].set_xlim(0, 64)
axe[2].set_yticks([])
axe[2].set_xticks([])
axe[2].set_title("(c) Cadrul de 12 s, sloturile impartite intre terminale")
fig2.tight_layout()
fig2.savefig("fig_structura_timp.png", dpi=140)


emitator = ["Mesaj text\n-> pachete", "Antet + cuvinte J\n+ CRC-5", "Reed-Solomon",
            "Intretesere", "CCSK\n5 biti -> 32 chipuri", "MSK\n5 Mchip/s"]
canal = ["Salt de frecventa\n51 canale", "+ zgomot termic\n(Eb/N0)", "+ bruiaj AWGN\npe subbenzi (J/S)"]
receptor = ["Filtru adaptat\nMSK", "Corelare CCSK", "De-intretesere",
            "Decodare RS", "Verificare CRC", "Mesaj\nreasamblat"]

def bloc(ax, x, y, text, culoare):
    ax.add_patch(FancyBboxPatch((x - 0.85, y - 0.45), 1.7, 0.9,
                                boxstyle="round,pad=0.03", fc=culoare, ec="black"))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5)

def sageata(ax, x1, y1, x2, y2, stil="arc3"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", connectionstyle=stil))

fig3, ax = plt.subplots(figsize=(15, 6.5))
pas = 2.3
xs = [1 + i * pas for i in range(6)]
xc = [xs[-1] - i * pas * 1.25 for i in range(3)]
for x, t in zip(xs, emitator):
    bloc(ax, x, 5, t, "#d6e6f5")
for x, t in zip(xc, canal):
    bloc(ax, x, 3, t, "#f8d7d7")
for x, t in zip(xs, receptor):
    bloc(ax, x, 1, t, "#dcefdc")
for a, b in zip(xs, xs[1:]):
    sageata(ax, a + 0.85, 5, b - 0.85, 5)
    sageata(ax, a + 0.85, 1, b - 0.85, 1)
sageata(ax, xs[-1], 4.55, xc[0], 3.45)
for a, b in zip(xc, xc[1:]):
    sageata(ax, a - 0.85, 3, b + 0.85, 3)
sageata(ax, xc[-1] - 0.85, 3, xs[0], 1.45, "angle,angleA=180,angleB=90,rad=0")
for y, nume in ((5, "EMITATOR"), (3, "CANAL"), (1, "RECEPTOR")):
    ax.text(-0.6, y, nume, rotation=90, ha="center", va="center", weight="bold")
ax.set_xlim(-1, xs[-1] + 1)
ax.set_ylim(0.3, 5.7)
ax.axis("off")
ax.set_title("Schema-bloc a simulatorului de bruiaj Link 16")
fig3.tight_layout()
fig3.savefig("fig_schema_bloc.png", dpi=140)

print("\nFigurile au fost salvate: fig_plan_frecvente.png, "
      "fig_structura_timp.png, fig_schema_bloc.png")
plt.show()
