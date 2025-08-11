# RGB-cubes-detection

*Detekce a sledování dřevěných kostek (R/G/B) pomocí YOLOv8, HSV trackingu a jednoduchého P‑regulátoru. Výstupem je řízení levého/pravého motoru přes UART.*

> Tento README shrnuje jak projekt spustit na Raspberry Pi, jak nahrát kód přes síť (SCP/SSH), jak je zapojen UART na Arduino a jaká jsou aktuální omezení setupu. Níže je také „rychlý tahák“ s přímými příkazy.

---

## 📌 Rychlý tahák (cheat sheet)

### 1) Nahrání projektu z notebooku/PC na Raspberry Pi (SCP)
```bash
scp -r ~/Plocha/Programing/Rasberry/RGB-cubes-detection/*  pi@10.42.0.1:/home/pi/Desktop/CUBES
# -r   : rekurzivně
# pi@… : uživatel a IP adresa Raspberry Pi
# cílová cesta vytvoří složku CUBES (pokud už neexistuje)
```
> Tip: pro rychlejší a „chytřejší“ aktualizace použij `rsync` (přenese jen změněné soubory):
```bash
rsync -av --delete ~/Plocha/Programing/Rasberry/RGB-cubes-detection/  pi@10.42.0.1:/home/pi/Desktop/CUBES/
```

### 2) Přihlášení na Raspberry Pi (SSH)
```bash
ssh -Y pi@10.42.0.1
# -Y = X11 forwarding (není potřeba, pokud nespouštíš GUI okna)
```

### 3) Spuštění skriptu na Raspberry Pi
**Varianta A: po přihlášení**
```bash
python3 /home/pi/Desktop/CUBES/cube_best.py
```

**Varianta B: jedním příkazem z PC (bez interaktivního shellu)**
```bash
ssh pi@10.42.0.1 'python3 /home/pi/Desktop/CUBES/cube_best.py'
```

**(volitelné) Spuštění v tmux, aby skript běžel i po odpojení SSH:**
```bash
ssh pi@10.42.0.1
tmux new -s cubes
python3 /home/pi/Desktop/CUBES/cube_best.py
# odpojení:  Ctrl+B, pak D   (návrat: tmux attach -t cubes)
```

**(volitelné) Autostart pomocí systemd (náčrt):**
```ini
# /etc/systemd/system/cubes.service
[Unit]
Description=CUBES tracker
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/Desktop/CUBES
ExecStart=/usr/bin/python3 /home/pi/Desktop/CUBES/cube_best.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cubes.service
journalctl -u cubes.service -f
```

---
![Demo ukázka](media/demo.gif)


## 🧭 Co projekt dělá (stručně)
- **Detekuje kostky** pomocí modelu **YOLOv8** (`best.pt`), filtruje přibližně čtvercové boxy nad minimální velikost.  
- **Zvolí „nejlepší“ kostku** – preferuje tu blíž **středu** a **spodnímu okraji** obrazu (typicky cíl před robotem).  
- **Změří barvu v HSV** v okolí středu detekce a dále kostku **sleduje** přes HSV masku + morfologii + kontury.  
- Z odchylky od středu spočítá **P‑regulátor** (`Kp`, `v_const`) a posílá **PWM páry** `L,R` po **UART** (0–255).  
- **YOLO běží v odděleném vlákně** v intervalu `DETECT_PERIOD`, hlavní smyčka jede ~50 Hz (podle HW).  
```mermaid
flowchart TD
    CAM[USB/CSI kamera 640x480] --> YOLO[YOLOv8 detekce]
    YOLO -->|nejlepší box + průměr HSV| HSV[HSV tracking]
    HSV -->|střed cíle (x,y)| P[P‑regulátor]
    P --> UART[UART "L,R\n"]
    UART --> MCU[Arduino / driver motorů]
```

---

## 🔌 UART zapojení a chování Arduina
**Směr komunikace je pouze jednosměrný z Raspberry Pi → Arduino.**  
**Spoje:**
```
Raspberry Pi TXD0 (GPIO14, 3.3 V)  --->  Arduino RX (D0)
GND                                   --  GND  (společná zem je povinná)
# Arduino TX (5 V) do RPi NEPŘIPOJUJEME (aby 5 V nešlo do 3.3 V logiky RPi)
```
- Skript na RPi posílá po UART každých ~20 ms řádky ve formátu `L,R\n`, kde **L** a **R** jsou v rozsahu **0–255**.  
- Arduino **jen čte** tyto hodnoty a nastavuje podle nich **rychlost levého/pravého motoru** (PWM).  
- V kódu je port `/dev/serial0 @ 115200 baud` (v případě potřeby uprav).

> Rychlá diagnostika na RPi:
```bash
ls -l /dev/serial*           # dostupné UART zařízení
stty -F /dev/serial0 115200  # nastavení rychlosti (pro test)
```

---

## ⚠️ Známá omezení aktuálního setupu
- **Levné žluté motory a drivery** v současné verzi umožňují **jen jeden směr otáčení**. To přirozeně **zhoršuje manévrovatelnost** a občas robot necílí optimálně – není to chybou algoritmu, ale limitem HW.
- **Kamera má buffer** a občas posílá **opožděné snímky** (lag). To se projeví latencí v reakci na cíl.  
  *Tipy:* zkusit snížit rozlišení / FPS, vypnout auto‑nastavení, případně **pravidelně „odpouštět“ buffer** (přečíst více snímků a použít poslední), nebo nastavit `CAP_PROP_BUFFERSIZE=1` (pokud backend podporuje).
- **Napájení a zem**: u slabého napájení motorů může docházet k rušení (projeví se „cukáním“). Zkontroluj společnou zem a filtraci.

**Plán do budoucna:**
- Lepší motory a driver s obousměrným řízením.
- Ideálně **nepoužívat Arduino** a přejít na lepší řídicí desku s kvalitnějším motor driverem.
- Nasazení na dalších robotech (přenositelnost kódu).

---

## 🧪 Příklad UART výstupu skriptu
Každých ~20 ms:
```
L,R\n   # např. 128,200  (levý pomaleji, pravý rychleji → otáčení doleva)
```
V terminálu zároveň běží log: režim (`HSV/IDLE`), střed cíle, aktuální PWM a odhadnutá barva HSV.

---

## 🧩 Požadavky

### Hardware (příkladové)
- Raspberry Pi (Linux, **/dev/serial0**), USB/CSI **kamera** (ideálně 640×480@30 fps).
- Arduino s driverem motorů (aktuálně jednosměrné řízení).

### Software
- **Python 3.9+**
- Knihovny: `ultralytics`, `opencv-python`, `numpy`, `pyserial`

```bash
# doporučené: virtuální prostředí
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python numpy pyserial
```

---

## 🚀 Instalace a spuštění (detailně)

1. Naklonuj projekt, připrav prostředí (viz výše) a **přidej model** `best.pt` do kořene repozitáře.
2. Připoj kameru a UART (viz schéma). Na RPi zkontroluj, že `/dev/serial0` je aktivní a nebyl kolidující login na UARTu (případně `raspi-config` → Interface → Serial, vypnout login, nechat HW UART).
3. Spusť hlavní skript:
```bash
python3 cube_best.py
```
> Ukončení: **Ctrl+C** (skript uvolní zdroje a ukončí vlákna).

**Výchozí hodnoty (viz `cube_best.py`):**
| Parametr         | Default        | Popis |
|------------------|----------------|-------|
| `DETECT_PERIOD`  | `1.5`          | perioda YOLO detekcí (vlákno) |
| `HSV_H_TOL`      | `10`           | tolerance odstínu (Hue) |
| `HSV_S_TOL`      | `50`           | tolerance sytosti (Saturation) |
| `HSV_V_TOL`      | `50`           | tolerance jasu (Value) |
| `CONF_THRESH`    | `0.4`          | minimální důvěra detekce |
| `IMG_W, IMG_H`   | `640, 480`     | rozlišení kamery |
| `Kp`             | `0.0009`       | zesílení P‑regulátoru (otáčení) |
| `v_const`        | `0.7`          | základní rychlost vpřed (0–1) |
| UART             | `/dev/serial0 @ 115200` | rozhraní a rychlost |

---


## 🗂 Struktura (minimum)
```
.
├── cube_best.py   # hlavní skript (YOLO + HSV + UART)
├── README.md
└── best.pt        # YOLOv8 model (není součástí repa)
```

---

## 🧭 Tipy a užitečné příkazy navíc
```bash
# kopírování s kompresí
scp -rC <lokální_složka>/  pi@10.42.0.1:/home/pi/Desktop/CUBES/

# SSH bez hesla (jednou vygeneruj klíč a nahraj na RPi)
ssh-keygen -t ed25519 -C "moje-pi"
ssh-copy-id pi@10.42.0.1

# kontrola kamery na RPi
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext

# sériová práva (pokud by bylo potřeba)
sudo usermod -aG dialout $USER  # po přihlášení znovu
```

---

## 📜 Licence
Zatím neuvedena. Doporučení: MIT nebo Apache‑2.0.
