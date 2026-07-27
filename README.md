# DepthGen

Lokalna aplikacja: **obraz → mapa głębi → szczegółowa płaskorzeźba 3D** z podglądem live i pełną
kontrolą parametrów. Wszystko liczy się na Twoim komputerze — żadnych usług chmurowych, żadnych
limitów. Modele pobierane są raz z HuggingFace i potem działają offline.

## Uruchomienie

Kliknij dwukrotnie **`DepthGen.bat`** — to wszystko. Skrypt sam sprawdzi środowisko (przy pierwszym
uruchomieniu zaproponuje instalację), znajdzie wolny port, wypisze wykrytą kartę graficzną
i otworzy przeglądarkę. Zamknięcie okna konsoli kończy pracę serwera.

```bash
D:\apps\depthgen\DepthGen.bat
```

Wymagania: Python 3.10+ i (opcjonalnie) karta NVIDIA z CUDA. Bez GPU działa na CPU, tylko wolniej.
`setup.bat` przydaje się osobno tylko przy przenoszeniu projektu na inny komputer.

## Jak to działa

1. **Mapa głębi** — sieć neuronowa (Depth Anything V2 / MiDaS DPT) szacuje względną odległość
   każdego piksela. Opcjonalny **przebieg kafelkowy** dzieli obraz na kafle, liczy każdy w pełnej
   rozdzielczości sieci i wszczepia z nich wysokie częstotliwości w globalną mapę — to daje
   wyraźnie więcej mikrodetalu (włosy, tkanina, faktura skóry).
2. **Mapa wysokości** — obróbka: normalizacja percentylowa, gamma, krzywa S, wyostrzenie
   z ochroną krawędzi, mikrodetal z luminancji obrazu, spłaszczanie tła, kształt płyty.
3. **Siatka** — pole wysokości zamieniane na trójkąty; opcjonalnie zamknięta bryła z płytą bazową
   i ściankami bocznymi (szczelna, gotowa do druku 3D).

## Parametry

**Obraz źródłowy**
| Parametr | Znaczenie |
|---|---|
| Model głębi | Small = szybki, Large = najlepszy detal. BEiT-L 512 daje inny, ostrzejszy charakter |
| Rozdz. sieci | Wyższa = więcej detalu i dłuższy czas. Sensowny zakres 518–1036 |
| Przebieg kafelkowy | 2×2 lub 3×3 kafle = znacznie więcej mikrodetalu (kilka razy dłużej) |
| Siła detalu z kafli | Ile wysokich częstotliwości z kafli wchodzi do mapy |

**Głębia**
| Parametr | Znaczenie |
|---|---|
| Odwróć | Zamienia bliskie/dalekie (np. do litofanii) |
| Odcięcie dołu/góry | Percentyle rozciągnięcia zakresu — obcina wartości odstające |
| Gamma | <1 podbija wypukłości, >1 spłaszcza pierwszy plan |
| Kontrast (krzywa S) | Rozdziela plany; ujemny spłaszcza |

**Detal i wygładzanie**
| Parametr | Znaczenie |
|---|---|
| Odszumianie medianowe | Usuwa pojedyncze piksele-kolce (typowo na włosach) |
| Wygładzanie z zach. krawędzi | Filtr bilateralny — kasuje szum, zostawia ostre przejścia |
| Rozmycie | Zwykły gauss, gdy mapa jest zbyt „ziarnista" |
| Wyostrzenie reliefu | Podbija detal (unsharp na mapie wysokości) |
| Promień wyostrzenia | Skala podbijanego detalu w px |
| **Ochrona krawędzi sylwetki** | Blokuje przestrzał wyostrzania na skoku głębi — bez tego na obrysie robią się kolce |
| **Limit amplitudy detalu** | Twardy sufit dokładanego detalu |
| Mikrodetal z obrazu | Wszczepia fakturę z jasności zdjęcia — włosy, tkanina, rzeźbienia |
| Promień mikrodetalu | Zwykle 1.5–3 px |

**Tło i kształt**
| Parametr | Znaczenie |
|---|---|
| Próg tła / miękkość | Wszystko poniżej progu opada do płaskiego tła |
| Zejście przy krawędziach | Relief łagodnie schodzi do zera przy brzegach płyty |
| Kształt płyty | prostokąt / zaokrąglone rogi / owal |
| Margines | Pusta ramka wokół reliefu |
| Przytnij bryłę do kształtu | Siatka jest realnie wycinana do kształtu (nie tylko płaskie tło) |

**Wycięcie sylwetki** — odcina płaską płytę i zostawia sam relief. Działa razem
z „Przytnij bryłę do kształtu".

| Parametr | Znaczenie |
|---|---|
| Tnij po przezroczystości | Używa kanału alfa obrazu jako obrysu bryły. Włącza się samo, gdy wgrany PNG ma przezroczyste tło |
| Próg alfy | Od jakiej krycia piksel należy do sylwetki (istotne przy wygładzonych krawędziach wycinków) |
| Poszerz / zwęź sylwetkę | Zapas materiału na krawędzi; ujemne wartości ścinają obwódkę zostawioną przez wycinanie |
| Odetnij poniżej wysokości | Wycięcie dla obrazów **bez** alfy — usuwa wszystko poniżej progu wysokości |
| Usuń wysepki | Kasuje drobne okruchy, które inaczej zostają w druku jako luźne odpryski |

Obrys nie musi być wypukły: kształty wklęsłe, z otworami i rozpadające się na kilka
osobnych brył dają poprawną, szczelną siatkę (dno jest wtedy lustrem siatki wierzchu,
a nie wachlarzem).

**Siatka i wymiary**
| Parametr | Znaczenie |
|---|---|
| Rozdzielczość podglądu | Próbki wzdłuż dłuższej krawędzi; 400–700 = płynna praca |
| Szerokość / Wysokość reliefu / Grubość bazy | Wymiary fizyczne w mm |
| Zamknięta bryła | Szczelna bryła z bazą i ściankami (do druku). Wyłączone = sama powierzchnia |

**Eksport** — STL, OBJ, PLY, GLB, 3MF; osobna, wyższa rozdzielczość niż podgląd.
Można też zapisać samą mapę wysokości jako 16-bitowy PNG (do ZBrush, Blendera, Fusion).

## Presety

`Portret`, `Maksymalny detal`, `Logo / grafika płaska`, `Krajobraz`, `Medalion owalny`,
`Wycinanka PNG`, `Litofania`. Ustawienia zapisują się w przeglądarce między sesjami.

## Podgląd 3D

Obrót myszą, kółko = zoom, PPM = przesuwanie. Materiały (glina/gips/brąz/stal/normalne),
siatka, obrót, oraz **suwak kierunku światła** — światło ślizgowe najlepiej pokazuje, ile detalu
naprawdę ma relief. Podgląd przelicza się na bieżąco przy każdej zmianie parametru
(można wyłączyć przełącznikiem „podgląd live").

## Wskazówki praktyczne

- **Za mało detalu?** Podnieś rozdz. sieci, włącz kafle 2×2, dodaj „Mikrodetal z obrazu".
- **Kolce i postrzępione krawędzie?** Podnieś „Ochronę krawędzi sylwetki", zmniejsz
  „Limit amplitudy", włącz odszumianie medianowe 3 px.
- **Tło ma niechcianą fakturę?** Podnieś „Próg tła".
- **Do druku 3D**: „Zamknięta bryła" włączona, baza min. 1.5–2 mm, eksport STL.
  Rozdzielczość eksportu 2000+ daje ~8 mln trójkątów i plik rzędu kilkuset MB — dla większości
  drukarek 1200–1600 w zupełności wystarcza.
- **Do CNC / ZBrush**: eksportuj 16-bitową mapę wysokości i użyj własnego pipeline'u.

## Struktura

```
app/depth.py       modele głębi, przebieg kafelkowy
app/heightmap.py   obróbka mapy głębi -> mapa wysokości
app/mesh.py        pole wysokości -> siatka, eksport
app/main.py        API (FastAPI)
app/static/        interfejs + podgląd three.js (wersja lokalna, działa offline)
tests/             testy geometrii i test end-to-end
```

## Testy

```bash
.venv\Scripts\python.exe tests\test_mesh.py
```
```bash
.venv\Scripts\python.exe tests\test_params.py
```

`test_mesh.py` — szczelność bryły, spójność orientacji trójkątów, dodatnia objętość, wymiary
w mm, wszystkie formaty eksportu.

`test_params.py` — przechodzi po **każdym** parametrze z interfejsu, buduje siatkę przed i po
zmianie i mierzy maksymalne przesunięcie wierzchołków w mm. Wyłapuje martwe kontrolki, czyli
suwaki, które ruszają się w UI, ale niczego nie zmieniają w modelu. Sprawdza też determinizm:
te same parametry muszą dać bit w bit tę samą siatkę.

```bash
.venv\Scripts\python.exe tests\test_cut.py
```
```bash
.venv\Scripts\python.exe tests\test_tiles.py
```

`test_cut.py` — wycinanie sylwetki na kształtach wklęsłych, z otworem i rozpadających się
na kilka brył; sprawdza szczelność, faktyczne zniknięcie płyty i filtr wysepek.

`test_tiles.py` — artefakty przebiegu kafelkowego. Porównuje obecny algorytm ze starym
i mierzy poświatę wokół sylwetki, płaskość tła oraz realny przyrost mikrodetalu.
Wymaga pliku `tests/sample.jpg` i pobiera model Depth Anything V2 Small.
