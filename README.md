# License Plate Recognition

A `requirments.txt` tartalmazza a szükséges függőségeket. Python verzióból 3.9.7-et használtunk. A projekt 22.04-es ubuntu alatt nem fog működni, a paddle miatt. 20.04-es ubuntu-t ajánljuk.

A kiértékelés futtatásához a tesztképeket egy directoryba kell helyezni, majd futtatni az evaluate_solution.py-t.
Ez az alábbi paranccsal lehetséges:

`python evaluate_solution.py -i <inputdir> (optional) -o <outputfile> -d <device_type> [\'cpu\', \'gpu\']`

Az -i tagnek kell átadni a directory elérési útját, amiben a tesztképeket tároljuk.

A többi tag optional, ha az output file nevét nem adjuk meg, akkor `result + timestap.csv` formátumban menti el őket.
Így például:
`python evaluate_solution.py -i ./test`

A device_type tagot megadva lehet definiálni, hogy a futtatás során a CPU vagy a GPU-t használja a program. Alapértelmezett értéke a CPU.

Amennyiben GUI-val akarjuk használni a programot, akkor a ui.py-t kell futtatni. Ezt a parancssorból a következő paranccsal lehet elérni:

`python ui.py`

A Load image gombbal lehet betölteni a képet, a Recognize gombbal pedig elindítani a felismerést. A program bejelöli a talált rendszámokat a képen, majd kiírja az ezekből kiolvasott rendszámokat.

A `utils` mappában találhatóak a segéd scriptek, amiket a projekt készítése során használtunk.

A rendszámok olvasását a `PlateReader` class hajtja végre. A `read(img)` függvénnyel lehetséges a felismerés.
