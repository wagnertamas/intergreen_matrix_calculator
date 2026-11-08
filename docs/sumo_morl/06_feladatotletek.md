# Feladatötletek hallgatóknak

Nehézség szerint növekvő sorrendben.

## Bevezető

1. **Ismerkedés.** Futtasd az `01_ismerkedes.py` példát `--gui` kapcsolóval.
   Figyeld meg az átmeneteket (sárga, piros-sárga). Melyik fázisváltás a
   leghosszabb, és miért?
2. **Baseline-ok.** Futtasd a `02_baseline_osszehasonlitas.py`-t. Keress olyan
   fix idejű zöldidő-kombinációt, amely a `queue` célban jobb a megadottaknál!
   Melyik másik cél romlik közben?

## Egycélú → többcélú

3. **Lineáris skalarizálás.** Taníts DQN-t vagy PPO-t 5–10 különböző
   súlyvektorral (`03_sb3_sulyozott_tanitas.py`). Rajzold fel a Pareto-frontot,
   és számold ki a hipertérfogatot. Vesd össze a baseline-okkal.
4. **Súly-kondicionált ágens.** Taníts egyetlen ágenst `WeightConditioned`
   wrapperrel, majd értékeld ki ugyanazokkal a súlyokkal, mint a 3. feladatban.
   Melyik front jobb? Mennyivel kevesebb tanítás kellett?
5. **Saját cél.** Definiálj új célfüggvényt (pl. a mellékág várakozása, vagy a
   főirány haladási sebessége az előjelző hurkokon), és vizsgáld meg, hogyan
   változik a front.

## Haladó

6. **Valódi MORL algoritmus.** Implementálj egy vektor-Q-értékes módszert (pl.
   Envelope Q-learning vagy Pareto-alapú DQN) a vektor-jutalomra.
7. **Nemlineáris hasznosság.** Optimalizálj a lineáris helyett pl. Csebisev
   (Tchebycheff) skalarizálással. Talál-e a front nem konvex részén pontot?
8. **Változó lépéshossz.** A lépések hossza változik (5 s vagy 9–11 s). Írj
   wrappert, amely a diszkontot `gamma ** info["elapsed"]` szerint alkalmazza
   (SMDP), és vizsgáld meg a hatását.
9. **Robusztusság.** Tanítsd az ágenst csak alacsony forgalmon, majd értékeld ki
   csúcsórás profillal (`Demand.random(rng, profile=(0.5, 1, 1.5, 1))`). Hogyan
   romlik a teljesítmény?
10. **Kevesebb szenzor.** Mi történik, ha egy detektortípust elhagysz a
    megfigyelésből (pl. csak stopvonali hurkok)? Melyik detektor a
    legfontosabb az ágensnek?

## Kiértékelési tanácsok

* Mindig **azonos forgalmon** (azonos `seed`) hasonlíts össze vezérlőket.
* Több epizód (legalább 3–5) átlagát és szórását add meg.
* A dolgozatban a jutalmon kívül a valós mutatókat (`info["ground_truth"]`,
  pl. átlagos késés) is érdemes közölni, de ezeket ne használd tanításhoz.
