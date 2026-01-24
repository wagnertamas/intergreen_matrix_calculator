"""
Reward Monitor - CO2 és Travel Time metrikák logolása CSV fájlokba.

Használat:
    from misc.reward_monitor import RewardMonitor

    monitor = RewardMonitor(output_dir="reward_logs")

    # Training loop-ban:
    obs, reward, terminated, truncated, info = env.step(actions)
    monitor.log_step(step_num, info)

    # Végén mentés:
    monitor.save_all()

Kimenet:
    output_dir/
        junction_A_co2.csv
        junction_A_travel_time.csv
        junction_B_co2.csv
        junction_B_travel_time.csv
        ...

CSV formátum (pandas-kompatibilis):
    step,value
    0,12.5
    1,13.2
    ...
"""

import os
import csv
from typing import Dict, Any, Optional
from collections import defaultdict


class RewardMonitor:
    """
    Súlyozatlan CO2 és travel time metrikák monitorozása.
    Minden kereszteződéshez és metrikához külön CSV fájl.
    """

    MAX_JITTERS = ['co2', 'travel_time', 'veh_count']

    def __init__(self, output_dir: str = "reward_logs", auto_save_interval: Optional[int] = None):
        """
        Args:
            output_dir: Kimeneti mappa a CSV fájloknak
            auto_save_interval: Ha megadva, ennyi lépésenként automatikusan ment (None = csak manuális mentés)
        """
        self.output_dir = output_dir
        self.auto_save_interval = auto_save_interval

        # Adatok tárolása: {junction_id: {'co2': [], 'travel_time': [], 'veh_count': []}}
        self.data: Dict[str, Dict[str, list]] = defaultdict(lambda: {'co2': [], 'travel_time': [], 'veh_count': []})

        self.step_count = 0
        self.junction_ids = set()

        # Mappa létrehozása
        os.makedirs(output_dir, exist_ok=True)

    def log_step(self, step: int, infos: Dict[str, Dict[str, Any]]) -> None:
        """
        Egy lépés metrikáinak logolása.

        Args:
            step: Aktuális lépésszám
            infos: Az env.step() által visszaadott info dict
                   Elvárt struktúra: {junction_id: {'metric_travel_time': float, 'metric_co2': float, ...}}
        """
        for junction_id, info in infos.items():
            # __all__ kulcs kihagyása (ez a terminated/truncated flag)
            if junction_id == "__all__":
                continue

            self.junction_ids.add(junction_id)

            # CO2 mentése (súlyozatlan, nyers érték)
            co2_value = info.get('metric_co2', 0.0)
            self.data[junction_id]['co2'].append((step, co2_value))

            # Travel time mentése (súlyozatlan, nyers érték)
            travel_time_value = info.get('metric_travel_time', 0.0)
            self.data[junction_id]['travel_time'].append((step, travel_time_value))

            # [NEW] Vehicle Count mentése
            veh_count_value = info.get('metric_veh_count', 0.0)
            self.data[junction_id]['veh_count'].append((step, veh_count_value))

        self.step_count = step

        # Auto-save ha be van állítva
        if self.auto_save_interval and step > 0 and step % self.auto_save_interval == 0:
            self.save_all()

    def log_from_env_step(self, obs, rewards, terminated, truncated, infos) -> None:
        """
        Kényelmi függvény: közvetlenül az env.step() kimenetéből logol.
        Automatikusan növeli a step countert.

        Args:
            obs, rewards, terminated, truncated, infos: env.step() visszatérési értékei
        """
        self.step_count += 1
        self.log_step(self.step_count, infos)

    def save_junction(self, junction_id: str) -> None:
        """Egy adott kereszteződés adatainak mentése."""
        if junction_id not in self.data:
            return

        # CO2 fájl
        co2_file = os.path.join(self.output_dir, f"{junction_id}_co2.csv")
        self._write_csv(co2_file, self.data[junction_id]['co2'])

        # Travel time fájl
        tt_file = os.path.join(self.output_dir, f"{junction_id}_travel_time.csv")
        self._write_csv(tt_file, self.data[junction_id]['travel_time'])

        # [NEW] Veh Count fájl
        vc_file = os.path.join(self.output_dir, f"{junction_id}_veh_count.csv")
        self._write_csv(vc_file, self.data[junction_id]['veh_count'])

    def save_all(self) -> None:
        """Összes kereszteződés adatainak mentése."""
        for junction_id in self.junction_ids:
            self.save_junction(junction_id)
        print(f"[RewardMonitor] Mentve: {len(self.junction_ids)} kereszteződés, {self.step_count} lépés -> {self.output_dir}/")

    def _write_csv(self, filepath: str, data: list) -> None:
        """CSV fájl írása pandas-kompatibilis formátumban."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'value'])  # Header
            writer.writerows(data)

    def get_dataframe(self, junction_id: str, metric: str) -> 'pd.DataFrame':
        """
        Pandas DataFrame visszaadása egy adott metrikához.

        Args:
            junction_id: Kereszteződés azonosító
            metric: 'co2' vagy 'travel_time' vagy 'veh_count'

        Returns:
            pandas.DataFrame columns=['step', 'value']
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas szükséges a get_dataframe() használatához")

        if junction_id not in self.data or metric not in self.data[junction_id]:
            return pd.DataFrame(columns=['step', 'value'])

        return pd.DataFrame(self.data[junction_id][metric], columns=['step', 'value'])

    def get_all_dataframes(self) -> Dict[str, Dict[str, 'pd.DataFrame']]:
        """
        Összes adat DataFrame formátumban.

        Returns:
            {junction_id: {'co2': DataFrame, 'travel_time': DataFrame, 'veh_count': DataFrame}}
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas szükséges a get_all_dataframes() használatához")

        result = {}
        for junction_id in self.junction_ids:
            result[junction_id] = {
                'co2': self.get_dataframe(junction_id, 'co2'),
                'travel_time': self.get_dataframe(junction_id, 'travel_time'),
                'veh_count': self.get_dataframe(junction_id, 'veh_count')
            }
        return result

    def clear(self) -> None:
        """Összes tárolt adat törlése."""
        self.data.clear()
        self.data = defaultdict(lambda: {'co2': [], 'travel_time': []})
        self.junction_ids.clear()
        self.step_count = 0

    def get_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Összefoglaló statisztikák.

        Returns:
            {junction_id: {'co2': {'mean': x, 'sum': y, 'min': z, 'max': w}, 'travel_time': {...}}}
        """
        summary = {}
        for junction_id in self.junction_ids:
            summary[junction_id] = {}
            for metric in ['co2', 'travel_time']:
                values = [v for _, v in self.data[junction_id][metric]]
                if values:
                    summary[junction_id][metric] = {
                        'mean': sum(values) / len(values),
                        'sum': sum(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
                else:
                    summary[junction_id][metric] = {
                        'mean': 0, 'sum': 0, 'min': 0, 'max': 0, 'count': 0
                    }
        return summary

    def print_summary(self) -> None:
        """Összefoglaló kiírása konzolra."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("  REWARD MONITOR ÖSSZEFOGLALÓ")
        print("="*60)
        print(f"  Lépések száma: {self.step_count}")
        print(f"  Kereszteződések: {len(self.junction_ids)}")
        print("-"*60)

        for jid in sorted(self.junction_ids):
            print(f"\n  [{jid}]")
            for metric in ['co2', 'travel_time']:
                s = summary[jid][metric]
                print(f"    {metric:12s}: mean={s['mean']:8.2f}, sum={s['sum']:10.2f}, min={s['min']:8.2f}, max={s['max']:8.2f}")

        print("\n" + "="*60)


# Használati példa
if __name__ == "__main__":
    # Demo: szintetikus adatokkal
    import random

    monitor = RewardMonitor(output_dir="demo_reward_logs")

    # Szimulált lépések
    junction_ids = ["J1", "J2", "J3"]

    for step in range(100):
        # Szimulált info dict (mintha env.step()-ből jönne)
        infos = {}
        for jid in junction_ids:
            infos[jid] = {
                'ready': True,
                'metric_travel_time': random.uniform(10, 50),
                'metric_co2': random.uniform(100, 500)
            }

        monitor.log_step(step, infos)

    # Összefoglaló és mentés
    monitor.print_summary()
    monitor.save_all()

    print("\nCSV fájlok létrehozva a 'demo_reward_logs' mappában.")
    print("Pandas betöltés példa:")
    print("  import pandas as pd")
    print("  df = pd.read_csv('demo_reward_logs/J1_co2.csv')")
