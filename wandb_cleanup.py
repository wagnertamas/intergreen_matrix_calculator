#!/usr/bin/env python3
"""
WandB cleanup helper a grid search scripthez.

Használat:
  python wandb_cleanup.py --project sumo-rl-stat --done-dir grid_done

Megkeresi azokat a WandB runokat, amelyekhez nincs .done marker fájl
(azaz félbeszakadt vagy hibás futások), és felkínálja törlésre.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="WandB árva runok törlése")
    parser.add_argument("--project", required=True, help="WandB projekt neve")
    parser.add_argument("--done-dir", required=True, help="Marker fájlok könyvtára")
    parser.add_argument("--entity", default=None, help="WandB entity (auto-detect ha üres)")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb nincs telepítve, kihagyom az ellenőrzést.")
        sys.exit(0)

    api = wandb.Api()

    # Entity auto-detect
    entity = args.entity
    if not entity:
        entity = api.default_entity
    if not entity:
        print("Nem sikerült az entity-t meghatározni. Használd a --entity flaget.")
        sys.exit(1)

    project_path = f"{entity}/{args.project}"

    # WandB runok lekérése
    print(f"WandB runok lekérése: {project_path} ...")
    try:
        runs = api.runs(project_path)
    except Exception as e:
        print(f"WandB lekérés sikertelen: {e}")
        sys.exit(0)

    # Done markerek beolvasása
    done_names = set()
    if os.path.isdir(args.done_dir):
        for f in os.listdir(args.done_dir):
            if f.endswith(".done"):
                done_names.add(f[:-5])  # strip .done

    # Árva runok keresése (WandB-n van, de nincs .done marker)
    orphans = []
    for r in runs:
        name = r.name
        if name and name not in done_names:
            orphans.append(r)

    if not orphans:
        print(f"Nincs árva run. WandB ({len(runs)}) és done markerek ({len(done_names)}) szinkronban.")
        sys.exit(0)

    # Kiírás
    print(f"\n{'='*60}")
    print(f"  {len(orphans)} árva run találva a WandB-n")
    print(f"  (WandB-n létezik, de nincs .done marker)")
    print(f"{'='*60}")
    for r in orphans:
        print(f"  {r.name}  (state={r.state})")
    print(f"{'='*60}")
    print()
    print("Ezek félbeszakadt vagy hibás futások lehetnek.")
    print("Törlés után a grid search újra lefuttatja őket.")
    print()
    print("Törléshez írd be: delete")
    print("Kihagyáshoz nyomj ENTER-t.")
    print()

    try:
        answer = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nKihagyva.")
        sys.exit(0)

    if answer != "delete":
        print("Kihagyva — árva runok megmaradnak.")
        sys.exit(0)

    # Törlés
    print(f"\nTörlés folyamatban ({len(orphans)} run)...")
    deleted = 0
    for i, r in enumerate(orphans):
        try:
            r.delete()
            deleted += 1
        except Exception as e:
            print(f"  [HIBA] {r.name}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(orphans)} törölve...")

    print(f"Kész! {deleted}/{len(orphans)} run törölve a WandB-ről.")


if __name__ == "__main__":
    main()
