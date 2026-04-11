#!/usr/bin/env python3
"""
Benchmark de reconstruction 3D — métriques en fonction du nombre d'images.

Principe : à partir d'un dossier de N photos, lance des reconstructions
successives en ajoutant --step images à chaque itération (5, 10, 15, …, N).
Pour chaque palier, mesure le temps total, le pic RAM et le CPU moyen.

Usage :
  python benchmark.py --image_folder img_mapanything/
  python benchmark.py --image_folder img_mapanything/ --step 10
  python benchmark.py --compare   # régénère seulement le graphe
"""

# ── Bibliothèques standard ──────────────────────────────────────────────────
import os
import sys
import json
import time
import shutil
import socket
import argparse
import tempfile
import platform
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# ── Dépendances externes ────────────────────────────────────────────────────
#   pip install psutil matplotlib
#   pip install pynvml          (optionnel — GPU NVIDIA uniquement)

try:
    import psutil                  # pip install psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False
    print("⚠  psutil non installé — CPU/RAM non monitorés.  →  pip install psutil")

try:
    import matplotlib              # pip install matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    MPL_OK = True
except ImportError:
    MPL_OK = False
    print("⚠  matplotlib non installé — graphe désactivé.  →  pip install matplotlib")

try:
    import pynvml                  # pip install pynvml  (optionnel, GPU NVIDIA)
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
except Exception:
    NVML_OK = False

try:
    import torch                   # pip install torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ── Détection du device ─────────────────────────────────────────────────────
def detect_device():
    if not TORCH_OK:
        return "cpu"
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


# ── Configuration machine ────────────────────────────────────────────────────
def build_machine_config(device_label: str) -> dict:
    """Construit le dict de configuration de la machine courante."""
    cpu_name = platform.processor() or platform.machine() or "inconnu"
    ram_total = round(psutil.virtual_memory().total / 1024 ** 3, 1) if PSUTIL_OK else None
    return {
        "device_label": device_label,
        "machine":      socket.gethostname(),
        "os":           platform.system() + " " + platform.release(),
        "cpu":          cpu_name,
        "ram_total_gb": ram_total,
        "device":       detect_device(),
    }


def load_or_create_config(config_path: Path, device_label: str) -> dict:
    """
    Charge un fichier JSON de configuration machine existant,
    ou en crée un nouveau si absent / si device_label diffère.
    """
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        # Si le label correspond, on réutilise le fichier existant
        if cfg.get("device_label") == device_label:
            print(f"✓  Configuration machine chargée : {config_path}")
            return cfg

    cfg = build_machine_config(device_label)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"✓  Configuration machine sauvegardée : {config_path}")
    return cfg


# ── Monitoring des ressources ───────────────────────────────────────────────
class ResourceMonitor:
    """Surveille CPU et RAM dans un thread séparé (polling 0.5 s)."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self._stop = threading.Event()
        self.cpu_samples = []
        self.ram_samples = []   # en Go
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        if PSUTIL_OK:
            psutil.cpu_percent(interval=None)  # première mesure souvent à 0
        while not self._stop.is_set():
            if PSUTIL_OK:
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
                self.ram_samples.append(psutil.virtual_memory().used / 1024 ** 3)
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        cpu_avg  = round(sum(self.cpu_samples) / len(self.cpu_samples), 1) if self.cpu_samples else None
        ram_peak = round(max(self.ram_samples), 2)                         if self.ram_samples else None
        return {"cpu_avg_pct": cpu_avg, "ram_peak_gb": ram_peak}


# ── Sous-dossier temporaire avec N images ────────────────────────────────────
def make_image_subset(source_dir: Path, images: list[str], tmp_root: Path) -> Path:
    """
    Crée un dossier temporaire contenant les N premières images.
    Utilise des liens symboliques (rapides, pas de copie) sur Unix,
    copie réelle en fallback.
    """
    tmp = tmp_root / f"subset_{len(images)}"
    tmp.mkdir(parents=True, exist_ok=True)
    # Vider le dossier des éventuels liens précédents
    for f in tmp.iterdir():
        f.unlink()
    for img in images:
        src = source_dir / img
        dst = tmp / img
        try:
            os.symlink(src.resolve(), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
    return tmp


# ── Reconstruction pour un sous-ensemble ────────────────────────────────────
def run_reconstruction_step(image_folder: Path, output_ply: Path) -> dict:
    """Lance reconstruction_medium.py et retourne les métriques du palier."""
    script = Path(__file__).parent / "reconstruction_medium.py"
    cmd = [
        sys.executable, str(script),
        "--image_folder", str(image_folder),
        "--output",       str(output_ply),
        "--no_visualize",
    ]

    monitor = ResourceMonitor()
    monitor.start()
    t0 = time.perf_counter()

    result = subprocess.run(cmd, capture_output=False, text=True)

    elapsed = round(time.perf_counter() - t0, 2)
    monitor.stop()
    resources = monitor.summary()

    return {
        "success":      result.returncode == 0,
        "total_time_s": elapsed,
        **resources,
    }


# ── Graphe métriques vs nombre d'images ─────────────────────────────────────
def generate_graph(results_dir: Path):
    if not MPL_OK:
        print("⚠  matplotlib absent — graphe non généré.")
        return

    all_file = results_dir / "all_results.json"
    if not all_file.exists():
        print("⚠  all_results.json introuvable.")
        return

    with open(all_file, encoding="utf-8") as f:
        all_runs = json.load(f)

    if not all_runs:
        print("⚠  all_results.json est vide.")
        return

    # Palette de couleurs par appareil
    PALETTE = ["#4a90ff", "#38f0b8", "#ff8800", "#ff5f5f", "#a06aff",
               "#ffd700", "#00cfff", "#ff69b4"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#080c10")

    metrics = [
        ("total_time_s", "Temps total (s)",  "Secondes"),
        ("ram_peak_gb",  "RAM pic (Go)",      "Go"),
        ("cpu_avg_pct",  "CPU moyen (%)",     "%"),
    ]

    for ax, (key, title, ylabel) in zip(axes, metrics):
        ax.set_facecolor("#0d1117")
        ax.set_title(title, color="#c8d8f0", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Nombre d'images", color="#507090", fontsize=9)
        ax.set_ylabel(ylabel, color="#507090", fontsize=9)
        ax.tick_params(colors="#c8d8f0", labelsize=9)
        ax.grid(color="#1a2535", linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#1a2535")
            spine.set_linewidth(0.5)

        for idx, run in enumerate(all_runs):
            steps   = run.get("steps", [])
            color   = PALETTE[idx % len(PALETTE)]
            label   = run.get("device_label", run.get("machine", f"run {idx+1}"))
            xs = [s["n_images"] for s in steps if s.get("success") and s.get(key) is not None]
            ys = [s[key]        for s in steps if s.get("success") and s.get(key) is not None]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", color=color, linewidth=2,
                    markersize=5, label=label, zorder=3)
            # Annoter le dernier point
            ax.annotate(
                f"{ys[-1]:.1f}",
                xy=(xs[-1], ys[-1]),
                xytext=(4, 4), textcoords="offset points",
                color=color, fontsize=7, fontweight="bold"
            )

        ax.legend(fontsize=8, facecolor="#0d1117", edgecolor="#1a2535",
                  labelcolor="#c8d8f0", loc="upper left")

    fig.suptitle(
        "Benchmark MapAnything — métriques vs nombre d'images",
        color="#38f0b8", fontsize=13, fontweight="bold"
    )
    fig.text(
        0.5, 0.01,
        f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        ha="center", color="#507090", fontsize=8
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = results_dir / "benchmark_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓  Graphe sauvegardé : {out}")


# ── Point d'entrée ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de reconstruction 3D MapAnything (métriques vs N images)"
    )
    parser.add_argument("--image_folder", type=str, default=None,
                        help="Dossier source contenant toutes les photos")
    parser.add_argument("--step", type=int, default=5,
                        help="Incrément du nombre d'images par palier (défaut: 5)")
    parser.add_argument("--results_dir", type=str, default="benchmark_results",
                        help="Dossier de sortie JSON + graphe (défaut: benchmark_results/)")
    parser.add_argument("--device_label", type=str, default=None,
                        help="Nom de l'appareil affiché sur le graphe (ex: 'MacBook Pro M2')")
    parser.add_argument("--config", type=str, default=None,
                        help="Chemin vers le fichier JSON de config machine (créé si absent)")
    parser.add_argument("--compare", action="store_true",
                        help="Régénérer uniquement le graphe sans lancer de reconstruction")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Mode graphe seul ────────────────────────────────────────────────────
    if args.compare:
        print("📊  Génération du graphe de comparaison…")
        generate_graph(results_dir)
        return

    # ── Sélection du dossier images ─────────────────────────────────────────
    image_folder = args.image_folder
    if not image_folder:
        print("📁  Quel dossier de photos utiliser pour le benchmark ?")
        image_folder = input("   → Chemin : ").strip().strip('"').strip("'")
    image_folder = Path(image_folder)
    if not image_folder.is_dir():
        print(f"✗  Dossier introuvable : {image_folder}")
        sys.exit(1)

    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    all_images = sorted(f for f in os.listdir(image_folder) if f.endswith(exts))
    if not all_images:
        print(f"✗  Aucune image trouvée dans : {image_folder}")
        sys.exit(1)
    print(f"   ✓ {len(all_images)} image(s) trouvée(s)")

    # ── Nom de l'appareil ───────────────────────────────────────────────────
    device_label = args.device_label
    if not device_label:
        print("\n🏷   Nom de l'appareil pour le graphe ?")
        print("    (ex : 'MacBook Pro M2', 'PC Bureau RTX 3090')")
        device_label = input("   → ").strip() or socket.gethostname()

    # ── Configuration machine ────────────────────────────────────────────────
    config_path = Path(args.config) if args.config else results_dir / f"config_{device_label.replace(' ', '_')}.json"
    machine_cfg = load_or_create_config(config_path, device_label)

    print(f"\n{'─'*55}")
    print(f"  Appareil : {machine_cfg['device_label']}")
    print(f"  OS       : {machine_cfg['os']}")
    print(f"  CPU      : {machine_cfg['cpu']}")
    print(f"  RAM      : {machine_cfg['ram_total_gb']} Go")
    print(f"  Device   : {machine_cfg['device']}")
    print(f"  Images   : {len(all_images)}  |  Paliers de {args.step}")
    print(f"{'─'*55}\n")

    # ── Paliers d'images ────────────────────────────────────────────────────
    paliers = list(range(args.step, len(all_images) + 1, args.step))
    if not paliers or paliers[-1] != len(all_images):
        paliers.append(len(all_images))

    # Dossier temporaire pour les sous-ensembles
    tmp_root = results_dir / "_tmp_subsets"

    steps_results = []

    for n in paliers:
        subset = all_images[:n]
        print(f"\n{'═'*55}")
        print(f"  PALIER {n} images")
        print(f"{'═'*55}")

        subset_dir = make_image_subset(image_folder, subset, tmp_root)
        ply_out    = tmp_root / f"recon_tmp.ply"

        metrics = run_reconstruction_step(subset_dir, ply_out)

        # Supprimer le PLY temporaire immédiatement après la mesure
        if ply_out.exists():
            ply_out.unlink()

        step_data = {
            "n_images":     n,
            "success":      metrics["success"],
            "total_time_s": metrics["total_time_s"],
            "ram_peak_gb":  metrics["ram_peak_gb"],
            "cpu_avg_pct":  metrics["cpu_avg_pct"],
        }
        steps_results.append(step_data)

        status = "✓" if metrics["success"] else "✗"
        print(f"\n  {status}  {n} images → "
              f"temps={metrics['total_time_s']:.1f}s  "
              f"RAM pic={metrics['ram_peak_gb'] or '?'} Go  "
              f"CPU moy={metrics['cpu_avg_pct'] or '?'} %")

    # Nettoyage dossier temporaire
    shutil.rmtree(tmp_root, ignore_errors=True)

    # ── Construction du résultat complet ─────────────────────────────────────
    run_result = {
        **machine_cfg,
        "date":         datetime.now().isoformat(),
        "image_folder": str(image_folder),
        "step":         args.step,
        "steps":        steps_results,
    }

    # Sauvegarde JSON individuel
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"{device_label.replace(' ', '_')}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)
    print(f"\n✓  Résultat sauvegardé : {out_file}")

    # Mise à jour all_results.json
    all_file = results_dir / "all_results.json"
    all_results = []
    if all_file.exists():
        try:
            with open(all_file, encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            pass
    all_results.append(run_result)
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓  all_results.json mis à jour")

    # ── Graphe final ─────────────────────────────────────────────────────────
    generate_graph(results_dir)
    print(f"\n✓  Benchmark terminé — {len(paliers)} paliers réalisés.\n")


if __name__ == "__main__":
    main()
