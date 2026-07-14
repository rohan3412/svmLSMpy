"""
Generate tiny synthetic lesion datasets for exercising the svmLSMpy pipeline
end to end. NOT scientifically meaningful - just enough voxels and a planted
signal so the SVR/SVC machinery runs and can detect a lesion->behavior relation.

Three self-contained datasets are written under tests/fixtures/data/, each with
its own lesions/ folder + behavior.csv so class balance is clean per task:

  svr/         continuous behavior in [0,100]  (graded load of signal blob A) -> SVR
  binary/      behavior in {0,1}   (blob A present / absent)                  -> binary SVC
  threeclass/  behavior in {0,1,2} (none / blob A / blob B by location)       -> multiclass SVC

Every CSV has columns: filename, behavior, age  (age is a covariate).

The affine is MNI-aligned (10 mm isotropic, centred near the MNI origin) so that
nilearn's compute_brain_mask - which resamples the MNI152 template mask onto the
image grid, independent of voxel values - returns a non-empty ~277-voxel mask.
Both signal blobs are placed to fall fully inside that mask.
"""
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

SEED = 42
SHAPE = (16, 16, 16)
VOX = 10.0
AFFINE = np.diag([VOX, VOX, VOX, 1.0])
AFFINE[:3, 3] = [-VOX * 8, -VOX * 8, -VOX * 8]      # centre grid near MNI origin

BRAIN = (slice(2, 14), slice(2, 14), slice(2, 14))   # background-lesion region
SIGNAL_A = (slice(3, 6), slice(2, 5), slice(10, 13))  # blob A (fully inside mask)
SIGNAL_B = (slice(4, 7), slice(10, 13), slice(8, 11)) # blob B (fully inside mask)
SIGNAL_C = (slice(4, 7), slice(6, 9), slice(9, 12))   # blob C (fully inside mask)
N_SUBJECTS = 30


def _background(rng):
    brain = np.zeros(SHAPE, dtype=bool)
    brain[BRAIN] = True
    vol = np.zeros(SHAPE, dtype=np.uint8)
    vol[(rng.random(SHAPE) < 0.15) & brain] = 1
    return vol


def _save_set(name, build_subject, behavior_fn, rng):
    """build_subject(i) -> (uint8 volume, latent); behavior_fn(latents) -> array."""
    out = Path(__file__).parent / "fixtures" / "data" / name
    lesdir = out / "lesions"
    lesdir.mkdir(parents=True, exist_ok=True)

    filenames, latents = [], []
    for i in range(N_SUBJECTS):
        vol, latent = build_subject(i)
        fname = f"sub-{i:02d}.nii.gz"
        nib.save(nib.Nifti1Image(vol, AFFINE), lesdir / fname)
        filenames.append(fname)
        latents.append(latent)

    ages = rng.integers(40, 80, size=N_SUBJECTS).astype(float)
    behavior = behavior_fn(np.array(latents))
    pd.DataFrame({"filename": filenames, "behavior": behavior, "age": ages}).to_csv(
        out / "behavior.csv", index=False
    )
    return behavior


def main():
    rng = np.random.default_rng(SEED)

    # --- SVR: continuous, graded load of blob A ---
    u = rng.random(N_SUBJECTS)
    noise = rng.normal(0, 0.08, size=N_SUBJECTS)

    def build_svr(i):
        vol = _background(rng)
        sig = np.zeros(SHAPE, dtype=bool); sig[SIGNAL_A] = True
        vol[(rng.random(SHAPE) < u[i]) & sig] = 1   # graded blob-A load
        return vol, u[i]

    cont = _save_set("svr", build_svr,
                     lambda lat: np.clip(20 + 60 * lat + 100 * noise, 0, 100).round(1), rng)

    # --- binary: blob A vs blob B, volume-matched so the signal is LOCATION ---
    # (class-vs-absent would confound class with lesion volume, which the default
    #  lesion-volume covariate regression then strips out; equal-size blobs in
    #  different locations keep a discriminative signal that survives that regression.)
    gb = np.resize([0, 1], N_SUBJECTS).copy()
    rng.shuffle(gb)

    def build_bin(i):
        vol = _background(rng)
        vol[SIGNAL_A if gb[i] == 1 else SIGNAL_B] = 1
        return vol, gb[i]

    binary = _save_set("binary", build_bin, lambda lat: lat.astype(int), rng)

    # --- 3-class: blob C / blob A / blob B by location, volume-matched, balanced ---
    g3 = np.resize([0, 1, 2], N_SUBJECTS).copy()
    rng.shuffle(g3)
    blob_for_class = {0: SIGNAL_C, 1: SIGNAL_A, 2: SIGNAL_B}

    def build_3(i):
        vol = _background(rng)
        vol[blob_for_class[g3[i]]] = 1
        return vol, g3[i]

    three = _save_set("threeclass", build_3, lambda lat: lat.astype(int), rng)

    print("Fixtures written under tests/fixtures/data/")
    print(f"  svr:        continuous min={cont.min()} max={cont.max()}")
    print(f"  binary:     counts={np.bincount(binary).tolist()}")
    print(f"  threeclass: counts={np.bincount(three).tolist()}")


if __name__ == "__main__":
    main()
