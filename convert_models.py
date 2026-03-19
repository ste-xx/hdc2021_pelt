#!/usr/bin/env python3
"""Convert MSD network HDF5 model files to JSON format for in-browser use.

Run this script once to generate the model files needed by index.html:

    python convert_models.py

The script reads the trained networks from trainednets/ and writes
JSON files to models/category{N}/scale{M}.json.

Each JSON file contains all network weights encoded as base64 float32
binary strings, plus the 20x-downscaled input mean image for the
corresponding blur category (included in scale20.json only).
"""

import base64
import json
import os
from pathlib import Path

import h5py
import numpy as np
import skimage.transform as st


def encode_f32(arr):
    """Encode a numpy array as a base64-encoded float32 binary string."""
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def convert_network(h5_path):
    """Read an MSD network from HDF5 and return it as a JSON-serialisable dict."""
    with h5py.File(h5_path, "r") as fh:
        net = fh["network"]
        d = int(net.attrs["d"])
        nin = int(net.attrs["nin"])
        nout = int(net.attrs["nout"])

        dl = net["dl"][()].tolist()
        gam_in = net["gam_in"][()].tolist()
        off_in = net["off_in"][()].tolist()
        gam_out = net["gam_out"][()].tolist()
        off_out = net["off_out"][()].tolist()
        oo = float(net["oo"][()].flat[0])

        # Flatten (nout, nin+d) -> 1-D
        w = encode_f32(net["w"][()].flatten())
        # Per-layer biases (d,)
        o = encode_f32(net["o"][()])

        # Concatenate all filter weights in layer order.
        # Filter for layer i has shape (nin+i, 3, 3).
        all_filters = []
        for i in range(d):
            key = f"{i:05d}"
            filt = net["f"][key][()]  # (nin+i, 3, 3)
            all_filters.append(filt.flatten())
        f_data = encode_f32(np.concatenate(all_filters))

    return {
        "d": d,
        "nin": nin,
        "nout": nout,
        "dl": dl,
        "gam_in": gam_in,
        "off_in": off_in,
        "gam_out": gam_out,
        "off_out": off_out,
        "oo": oo,
        "w": w,
        "o": o,
        "f": f_data,
    }


def load_mean_downscaled(cat):
    """Load the 16-bit input mean PNG and return a 20x downscaled float32 array."""
    import imageio.v2 as imageio

    mean_path = Path("trainednets/inputmeans") / f"category{cat}.png"
    mnim = imageio.imread(str(mean_path)).astype(np.float32)  # (H, W) uint16 values
    # 20x average-pool downscale
    mnim20 = st.downscale_local_mean(mnim, (20, 20))
    return mnim20.astype(np.float32)


def main():
    base = Path("trainednets")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    scales = [
        ("regr_params20.h5", "scale20"),
        ("regr_params10.h5", "scale10"),
        ("regr_params4.h5", "scale4"),
        ("regr_params2.h5", "scale2"),
        ("regr_paramsf.h5", "scalef"),
    ]

    for cat in range(20):
        cat_dir = base / f"category{cat}"
        out_dir = models_dir / f"category{cat}"
        out_dir.mkdir(exist_ok=True)

        print(f"Category {cat}:")

        # Load and embed the 20x-downscaled input mean in scale20.json
        print("  Loading input mean ...", end="", flush=True)
        mean20 = load_mean_downscaled(cat)
        mean_shape = list(mean20.shape)  # [rows, cols]
        mean_b64 = encode_f32(mean20.flatten())
        print(f" shape={mean_shape}")

        for h5_name, json_name in scales:
            h5_path = cat_dir / h5_name
            out_path = out_dir / f"{json_name}.json"
            print(f"  {h5_path} -> {out_path}", end="", flush=True)

            model_data = convert_network(h5_path)

            # Embed the downscaled mean only in scale20 (used first in the pipeline)
            if json_name == "scale20":
                model_data["mean20_shape"] = mean_shape
                model_data["mean20"] = mean_b64

            with open(out_path, "w") as jf:
                json.dump(model_data, jf, separators=(",", ":"))

            size_kb = os.path.getsize(out_path) // 1024
            print(f" ({size_kb} KB)")

    print("Done! Model files written to models/")


if __name__ == "__main__":
    main()
