"""Convert aasharma90/RetinexNet_PyTorch DecomNet checkpoint keys to the
layout expected by `ELIR.models.decomposers.DecomNet`.

Usage:
    python third_party/retinexnet/convert_weights.py \\
        --src path/to/raw_decomnet.pth \\
        --dst third_party/retinexnet/decom_net.pth

The source checkpoint can be:
  (a) a torch state_dict where the DecomNet weights are named with a
      "DecomNet." or "module.DecomNet." prefix, or
  (b) a state_dict directly of the DecomNet with keys like
      "net1_conv0.weight", "net1_convs.0.weight", ..., "net1_recon.weight".
The script strips known prefixes and writes the cleaned state_dict to `dst`.
"""
import argparse
import torch


KNOWN_PREFIXES = ("module.DecomNet.", "DecomNet.", "module.", "decom_net.", "net.")


def strip_prefix(k: str) -> str:
    for p in KNOWN_PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    raw = torch.load(args.src, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]

    cleaned = {}
    for k, v in raw.items():
        nk = strip_prefix(k)
        # keep only DecomNet-related params
        if nk.startswith(("net1_conv0", "net1_convs", "net1_recon")):
            cleaned[nk] = v

    if not cleaned:
        raise RuntimeError(
            "No DecomNet keys recognized in source checkpoint. "
            "Keys found: {}".format(list(raw.keys())[:10])
        )

    torch.save(cleaned, args.dst)
    print("wrote {} keys -> {}".format(len(cleaned), args.dst))


if __name__ == "__main__":
    main()
