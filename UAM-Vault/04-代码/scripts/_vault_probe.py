from pathlib import Path

v = Path(r"F:/UAM/UAM-Vault")
for name in sorted(v.iterdir()):
    esc = name.name.encode("unicode_escape").decode()
    if not name.is_dir() or name.name == ".obsidian":
        print("FILE", esc)
        continue
    n = sum(1 for p in name.rglob("*") if p.is_file())
    print(f"DIR {esc} files={n}")
