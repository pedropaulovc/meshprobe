"""Regenerate the pinned CC0 curated-source and assembly catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMIT = "56db2d4088512531a070d0bf3eb9d284d077528d"
ROOT = f"https://raw.githubusercontent.com/ToxSam/cc0-models-Polygonal-Mind/{COMMIT}"
LICENSE = f"https://github.com/ToxSam/cc0-models-Polygonal-Mind/blob/{COMMIT}/License.md"
ATTRIBUTION = "Polygonal Mind Open Source Initiative; GLB conversion by Sam Hamilton"

SOURCES = (
    (
        "trash-polka-021",
        "PoapMachine",
        "trash-polka/PoapMachine.glb",
        "ce5f351d72b8949552d8eb0c0fbf75e74926199002eaf92c04888c6e9dd2d5c9",
        "087bfb43f2848ce7b8e101c8ec9a2e8fe83663b04d81b60e01cd1441e720986a",
    ),
    (
        "transit-004",
        "Pipe_01_Art",
        "transit/Pipe_01_Art.glb",
        "6116540fd3d1df2207c40f019beb9fd8cd84004a2d45c5ce6b4d5ef98ccb3342",
        "249866456b05995863a5b4037526434feee82bdc7628cb758249bad85e383d1f",
    ),
    (
        "transit-014",
        "Tower_Station_CallerButton_Art",
        "transit/Tower_Station_CallerButton_Art.glb",
        "d17291f9ddaaa95bfe16c73999e010a676dad6713dc793eff375d7b4ba00c67b",
        "796352bd44fac026099d1aa69893f477266dd73fb7e621413d96ffee583ebfac",
    ),
    (
        "transit-015",
        "Tower_Station_Elevator_Art",
        "transit/Tower_Station_Elevator_Art.glb",
        "3880944012db8ca8fd8547b332a0271582f72d068563a46f607a79c7fd54b495",
        "961dbc58f16cc218bdbbf2d9279cd69fa1a6840748c93baf742751723fae148b",
    ),
    (
        "transit-017",
        "Tower_Station_Light_Art",
        "transit/Tower_Station_Light_Art.glb",
        "b7e696861bc74a28ca8af50a3c0e5ec41e2ea77b76541bf9274b93732c581980",
        "7ed95e69b93c2df1e78fb3cf813ae5c2bb84aeb5a4f267528956181a16330ae9",
    ),
    (
        "transit-019",
        "Tower_Station_Telephone_Art",
        "transit/Tower_Station_Telephone_Art.glb",
        "0b75da68ea04e850ac01c08d7bdaa885b5fb24a60639f2dc8702ef74e2548ff6",
        "a9e2d955f648d763a3a793578556f598f33c80a7bdc7324691ddb62a3b740dcd",
    ),
    (
        "transit-021",
        "Train_01_Art",
        "transit/Train_01_Art.glb",
        "f0c7d1ee7d1a5fce981468a2c28f6dd44d0232b70dadeda88debd49957aa1d6a",
        "9899a8ae0277a7ba5cd5a252450aed06cd7d9794ceb124f8c2346b8d59c041af",
    ),
    (
        "transit-022",
        "Train_02_Art",
        "transit/Train_02_Art.glb",
        "a28a3db20c71b493b5541cb7f9f959259290cc423afef094d19a983b832b9ee4",
        "598c2a018094a315e62cc21d707285f5ac211ef09648fae231b4bf605c5863ce",
    ),
    (
        "aero-system-001",
        "Aero_Airship_01",
        "aero-system/Aero_Airship_01.glb",
        "8855249110f2acabf69dd6e82061a01feea35e11c615f68e56d6614265ed91a2",
        "1bb3a2e5fa588fd77897f82c217940903ef4260b5cc9b24cb094975fcaecc149",
    ),
    (
        "aero-system-002",
        "Aero_Door_01",
        "aero-system/Aero_Door_01.glb",
        "ef486ad71b5d6819630f15f1292ce9f3efc0eff44b71d11c7dd4421e33f6c03c",
        "5e167d8f5fa6d79033fac90505dda7420c09d08d88eb8b02d9af447d246db988",
    ),
    (
        "aero-system-006",
        "Aero_Lampost_01",
        "aero-system/Aero_Lampost_01.glb",
        "582a8ee73324def1b5c855f3b0ff7322649de26fb11b8bace274959db22df4ea",
        "e04533ce2d698ef3c9a55b9640af7d86c567dc28b52a0e0e6a6772d302ef11b0",
    ),
    (
        "aero-system-007",
        "Aero_Station_01_Art",
        "aero-system/Aero_Station_01_Art.glb",
        "ed90cad03c90ee7a40c87d27c7e2d6e60a7b1ea15006096db82ed630fa0665d5",
        "b347a33e199591e8d983e6838016ad3cc1205488b6547d53f440ef2327084f56",
    ),
    (
        "crystal-crossroads-049",
        "SciFi_Machine",
        "crystal-crossroads/SciFi_Machine.glb",
        "2a9248c233835e376900da3ea785472ce56dea9af7a0aa7942e594f99149a424",
        "57cf1ab39416c1f8cf5a8c8acbf59331d174adfda108b3942abb9f34f4e79f2b",
    ),
    (
        "medieval-fair-011",
        "Cart",
        "medieval-fair/Cart.glb",
        "cb48070abb0689307030cef9e593e4522a34972547caa058a83238df475d078b",
        "5422d2ce2d2f518837c734ce677eb96b5f9dae2e27a69dba0e16d8af87af1715",
    ),
    (
        "towers-024",
        "Colony_Rocket_Art",
        "towers/Colony_Rocket_Art.glb",
        "c6436cb3ffacdca48c60e03d58d93758a84f7a74d539a6dbef8bb725752d6d86",
        "04c3e1dde05c2cbc8c912852866ef85fe280a19c420261e95d9f8d5bf5090632",
    ),
    (
        "towers-030",
        "Colony_UFO_Art",
        "towers/Colony_UFO_Art.glb",
        "162ba6124c7f3e677b44164ded9b2b79665e704481230de8ec5b4c74fba50c08",
        "67d7e4f812dfa106af07e0e13092e55b7738c9df416f51ea1966c9e242ce4657",
    ),
    (
        "towers-031",
        "ControlPoint_Art",
        "towers/ControlPoint_Art.glb",
        "1ee5a9ade2486888933cc930f6f8f8cf96470fb1eed29907fa3bb0e461db73bb",
        "1aa519329f06b244fba21929d66b98aa068bdc16c14852f19c12c2bfe248cda6",
    ),
    (
        "towers-049",
        "LoveDeath_MineCart_Art",
        "towers/LoveDeath_MineCart_Art.glb",
        "90944436c94cc5f2f1d8d856b6a63de0e4198ef591e28ca19177b42cfa37e0fc",
        "8c857874f8883cb813c5f64e5ce809716cf2e7b1ead72d284bf7b929a1b32ec3",
    ),
    (
        "towers-050",
        "LoveDeath_MineWheel_Art",
        "towers/LoveDeath_MineWheel_Art.glb",
        "a625973f38794a94c5ed476972dc29efff9d53dca5f76a1effd597e075244bd1",
        "49b2193405709fbc72e98260de2e786600b82e38fbf8ce2e84399d99e5101887",
    ),
    (
        "towers-068",
        "MemeFactory_Terminal_Art",
        "towers/MemeFactory_Terminal_Art.glb",
        "a32d1e0474a6dfd137d57d9983cfc990f7a0858ddb1dbfcf71f89bb37a8ba055",
        "872c4433a5ab40278a9d827a2da8a4a1d57afa6540815035e625560380fc2661",
    ),
)


def build_catalog() -> dict[str, object]:
    sources = [
        {
            "source_id": source_id,
            "name": name,
            "download_url": f"{ROOT}/projects/{relative_path}",
            "source_commit": COMMIT,
            "source_sha256": source_hash,
            "topology_sha256": topology_hash,
            "license_spdx": "CC0-1.0",
            "license_url": LICENSE,
            "attribution": ATTRIBUTION,
        }
        for source_id, name, relative_path, source_hash, topology_hash in SOURCES
    ]
    assemblies = []
    for index, (source_id, name, *_rest) in enumerate(SOURCES):
        companion = SOURCES[(index + 7) % len(SOURCES)]
        blocker = SOURCES[(index + 13) % len(SOURCES)]
        assemblies.append(
            {
                "assembly_id": f"curated-{index + 1:02d}-{source_id}",
                "description": f"Inspection scene centered on {name}",
                "placements": [
                    {"source_id": source_id, "role": "target", "position_mm": [0, 0, 0]},
                    {"source_id": companion[0], "role": "reference", "position_mm": [2200, 350, 0]},
                    {"source_id": blocker[0], "role": "occluder", "position_mm": [300, -900, 150]},
                ],
            }
        )
    return {
        "schema_version": 1,
        "catalog_version": "polygonal-mind-cc0-v1",
        "sources": sources,
        "assemblies": assemblies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_catalog(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
