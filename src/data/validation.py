import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image

TECNALIA_REQUIRED_COLUMNS = [
    "Fecha",
    "Vmpp(V)",
    "Impp(A)",
    "Temp. Mod (°C)",
    "GHI (W/m²)",
    "DHI (W/m²)",
    "DNI (W/m²)",
    "Wind Speed (m/s)",
    "Isc(A)",
    "Voc(A)",
    "FF",
    "Front GPOA (W/m²)",
    "Amb. Temp. (°C)",
    "Pmpp (W)",
    "Vac(V)",
    "Iac(A)",
    "Pac(W)",
    "Back GPOA (W/m²)",
    "SR (Wh/m²)",
    "Pressure (mmHg)",
    "Humidity(%)",
    "Rain Accumulation (mm)",
    "Intensity(mm/h)",
    "Direction of wind",
]

TECNALIA_MODULES = [
    "data_Atersa",
    "data_JaSolar3",
    "data_NingboSolar",
    "data_Photowatt",
    "data_TrinaSolar",
]


def validate_tecnalia_schema(raw_dir: str | Path) -> dict:
    """Validate the schema of all TECNALIA module CSV files."""

    raw_dir = Path(raw_dir) / "TECNALIA"
    results = {}

    for module in TECNALIA_MODULES:
        csv_path = raw_dir / module / f"{module}.csv"

        if not csv_path.exists():
            results[module] = {
                "status": "FAIL",
                "reason": "File not found",
            }
            continue

        df = pd.read_csv(csv_path, sep=";")

        missing_columns = [
            column
            for column in TECNALIA_REQUIRED_COLUMNS
            if column not in df.columns
        ]

        unexpected_columns = [
            column
            for column in df.columns
            if column not in TECNALIA_REQUIRED_COLUMNS
        ]

        if missing_columns:
            status = "FAIL"
        elif unexpected_columns:
            status = "WARNING"
        else:
            status = "PASS"

        results[module] = {
            "status": status,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
        }

    return results

def validate_tecnalia_timestamps(raw_dir: str | Path) -> dict:
    """Validate timestamp integrity for all TECNALIA module CSV files."""

    raw_dir = Path(raw_dir) / "TECNALIA"
    results = {}

    for module in TECNALIA_MODULES:
        csv_path = raw_dir / module / f"{module}.csv"

        if not csv_path.exists():
            results[module] = {
                "status": "FAIL",
                "reason": "File not found",
            }
            continue

        df = pd.read_csv(csv_path, sep=";")

        timestamps = pd.to_datetime(df["Fecha"], errors="coerce")

        invalid_timestamps = int(timestamps.isna().sum())
        duplicate_timestamps = int(timestamps.duplicated().sum())
        sorted_timestamps = bool(timestamps.is_monotonic_increasing)

        valid_timestamps = timestamps.dropna()

        if len(valid_timestamps) > 1:
            gaps = valid_timestamps.diff().dropna()

            min_gap_minutes = gaps.min().total_seconds() / 60
            max_gap_minutes = gaps.max().total_seconds() / 60

            gap_counts = (
                gaps.dt.total_seconds()
                .div(60)
                .value_counts()
                .sort_index()
                .to_dict()
            )
        else:
            min_gap_minutes = None
            max_gap_minutes = None
            gap_counts = {}

        status = (
            "PASS"
            if (
                invalid_timestamps == 0
                and duplicate_timestamps == 0
                and sorted_timestamps
            )
            else "FAIL"
        )

        results[module] = {
            "status": status,
            "invalid_timestamps": invalid_timestamps,
            "duplicate_timestamps": duplicate_timestamps,
            "sorted": sorted_timestamps,
            "start": (
                valid_timestamps.min().isoformat()
                if len(valid_timestamps)
                else None
            ),
            "end": (
                valid_timestamps.max().isoformat()
                if len(valid_timestamps)
                else None
            ),
            "min_gap_minutes": min_gap_minutes,
            "max_gap_minutes": max_gap_minutes,
            "gap_distribution_minutes": gap_counts,
        }

    return results

def validate_tecnalia_missingness(raw_dir: str | Path) -> dict:
    """Report missing-value statistics for all TECNALIA module CSV files."""

    raw_dir = Path(raw_dir) / "TECNALIA"
    results = {}

    for module in TECNALIA_MODULES:
        csv_path = raw_dir / module / f"{module}.csv"

        if not csv_path.exists():
            results[module] = {
                "status": "FAIL",
                "reason": "File not found",
            }
            continue

        df = pd.read_csv(csv_path, sep=";")

        missing_counts = df.isna().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        completely_missing = [
            column
            for column in df.columns
            if missing_percentages[column] == 100.0
        ]

        partially_missing = {
            column: round(float(missing_percentages[column]), 4)
            for column in df.columns
            if 0.0 < missing_percentages[column] < 100.0
        }

        no_missing = [
            column
            for column in df.columns
            if missing_percentages[column] == 0.0
        ]

        results[module] = {
            "status": "PASS",
            "rows": len(df),
            "total_missing_values": int(missing_counts.sum()),
            "completely_missing_columns": completely_missing,
            "partially_missing_columns": partially_missing,
            "complete_columns": no_missing,
        }

    return results

TECNALIA_SENTINEL_VALUES = [-9999]


def validate_tecnalia_sentinels(raw_dir: str | Path) -> dict:
    """Detect known sentinel values in TECNALIA module CSV files."""

    raw_dir = Path(raw_dir) / "TECNALIA"
    results = {}

    for module in TECNALIA_MODULES:
        csv_path = raw_dir / module / f"{module}.csv"

        if not csv_path.exists():
            results[module] = {
                "status": "FAIL",
                "reason": "File not found",
            }
            continue

        df = pd.read_csv(csv_path, sep=";")

        sentinel_counts = {}

        for sentinel in TECNALIA_SENTINEL_VALUES:
            columns_with_sentinel = {}

            for column in df.select_dtypes(include="number").columns:
                count = int((df[column] == sentinel).sum())

                if count > 0:
                    columns_with_sentinel[column] = count

            if columns_with_sentinel:
                sentinel_counts[str(sentinel)] = columns_with_sentinel

        results[module] = {
            "status": "PASS",
            "sentinel_values": TECNALIA_SENTINEL_VALUES,
            "detected_sentinels": sentinel_counts,
        }

    return results

TECNALIA_NON_NEGATIVE_COLUMNS = [
    "Vmpp(V)",
    "Impp(A)",
    "Pmpp (W)",
    "Isc(A)",
    "Voc(A)",
    "FF",
    "Front GPOA (W/m²)",
    "Wind Speed (m/s)",
]


def validate_tecnalia_ranges(raw_dir: str | Path) -> dict:
    """Check conservative physical range rules for TECNALIA data."""

    raw_dir = Path(raw_dir) / "TECNALIA"
    results = {}

    for module in TECNALIA_MODULES:
        csv_path = raw_dir / module / f"{module}.csv"

        if not csv_path.exists():
            results[module] = {
                "status": "FAIL",
                "reason": "File not found",
            }
            continue

        df = pd.read_csv(csv_path, sep=";")

        negative_counts = {}

        for column in TECNALIA_NON_NEGATIVE_COLUMNS:
            if column not in df.columns:
                continue

            count = int((df[column] < 0).sum())

            if count > 0:
                negative_counts[column] = count

        results[module] = {
            "status": "PASS" if not negative_counts else "WARNING",
            "negative_values": negative_counts,
        }

    return results

RAPTORMAPS_EXPECTED_IMAGE_COUNT = 20_000
RAPTORMAPS_EXPECTED_SIZE = (24, 40)
RAPTORMAPS_EXPECTED_MODE = "L"


def validate_raptormaps_integrity(raw_dir: str | Path) -> dict:
    """Validate RaptorMaps metadata and image-file integrity."""

    raw_dir = Path(raw_dir) / "InfraredSolarModules"

    metadata_path = raw_dir / "module_metadata.json"

    if not metadata_path.exists():
        return {
            "status": "FAIL",
            "reason": "module_metadata.json not found",
        }

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    image_records = metadata

    missing_images = []
    unreadable_images = []
    invalid_dimensions = []
    invalid_modes = []

    for record_id, record in image_records.items():
        image_path = raw_dir / record["image_filepath"]

        if not image_path.exists():
            missing_images.append(str(image_path))
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                if image.size != RAPTORMAPS_EXPECTED_SIZE:
                    invalid_dimensions.append(
                        {
                            "path": str(image_path),
                            "size": image.size,
                        }
                    )

                if image.mode != RAPTORMAPS_EXPECTED_MODE:
                    invalid_modes.append(
                        {
                            "path": str(image_path),
                            "mode": image.mode,
                        }
                    )

        except Exception:
            unreadable_images.append(str(image_path))

    status = (
        "PASS"
        if (
            len(image_records) == RAPTORMAPS_EXPECTED_IMAGE_COUNT
            and not missing_images
            and not unreadable_images
            and not invalid_dimensions
            and not invalid_modes
        )
        else "FAIL"
    )

    return {
        "status": status,
        "metadata_records": len(image_records),
        "expected_image_count": RAPTORMAPS_EXPECTED_IMAGE_COUNT,
        "missing_images": len(missing_images),
        "unreadable_images": len(unreadable_images),
        "invalid_dimensions": len(invalid_dimensions),
        "invalid_modes": len(invalid_modes),
    }

RAPTORMAPS_EXPECTED_CLASSES = {
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Shadowing",
    "Diode",
    "Diode-Multi",
    "Vegetation",
    "Soiling",
    "Offline-Module",
    "No-Anomaly",
}


def validate_raptormaps_labels(raw_dir: str | Path) -> dict:
    """Validate RaptorMaps anomaly labels and class distribution."""

    raw_dir = Path(raw_dir) / "InfraredSolarModules"
    metadata_path = raw_dir / "module_metadata.json"

    if not metadata_path.exists():
        return {
            "status": "FAIL",
            "reason": "module_metadata.json not found",
        }

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    missing_labels = []
    invalid_labels = []
    class_counts = {}

    for record_id, record in metadata.items():
        label = record.get("anomaly_class")

        if not label:
            missing_labels.append(record_id)
            continue

        class_counts[label] = class_counts.get(label, 0) + 1

        if label not in RAPTORMAPS_EXPECTED_CLASSES:
            invalid_labels.append(
                {
                    "record_id": record_id,
                    "label": label,
                }
            )

    unexpected_classes = sorted(
        set(class_counts) - RAPTORMAPS_EXPECTED_CLASSES
    )

    missing_classes = sorted(
        RAPTORMAPS_EXPECTED_CLASSES - set(class_counts)
    )

    status = (
        "PASS"
        if (
            len(missing_labels) == 0
            and len(invalid_labels) == 0
            and not unexpected_classes
            and not missing_classes
            and sum(class_counts.values()) == len(metadata)
        )
        else "FAIL"
    )

    return {
        "status": status,
        "metadata_records": len(metadata),
        "label_records": sum(class_counts.values()),
        "missing_labels": len(missing_labels),
        "invalid_labels": len(invalid_labels),
        "unexpected_classes": unexpected_classes,
        "missing_classes": missing_classes,
        "class_counts": dict(sorted(class_counts.items())),
    }

def validate_raptormaps_duplicates(raw_dir: str | Path) -> dict:
    """Detect exact duplicate images and conflicting duplicate labels."""

    raw_dir = Path(raw_dir) / "InfraredSolarModules"
    metadata_path = raw_dir / "module_metadata.json"

    if not metadata_path.exists():
        return {
            "status": "FAIL",
            "reason": "module_metadata.json not found",
        }

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    hash_groups = {}

    for record_id, record in metadata.items():
        image_path = raw_dir / record["image_filepath"]

        if not image_path.exists():
            continue

        with image_path.open("rb") as file:
            image_hash = hashlib.md5(file.read()).hexdigest()

        hash_groups.setdefault(image_hash, []).append(
            {
                "record_id": record_id,
                "label": record["anomaly_class"],
                "image_path": str(image_path),
            }
        )

    duplicate_groups = {
        image_hash: records
        for image_hash, records in hash_groups.items()
        if len(records) > 1
    }

    conflicting_groups = {
        image_hash: records
        for image_hash, records in duplicate_groups.items()
        if len({record["label"] for record in records}) > 1
    }

    duplicate_images = sum(
        len(records)
        for records in duplicate_groups.values()
    )

    conflicting_images = sum(
        len(records)
        for records in conflicting_groups.values()
    )

    return {
        "status": "PASS",
        "metadata_records": len(metadata),
        "unique_image_hashes": len(hash_groups),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": duplicate_images,
        "conflicting_duplicate_groups": len(conflicting_groups),
        "conflicting_duplicate_images": conflicting_images,
        "conflicting_groups": conflicting_groups,
    }

def validate_raptormaps_split(raw_dir: str | Path) -> dict:
    """Validate the RaptorMaps train/validation/test split manifest."""

    raw_dir = Path(raw_dir) / "InfraredSolarModules"
    metadata_path = raw_dir / "module_metadata.json"
    manifest_path = (
        Path("data/processed/raptormaps") / "split_manifest.csv"
    )

    if not metadata_path.exists():
        return {
            "status": "FAIL",
            "reason": "module_metadata.json not found",
        }

    if not manifest_path.exists():
        return {
            "status": "FAIL",
            "reason": "split_manifest.csv not found",
        }

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    manifest = pd.read_csv(manifest_path)

    required_columns = {
        "metadata_id",
        "image_filepath",
        "anomaly_class",
        "split",
        "group_hash",
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        return {
            "status": "FAIL",
            "reason": "Missing manifest columns",
            "missing_columns": sorted(missing_columns),
        }

    valid_splits = {
        "train",
        "validation",
        "test",
        "excluded_conflicting_duplicate",
    }

    invalid_splits = sorted(
        set(manifest["split"].dropna()) - valid_splits
    )

    duplicate_metadata_ids = int(
        manifest["metadata_id"].duplicated().sum()
    )

    metadata_labels = {
        str(record_id): record["anomaly_class"]
        for record_id, record in metadata.items()
    }

    label_mismatches = []

    for _, row in manifest.iterrows():
        metadata_id = str(row["metadata_id"])
        expected_label = metadata_labels.get(metadata_id)

        if expected_label != row["anomaly_class"]:
            label_mismatches.append(metadata_id)

    # A duplicate image group must never span train/validation/test.
    included = manifest[
        manifest["split"].isin({"train", "validation", "test"})
    ]

    cross_split_duplicate_groups = (
        included.groupby("group_hash")["split"]
        .nunique()
    )

    cross_split_duplicate_groups = int(
        (cross_split_duplicate_groups > 1).sum()
    )

    split_counts = manifest["split"].value_counts().to_dict()

    excluded = manifest[
        manifest["split"] == "excluded_conflicting_duplicate"
    ]

    return {
        "status": "PASS",
        "manifest_records": len(manifest),
        "included_records": len(included),
        "excluded_records": len(excluded),
        "duplicate_metadata_ids": duplicate_metadata_ids,
        "invalid_splits": invalid_splits,
        "label_mismatches": label_mismatches,
        "cross_split_duplicate_groups": (
            cross_split_duplicate_groups
        ),
        "split_counts": split_counts,
        "excluded_metadata_ids": sorted(
            excluded["metadata_id"].astype(str).tolist()
        ),
    }

def validate_raptormaps(raw_dir: str | Path) -> dict:
    """Run all RaptorMaps validation checks."""

    return {
        "integrity": validate_raptormaps_integrity(raw_dir),
        "labels": validate_raptormaps_labels(raw_dir),
        "duplicates": validate_raptormaps_duplicates(raw_dir),
        "split": validate_raptormaps_split(raw_dir),
    }