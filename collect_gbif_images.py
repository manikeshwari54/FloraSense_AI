"""Collect explicitly licensed GBIF plant media into a review-only staging tree.

The collector never changes the active dataset or model. GBIF occurrence labels
and image contents still require human review before training use.
"""

import argparse
import hashlib
import io
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps


GBIF_API = "https://api.gbif.org/v1"
GBIF_MATCH_API = "https://api.gbif.org/v2/species/match"
GBIF_LICENSE_FILTERS = ("CC0_1_0", "CC_BY_4_0")
LICENSE_URLS = {
    "CC0_1_0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC_BY_4_0": "https://creativecommons.org/licenses/by/4.0/",
}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 60_000_000
MIN_IMAGE_SIDE = 224
PAGE_SIZE = 300
USER_AGENT = "FloraSenseAI-dataset-collector/1.0 (research; contact project maintainer)"

# All names are distinct from the existing FloraSense class names and taxa.
SPECIES = (
    {"label": "Anthurium", "scientific_name": "Anthurium andraeanum"},
    {"label": "Birds Nest Fern", "scientific_name": "Asplenium nidus"},
    {"label": "Chinese Money Plant", "scientific_name": "Pilea peperomioides"},
    {"label": "Thanksgiving Cactus", "scientific_name": "Schlumbergera truncata"},
    {"label": "Dumb Cane", "scientific_name": "Dieffenbachia seguine"},
    {"label": "Elephant Ear", "scientific_name": "Alocasia macrorrhizos"},
    {"label": "Money Tree", "scientific_name": "Pachira aquatica"},
    {"label": "Parlor Palm", "scientific_name": "Chamaedorea elegans"},
    {"label": "Poinsettia", "scientific_name": "Euphorbia pulcherrima"},
    {"label": "Polka Dot Plant", "scientific_name": "Hypoestes phyllostachya"},
    {"label": "Ponytail Palm", "scientific_name": "Beaucarnea recurvata"},
    {"label": "Dwarf Umbrella Tree", "scientific_name": "Heptapleurum arboricola"},
    {"label": "Heartleaf Philodendron", "scientific_name": "Philodendron hederaceum"},
    {"label": "Baby Rubber Plant", "scientific_name": "Peperomia obtusifolia"},
    {"label": "Spider Plant", "scientific_name": "Chlorophytum comosum"},
)


def request_json(url, params=None):
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=(15, 60),
    )
    response.raise_for_status()
    return response.json()


def match_species(scientific_name):
    result = request_json(
        GBIF_MATCH_API,
        {"scientificName": scientific_name, "strict": "true"},
    )
    usage = result.get("usage") or {}
    diagnostics = result.get("diagnostics") or {}
    if (
        diagnostics.get("matchType") != "EXACT"
        or usage.get("rank") != "SPECIES"
        or not usage.get("key")
        or usage.get("status") not in (None, "ACCEPTED")
    ):
        raise ValueError(
            f"GBIF did not return an exact accepted species match for {scientific_name}: "
            f"{diagnostics.get('matchType')} / {usage.get('rank')} / {usage.get('status')}"
        )
    return usage


def is_allowed_license(value):
    if not value:
        return None
    normalized = value.strip().rstrip("/").lower()
    if normalized.upper() in LICENSE_URLS:
        return normalized.upper()
    if normalized.startswith(("http://", "https://")):
        normalized = "https://" + normalized.split("://", 1)[1]
        for license_code, license_url in LICENSE_URLS.items():
            accepted_urls = {
                license_url.rstrip("/").lower(),
                (license_url + "legalcode").rstrip("/").lower(),
            }
            if normalized in accepted_urls:
                return license_code
    return None


def image_url_is_http(url):
    return urlparse(url).scheme.lower() in {"http", "https"}


def download_image(url):
    if not image_url_is_http(url):
        raise ValueError("Media URL must use HTTP or HTTPS")
    with requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=(15, 60),
        stream=True,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the configured 25 MiB limit")
        content = bytearray()
        for chunk in response.iter_content(64 * 1024):
            content.extend(chunk)
            if len(content) > MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds the configured 25 MiB limit")

    with Image.open(io.BytesIO(content)) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image exceeds the configured {MAX_IMAGE_PIXELS:,}-pixel limit"
            )
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if min(width, height) < MIN_IMAGE_SIDE:
            raise ValueError(f"Image is too small: {width}x{height}")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
    return buffer.getvalue(), width, height


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_existing_hashes(output_dir):
    hashes = {}
    if not output_dir.exists():
        return hashes
    for image_path in output_dir.rglob("*.jpg"):
        try:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            hashes[digest] = image_path.relative_to(output_dir).as_posix()
        except OSError:
            continue
    return hashes


def iter_occurrences(taxon_key, max_records):
    offset = 0
    base_params = [
        ("taxonKey", str(taxon_key)),
        ("mediaType", "StillImage"),
        ("license", "CC0_1_0"),
        ("license", "CC_BY_4_0"),
    ]
    while offset < max_records:
        page_limit = min(PAGE_SIZE, max_records - offset)
        page_params = base_params + [
            ("limit", str(page_limit)),
            ("offset", str(offset)),
        ]
        page = request_json(f"{GBIF_API}/occurrence/search", page_params)
        results = page.get("results", [])
        if not results:
            break
        yield from results
        offset += len(results)
        if page.get("endOfRecords"):
            break
        time.sleep(0.15)


def build_manifest(output_dir, target, max_records, minimum_side):
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = load_existing_hashes(output_dir)
    manifest = {
        "source": "GBIF occurrence API and linked media",
        "source_url": "https://www.gbif.org/",
        "api_url": GBIF_API,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_policy": {
            "accepted_media_licenses": LICENSE_URLS,
            "media_without_explicit_accepted_license": "rejected",
            "note": "Only individual media with explicit CC0 1.0 or CC BY 4.0 are saved. "
            "CC BY requires attribution and retention of license terms.",
        },
        "quality_policy": {
            "minimum_short_side_px": minimum_side,
            "corrupt_images": "rejected",
            "exact_duplicates": "rejected globally across collected classes",
        },
        "review_note": "GBIF taxonomic records do not verify that the linked photo depicts "
        "the named plant. Every saved image is pending human label and suitability review; "
        "this staging tree must not be used for training as-is.",
        "species": [],
    }
    failures = []

    for species in SPECIES:
        summary = {
            **species,
            "status": "started",
            "target_images": target,
            "downloaded_images": 0,
            "records_examined": 0,
            "rejected": Counter(),
            "image_files": [],
        }
        try:
            usage = match_species(species["scientific_name"])
            taxon_key = int(usage["key"])
            summary.update({
                "gbif_taxon_key": taxon_key,
                "gbif_accepted_name": usage.get("name"),
                "gbif_match_url": f"https://www.gbif.org/species/{taxon_key}",
            })
            class_dir = output_dir / species["label"]
            class_dir.mkdir(parents=True, exist_ok=True)
            image_index = len(summary["image_files"])

            for occurrence in iter_occurrences(taxon_key, max_records):
                summary["records_examined"] += 1
                if occurrence.get("acceptedTaxonKey") != taxon_key:
                    summary["rejected"]["not_exact_accepted_taxon"] += 1
                    continue

                for media_index, media in enumerate(occurrence.get("media") or []):
                    if media.get("type") != "StillImage":
                        summary["rejected"]["not_still_image"] += 1
                        continue
                    media_license = is_allowed_license(media.get("license"))
                    if not media_license:
                        summary["rejected"]["media_license_not_allowed_or_missing"] += 1
                        continue
                    image_url = media.get("identifier")
                    if not image_url:
                        summary["rejected"]["missing_image_url"] += 1
                        continue
                    if summary["downloaded_images"] >= target:
                        break
                    try:
                        image_data, width, height = download_image(image_url)
                    except Exception as error:
                        summary["rejected"]["download_or_image_validation_error"] += 1
                        summary.setdefault("errors", []).append({
                            "gbif_id": occurrence.get("key"),
                            "media_url": image_url,
                            "error": str(error),
                        })
                        continue

                    digest = sha256(image_data)
                    if digest in seen_hashes:
                        summary["rejected"]["exact_duplicate"] += 1
                        continue

                    image_index += 1
                    filename = f"gbif_{occurrence['key']}_{media_index}_{image_index:05d}.jpg"
                    image_path = class_dir / filename
                    image_path.write_bytes(image_data)
                    relative_path = image_path.relative_to(output_dir).as_posix()
                    seen_hashes[digest] = relative_path
                    summary["image_files"].append({
                        "path": relative_path,
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "label_review": "pending",
                        "suitability_review": "pending",
                        "split": None,
                        "source": "GBIF occurrence API",
                        "gbif_id": occurrence.get("key"),
                        "occurrence_url": f"https://www.gbif.org/occurrence/{occurrence.get('key')}",
                        "scientific_name": occurrence.get("acceptedScientificName"),
                        "dataset_key": occurrence.get("datasetKey"),
                        "dataset_name": occurrence.get("datasetName"),
                        "record_license": occurrence.get("license"),
                        "media_url": image_url,
                        "media_reference": media.get("references"),
                        "license_code": media_license,
                        "license_url": LICENSE_URLS[media_license],
                        "creator": media.get("creator"),
                        "rights_holder": media.get("rightsHolder"),
                        "publisher": media.get("publisher"),
                    })
                    summary["downloaded_images"] += 1
                if summary["downloaded_images"] >= target:
                    break

            summary["status"] = "collected" if summary["downloaded_images"] else "no_usable_media"
        except Exception as error:
            summary["status"] = "error"
            summary["error"] = str(error)
            failures.append(species["label"])

        summary["rejected"] = dict(summary["rejected"])
        manifest["species"].append(summary)
        write_manifest(output_dir, manifest)
        print(
            f"{species['label']}: {summary['downloaded_images']} images; "
            f"{summary['records_examined']} occurrence records examined; {summary['status']}"
        )

    return manifest, failures


def write_manifest(output_dir, manifest):
    manifest_path = output_dir / "source_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)


def inventory(max_records):
    rows = []
    for species in SPECIES:
        row = {**species, "gbif_taxon_key": None, "match": None, "eligible_occurrence_records": None}
        try:
            usage = match_species(species["scientific_name"])
            taxon_key = int(usage["key"])
            params = [
                ("taxonKey", str(taxon_key)),
                ("mediaType", "StillImage"),
                ("license", "CC0_1_0"),
                ("license", "CC_BY_4_0"),
                ("limit", "0"),
            ]
            result = request_json(f"{GBIF_API}/occurrence/search", params)
            row.update({
                "gbif_taxon_key": taxon_key,
                "accepted_name": usage.get("name"),
                "match": "exact species",
                "eligible_occurrence_records": result.get("count", 0),
                "count_kind": "GBIF occurrence records, not images or human-verified usable files",
                "query_limit": max_records,
            })
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)
        print(
            f"{row['label']}: {row.get('eligible_occurrence_records', 'unavailable')} "
            "license-filtered occurrence records (not image counts)"
        )
    return {
        "source": "GBIF occurrence API",
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_record_licenses": list(GBIF_LICENSE_FILTERS),
        "image_license_check": "not included in occurrence-record count; checked per media during collection",
        "species": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="./data_gbif_staging")
    parser.add_argument("--target-images", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=100000)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--inventory-output", default="gbif_candidate_inventory.json")
    args = parser.parse_args()
    if args.target_images < 1 or args.max_records < 1:
        parser.error("--target-images and --max-records must be positive")

    if args.inventory_only:
        report = inventory(args.max_records)
        Path(args.inventory_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Candidate inventory saved to {args.inventory_output}")
        return

    manifest, failures = build_manifest(
        Path(args.output_dir), args.target_images, args.max_records, MIN_IMAGE_SIDE
    )
    total = sum(item["downloaded_images"] for item in manifest["species"])
    print(f"Staged {total} images across {len(manifest['species'])} candidate species.")
    print(f"Per-file provenance: {Path(args.output_dir) / 'source_manifest.json'}")
    print("All staged files are pending human review and are not ready for training.")
    if failures:
        print(f"Species with API or taxonomy errors: {', '.join(failures)}")


if __name__ == "__main__":
    main()