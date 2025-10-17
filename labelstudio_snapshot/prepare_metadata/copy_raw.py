import pathlib
import re
import shutil

language_data: dict[str, dict[str, str | list[str] | int]] = {
    "eng_Latn": {"name": "English", "ccTLDs": ["uk", "us", "au", "nz"], "seed": 42},
    "spa_Latn": {"name": "Spanish", "ccTLDs": ["es"], "seed": 42},
    "rus_Cyrl": {"name": "Russian", "ccTLDs": ["ru"], "seed": 42},
    "cmn_Hans": {"name": "Mandarin Chinese (Simplified)", "ccTLDs": ["cn"], "seed": 42},
    "cat_Latn": {"name": "Catalan", "ccTLDs": ["cat"], "seed": 42},
    "fin_Latn": {"name": "Finnish", "ccTLDs": ["fi"], "seed": 43},
    "ces_Latn": {"name": "Czech", "ccTLDs": ["cz"], "seed": 43},
    "nob_Latn": {"name": "Norwegian Bokmål", "ccTLDs": ["no"], "seed": 43},
    "jpn_Jpan": {"name": "Japanese", "ccTLDs": ["jp"], "seed": 43},
    "eus_Latn": {"name": "Basque", "ccTLDs": ["eus"], "seed": 43},
    "nno_Latn": {"name": "Norwegian Nynorsk", "ccTLDs": ["no"], "seed": 44},
    "deu_Latn": {"name": "German", "ccTLDs": ["de"], "seed": 44},
    "bel_Cyrl": {"name": "Belarusian", "ccTLDs": ["by"], "seed": 44},
    "por_Latn": {"name": "Portuguese", "ccTLDs": ["pt"], "seed": 44},
    "fra_Latn": {"name": "French", "ccTLDs": ["fr"], "seed": 44},
    "cmn_Hant": {"name": "Mandarin Chinese (Traditional)", "ccTLDs": ["cn"], "seed": 45},
    "tur_Latn": {"name": "Turkish", "ccTLDs": ["tr"], "seed": 45},
    "ita_Latn": {"name": "Italian", "ccTLDs": ["it"], "seed": 45},
    "ben_Beng": {"name": "Bengali", "ccTLDs": ["bd"], "seed": 45},
    "slk_Latn": {"name": "Slovak", "ccTLDs": ["sk"], "seed": 45},
    "kor_Hang": {"name": "Korean", "ccTLDs": ["kr"], "seed": 46},
    "pol_Latn": {"name": "Polish", "ccTLDs": ["pl"], "seed": 46},
    "nld_Latn": {"name": "Dutch", "ccTLDs": ["nl"], "seed": 46},
    "slv_Latn": {"name": "Slovenian", "ccTLDs": ["si"], "seed": 46},
    "swe_Latn": {"name": "Swedish", "ccTLDs": ["se"], "seed": 46},
    "srp_Cyrl": {"name": "Serbian", "ccTLDs": ["rs"], "seed": 47},
    "ell_Grek": {"name": "Modern Greek (1453-)", "ccTLDs": ["gr"], "seed": 47},
    "kaz_Cyrl": {"name": "Kazakh", "ccTLDs": ["kz"], "seed": 47},
    "ron_Latn": {"name": "Romanian", "ccTLDs": ["ro"], "seed": 47},
    "vie_Latn": {"name": "Vietnamese", "ccTLDs": ["vn"], "seed": 47},
    "hin_Deva": {"name": "Hindi", "ccTLDs": ["in"], "seed": 48},
    "heb_Hebr": {"name": "Hebrew", "ccTLDs": ["il"], "seed": 48},
    "tha_Thai": {"name": "Thai", "ccTLDs": ["th"], "seed": 48},
    "arb_Arab": {"name": "Modern Standard Arabic", "ccTLDs": ["eg", "dz", "iq"], "seed": 40000},
    "arz_Arab": {"name": "Egyptian Arabic", "ccTLDs": ["eg"], "seed": 41000},
}


def copy_raw_sample(path_to_dir: str | pathlib.Path, dest_dir: str | pathlib.Path) -> None:
    """
    Copies parquet files from a source to a destination, organizing them by language.

    This function operates in a strict mode: it will raise an error and halt if:
    1. A file's name does not contain a parsable seed (e.g., '_seed_42').
    2. A language match cannot be found for a file's ccTLD and seed.
    3. A file with the same name already exists in the destination path.
    """
    if not isinstance(path_to_dir, pathlib.Path):
        path_to_dir = pathlib.Path(path_to_dir)

    if not isinstance(dest_dir, pathlib.Path):
        dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    all_parquet_files = list(path_to_dir.rglob("*.parquet"))
    print(f"Found {len(all_parquet_files)} parquet files to process.")

    for file_path in all_parquet_files:
        # Get ccTLD from file path
        cctld = file_path.parent.parent.name

        # Extract seed from filename. Using regex for robustness.
        match = re.search(r"_seed_(\d+)", file_path.name)
        # STRICT: Fail if seed cannot be parsed from the filename.
        if not match:
            msg = f"Could not extract seed from filename: {file_path.name}"
            raise ValueError(msg)
        seed = int(match.group(1))

        # Find language for the given ccTLD and seed
        language = None
        for lang, lang_data in language_data.items():
            if cctld in lang_data["ccTLDs"] and seed == lang_data["seed"]:  # type: ignore[operator]
                language = lang
                break

        # STRICT: Fail if no matching language is found.
        if language is None:
            msg = f"Could not find language for ccTLD '{cctld}' and seed '{seed}' (from file {file_path})"
            raise ValueError(msg)

        # Create destination directory
        language_dir = dest_dir.joinpath(language).joinpath(cctld)
        language_dir.mkdir(parents=True, exist_ok=True)

        # Prepare to copy file to destination directory
        dest_file_path = language_dir.joinpath(file_path.name)

        # STRICT: Fail if the destination file already exists.
        if dest_file_path.exists():
            msg = f"Destination file already exists: {dest_file_path}"
            raise FileExistsError(msg)

        shutil.copy(file_path, dest_file_path)

    # Final verification
    parquet_files_in_dest = list(dest_dir.rglob("*.parquet"))
    print(f"Successfully copied {len(parquet_files_in_dest)} parquet files to {dest_dir}")

    # STRICT: The number of files must match exactly.
    assert len(parquet_files_in_dest) == len(all_parquet_files), (
        f"Mismatch: Number of copied files ({len(parquet_files_in_dest)}) does not "
        f"match number of original files ({len(all_parquet_files)})."
    )
    print("All files copied and verified successfully.")
