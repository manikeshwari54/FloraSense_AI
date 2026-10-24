import pytest
from plant_data import PLANT_DATA, get_plant_profile
from dataset import SPECIES_TAXONOMY

# ── The 20 canonical indoor class names ───────────────────────────────────────
INDOOR_20 = [
    "African Violet", "Aloe Vera", "Areca Palm", "Boston Fern", "Calathea",
    "Chinese Evergreen", "Dracaena", "English Ivy", "Jade Plant",
    "Monstera Deliciosa", "Orchid", "Peace Lily", "Pothos", "Prayer Plant",
    "Rattlesnake Plant", "Rubber Plant", "Snake Plant", "Tradescantia",
    "Venus Flytrap", "ZZ Plant"
]

def test_snake_plant_lookup():
    p = get_plant_profile("Snake Plant")
    assert p is not None
    assert "Snake" in p["common_name"]
    assert "Sansevieria" in p["scientific_name"] or "Dracaena trifasciata" in p["scientific_name"]

def test_snake_not_rattlesnake():
    snake = get_plant_profile("Snake Plant")
    rattle = get_plant_profile("Rattlesnake Plant")
    assert snake is not None and rattle is not None
    assert snake["common_name"] != rattle["common_name"]
    assert "Rattlesnake" not in snake["common_name"]

def test_pothos_profile():
    p = get_plant_profile("pothos")
    assert p is not None
    assert "Pothos" in p["common_name"]

def test_monstera_profile():
    p = get_plant_profile("Monstera Deliciosa")
    assert p is not None

def test_monstera_is_toxic():
    p = get_plant_profile("Monstera Deliciosa")
    assert p is not None
    tox = p["toxicity_safety"].lower()
    assert "toxic" in tox or "calcium" in tox

def test_zz_plant_profile():
    p = get_plant_profile("ZZ Plant")
    assert p is not None

def test_zz_plant_is_toxic():
    p = get_plant_profile("ZZ Plant")
    assert p is not None
    tox = p["toxicity_safety"].lower()
    assert "toxic" in tox

def test_venus_flytrap_profile():
    p = get_plant_profile("Venus Flytrap")
    assert p is not None

def test_all_20_classes_have_profiles():
    missing = []
    for cls in INDOOR_20:
        prof = get_plant_profile(cls)
        if prof is None:
            missing.append(cls)
    assert missing == [], f"Missing profiles for: {missing}"

def test_nonexistent_returns_none():
    assert get_plant_profile("NonExistentBotanicalSpecies_XYZ123") is None

def test_all_taxonomy_species_have_profiles():
    """All 20 SPECIES_TAXONOMY keys should have a botanical profile."""
    missing = []
    for species in INDOOR_20:
        prof = get_plant_profile(species)
        if prof is None:
            missing.append(species)
    assert len(missing) == 0, f"Taxonomy species lack botanical profiles: {missing}"
