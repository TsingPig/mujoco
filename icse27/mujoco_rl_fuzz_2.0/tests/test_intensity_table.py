"""IntensityTable invariants."""
from src.mutations.intensity import INTENSITY_TABLE
from src.mutations.registry import MUTATOR_IDS, all_mutators


def test_table_has_entry_for_every_mutator():
    for mid in MUTATOR_IDS:
        assert mid in INTENSITY_TABLE, f"missing intensity entry for {mid}"


def test_each_mutator_has_at_least_one_mode():
    for m in all_mutators():
        assert len(m.intensity_modes) >= 1


def test_invalid_modes_are_subset_of_intensity_modes():
    for m in all_mutators():
        for mode in m.invalid_parseable_modes:
            assert mode in m.intensity_modes, \
                f"{m.id}: invalid mode {mode} not in intensity_modes"


def test_table_matches_mutator_modes():
    """Registry-declared modes must equal IntensityTable modes (single source)."""
    for m in all_mutators():
        assert list(m.intensity_modes) == list(INTENSITY_TABLE[m.id]), \
            f"{m.id} intensity_modes mismatch table"
