#! /home/student/catkin_ws/src/chatbot_lm/.venv python
from enum import Enum
from typing import Optional, Dict


class SweetType(Enum):
    SNICKERS = 1
    MILKYWAY = 2
    MAOAM = 0
    KINDERRIEGEL = 3

    @classmethod
    def from_string(cls, text: str) -> Optional["SweetType"]:
        """Convert a string to a SweetType enum value."""
        normalized = text.lower().replace(" ", "")
        mapping = {
            "snickers": cls.SNICKERS,
            "milkyway": cls.MILKYWAY,
            "milky-way": cls.MILKYWAY,
            "milky_way": cls.MILKYWAY,
            "maoam": cls.MAOAM,
            "kinderriegel": cls.KINDERRIEGEL,
            "kinder-riegel": cls.KINDERRIEGEL,
            "kinder_riegel": cls.KINDERRIEGEL,
        }
        return mapping.get(normalized)

    @classmethod
    def to_display_name(cls, sweet_type: "SweetType") -> str:
        """Convert a SweetType enum value to its display name."""
        display_names = {
            cls.SNICKERS: "Snickers",
            cls.MILKYWAY: "Milky Way",
            cls.MAOAM: "Maoam",
            cls.KINDERRIEGEL: "Kinderriegel",
        }
        return display_names[sweet_type]

    @property
    def display_name(self) -> str:
        """Get the display name for this sweet type."""
        return self.to_display_name(self)

    @classmethod
    def get_all_variants(cls) -> Dict[str, "SweetType"]:
        """Get a dictionary of all possible string variants mapping to their SweetType."""
        variants = {
            "snickers": cls.SNICKERS,
            "milkyway": cls.MILKYWAY,
            "milky way": cls.MILKYWAY,
            "milky-way": cls.MILKYWAY,
            "milky_way": cls.MILKYWAY,
            "maoam": cls.MAOAM,
            "kinderriegel": cls.KINDERRIEGEL,
            "kinder riegel": cls.KINDERRIEGEL,
            "kinder-riegel": cls.KINDERRIEGEL,
            "kinder_riegel": cls.KINDERRIEGEL,
        }
        return variants


# Testing
def main():
    # Convert string to enum
    sweet = SweetType.from_string("Milky Way")
    assert sweet == SweetType.MILKYWAY
    assert sweet.value == 1

    # Get display name
    assert sweet.display_name == "Milky Way"
    assert SweetType.to_display_name(SweetType.KINDERRIEGEL) == "Kinderriegel"

    # Get all variants
    variants = SweetType.get_all_variants()
    assert variants["milky way"] == SweetType.MILKYWAY
    assert variants["kinder-riegel"] == SweetType.KINDERRIEGEL

    # needle in a haystack-ish
    variants = SweetType.get_all_variants()
    sweet_str = "einen Maoam bitte"
    normalized_str = sweet_str.lower()
    for variant, sweet in variants.items():
        if variant.lower() in normalized_str:
            sweet_type = sweet
            break
    assert sweet_type == SweetType.MAOAM

    # needle in a haystack-ish v2
    variants = SweetType.get_all_variants()
    sweet_str = "Ich will einen Snickers."
    normalized_str = sweet_str.lower()
    for variant, sweet in variants.items():
        if variant.lower() in normalized_str:
            sweet_type = sweet
            break
    assert sweet_type == SweetType.SNICKERS

    # needle in a haystack-ish v3
    variants = SweetType.get_all_variants()
    sweet_str = "Loremipsumdolorsitamet,consetetursadipscingelitr,seddiamnonumyeirmodtemporinviduntutlaboreetdoloremagnaaliquyamerat,seddiamvoluptua.Atveroeosetaccusametjustoduodoloresetearebum.Stetclitakasdgubergren,noseatakimatasanctusestLoremipsumdolorsitamet.Loremipsumdolorsitamet,consetetursadipscingelitr,seddiamnonumyeirmodtemporinviduntutlaboreetdoloremagnaaliquyamerat,seddiamvoluptua.Atveroeosetaccusametjustoduodoloresetearebum.SnickersStetclitakasdgubergren,noseatakimatasanctusestLoremipsumdolorsitamet."
    normalized_str = sweet_str.lower()
    for variant, sweet in variants.items():
        if variant.lower() in normalized_str:
            sweet_type = sweet
            break
    assert sweet_type == SweetType.SNICKERS


if __name__ == "__main__":
    main()
