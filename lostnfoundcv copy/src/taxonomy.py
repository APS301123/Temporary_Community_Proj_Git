from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaxonomyMatch:
    group: str
    subgroup: str


KEYWORD_TO_TAXONOMY: list[tuple[str, TaxonomyMatch]] = [
    ("clothing on hanger", TaxonomyMatch("Apparel", "Clothing")),
    ("insulated bottle", TaxonomyMatch("Drinkware", "Insulated Bottle")),
    ("water bottle", TaxonomyMatch("Drinkware", "Water Bottle")),
    ("tumbler", TaxonomyMatch("Drinkware", "Coffee Tumbler")),
    ("mug", TaxonomyMatch("Drinkware", "Mug")),
    ("cup", TaxonomyMatch("Drinkware", "Cup")),
    ("lunch box", TaxonomyMatch("Food Containers", "Lunch Box")),
    ("food container", TaxonomyMatch("Food Containers", "Food Container")),
    ("lunch bag", TaxonomyMatch("Food Containers", "Lunch Bag")),
    ("hoodie", TaxonomyMatch("Apparel", "Hoodie")),
    ("jacket", TaxonomyMatch("Apparel", "Jacket")),
    ("sweater", TaxonomyMatch("Apparel", "Sweater")),
    ("sweatshirt", TaxonomyMatch("Apparel", "Sweater")),
    ("shirt", TaxonomyMatch("Apparel", "Shirt")),
    ("jersey", TaxonomyMatch("Apparel", "Shirt")),
    ("pants", TaxonomyMatch("Apparel", "Pants")),
    ("shorts", TaxonomyMatch("Apparel", "Shorts")),
    ("uniform", TaxonomyMatch("Apparel", "School Uniform")),
    ("clothing", TaxonomyMatch("Apparel", "Clothing")),
    ("garment", TaxonomyMatch("Apparel", "Clothing")),
    ("backpack", TaxonomyMatch("Bags", "Backpack")),
    ("bag", TaxonomyMatch("Bags", "Bag")),
    ("pencil pouch", TaxonomyMatch("School Supplies", "Pencil Pouch")),
    ("notebook", TaxonomyMatch("School Supplies", "Notebook")),
    ("textbook", TaxonomyMatch("School Supplies", "Textbook")),
    ("folder", TaxonomyMatch("School Supplies", "Folder")),
    ("calculator", TaxonomyMatch("School Supplies", "Calculator")),
    ("phone", TaxonomyMatch("Electronics", "Smartphone")),
    ("headphone", TaxonomyMatch("Electronics", "Headphones")),
    ("earbud", TaxonomyMatch("Electronics", "Earbuds")),
    ("charger", TaxonomyMatch("Electronics", "Charger")),
    ("tablet", TaxonomyMatch("Electronics", "Tablet")),
    ("laptop", TaxonomyMatch("Electronics", "Laptop")),
    ("umbrella", TaxonomyMatch("Accessories", "Umbrella")),
    ("hat", TaxonomyMatch("Accessories", "Hat")),
    ("cap", TaxonomyMatch("Accessories", "Cap")),
    ("keys", TaxonomyMatch("Personal Essentials", "Keys")),
    ("wallet", TaxonomyMatch("Personal Essentials", "Wallet")),
    ("glasses", TaxonomyMatch("Personal Essentials", "Eyeglasses")),
    ("shoe", TaxonomyMatch("Footwear", "Shoes")),
    ("sneaker", TaxonomyMatch("Footwear", "Sneakers")),
]


def map_label_to_taxonomy(label: str) -> TaxonomyMatch:
    lowered = label.lower()
    for keyword, match in KEYWORD_TO_TAXONOMY:
        if keyword in lowered:
            return match
    return TaxonomyMatch("Other", "Uncategorized")

