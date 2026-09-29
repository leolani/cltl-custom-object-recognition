from collections import Counter
from typing import Dict, Set

# Object labels that are considered the same when comparing images, as a VLM uses
# different words for the same object between images. Maps a category to its labels,
# extend as needed. Labels are matched in lower case; labels not listed are their own category.
CATEGORIES = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children",
               "kid", "adult", "guy", "lady", "gentleman", "human"],
    "picture": ["picture", "painting", "poster", "photo", "photograph", "artwork", "print", "drawing",
                "framed picture", "wall art"],
    "shirt": ["shirt", "t-shirt", "tshirt", "tee", "polo shirt", "blouse"],
    "glasses": ["glasses", "eyeglasses", "spectacles"],
    "cup": ["cup", "mug", "coffee cup", "teacup"],
    "sofa": ["sofa", "couch"],
    "television": ["television", "tv", "tv screen"],
    "monitor": ["monitor", "computer screen", "screen", "display"],
    "phone": ["phone", "cell phone", "mobile phone", "smartphone"],
    "table": ["table", "desk"],
    "chair": ["chair", "office chair", "seat"],
    "plant": ["plant", "potted plant", "houseplant"],
    "shoe": ["shoe", "shoes", "slipper", "slippers", "sandal", "sandals", "sneaker", "sneakers", "footwear"],
    "bag": ["bag", "backpack", "handbag"],
    "robe": ["robe", "kimono", "yukata"],
    "belt": ["belt", "obi", "sash"],
}

_CATEGORY_OF: Dict[str, str] = {label: category for category, labels in CATEGORIES.items() for label in labels}


def category(label: str) -> str:
    normalized = label.strip().lower()
    return _CATEGORY_OF.get(normalized, normalized)


def by_category(objects: Counter) -> Counter:
    """Sums the counts of the labels in objects per category."""
    counts = Counter()
    for label, count in objects.items():
        counts[category(label)] += count

    return counts


def labels_by_category(objects: Counter) -> Dict[str, Set[str]]:
    labels = {}
    for label in objects:
        labels.setdefault(category(label), set()).add(label)

    return labels
