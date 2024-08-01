from pathlib import Path
import svgwrite
import os
import cairosvg
import random
import textwrap
from typing import Callable, List, Dict, Optional, Tuple
from constraint import Constraint, Problem, FunctionConstraint


CATS = ["Ginger", "Duchess", "Mr. Mittens", "Tom Cat", "Sassy", "Pip Squeak"]
CRIME_ITEMS = [
    "bird cage",
    "coffee",
    "shoes",
    "fish bowl",
    "ball of yarn",
    "plant",
]
OTHER_ITEMS = [
    ["bird cage", "scratch marks"],
    ["bell ball", "paw print"],
    ["catnip", "scratch marks"],
    ["bell ball", "sock"],
    ["mouse", "paw print"],
    ["sock", "catnip"],
]
N_POSITIONS = 6
ALL_ITEMS = [[CRIME_ITEMS[i]] + OTHER_ITEMS[i] for i in range(N_POSITIONS)]


POSITIONS = [x for x in range(N_POSITIONS)]

def item_positions(item_name: str) -> list[int]:
    return [i for i, items in enumerate(ALL_ITEMS) if item_name in items]


def is_next_to(pos1: int | None, pos2: int | None) -> bool:
    if pos1 is None or pos2 is None:
        return False
    diff = abs(pos1 - pos2)
    return diff in [1, 5]


def is_next_to_one_of(pos: int, poses) -> bool:
    return any(is_next_to(pos, p) for p in poses)


def is_across_from(pos1: int | None, pos2: int | None) -> bool:
    if pos1 is None or pos2 is None:
        return False

    pairs = {0: 3, 1: 5, 2: 4, 3: 0, 4: 2, 5: 1}
    return pairs[pos1] == pos2


def is_in_front_of_items(cat_pos: int, items: list[str]) -> bool:
    return set(items).issubset(set(ALL_ITEMS[cat_pos]))


def item_pronoun(item: str) -> str:
    if item in CRIME_ITEMS:
        return "the"
    elif any(item.startswith(le) for le in ['a', 'e', 'i', 'o', 'u']):
        return "an"
    else:
        return "a"
    
def all_different_except_none(*cat_positions: list[str]) -> bool:
    non_none_values = [cp for cp in cat_positions if cp is not None]
    return len(set(non_none_values)) == len(non_none_values)

class Clue:
    
    def slug(self) -> str:
        return self.__class__.slug


def random_awake_cat_names(cat_positions: dict[str, int | None], n: int) -> list[str]:
    awake_cats = [c for c, pos in cat_positions.items() if pos]
    return random.sample(awake_cats, n)

def random_items(n: int) -> list[str]:
    if n == 1:
        choose_from = ALL_ITEMS
    else:
        choose_from = [oi for oi in OTHER_ITEMS if len(oi) > 1]

    items_for_pos = random.choice(choose_from)
    return random.sample(items_for_pos, n)


class NextToCat(Clue):

    slug = "next_to_cat"

    @staticmethod
    def attempt_generate(cat_positions: dict[str, int | None]) -> Optional["NextToCat"]: 
        cat1, cat2 = random_awake_cat_names(cat_positions, 2)
        if is_next_to(cat_positions[cat1], cat_positions[cat2]):
            return NextToCat(cat1, cat2)
        return None

    def __init__(self, cat1_name: str, cat2_name: str):
        self.cat1_name = cat1_name
        self.cat2_name = cat2_name
    
    def __str__(self) -> str:
        return str((self.slug, self.cat1_name, self.cat2_name))
    
    def constraint(self) -> tuple[Constraint, tuple[str]]:
        def next_to(cat1_pos: int | None, cat2_pos: int | None):
            return is_next_to(cat1_pos, cat2_pos)

        return (FunctionConstraint(next_to), (self.cat1_name, self.cat2_name))

    def full_text(self) -> str:
        return f"{self.cat1_name} was sitting next to {self.cat2_name}"
    

class SpecificSideOfCat(Clue):

    slug = "specific_side_of_cat"

    @staticmethod
    def attempt_generate(cat_positions: dict[str, int | None]) -> Optional["NextToCat"]: 
        cat1, cat2 = random_awake_cat_names(cat_positions, 2)
        direction = random.choice(["left", "right"])
        if is_next_to(cat_positions[cat1], cat_positions[cat2]):
            return SpecificSideOfCat(cat1, cat2, direction)
        return None

    def __init__(self, cat1_name: str, cat2_name: str, direction: str):
        # "{cat1} was sitting to {cat2}'s {direction}"
        self.cat1_name = cat1_name
        self.cat2_name = cat2_name
        self.direction = direction
    
    def __str__(self) -> str:
        return str((self.slug, self.cat1_name, self.cat2_name, self.direction))
    
    def constraint(self) -> tuple[Constraint, tuple[str]]:
        def is_specific_side_of(pos1: int | None, pos2: int | None, direction: str):
            if pos1 is None or pos2 is None:
                return False

            diff = (pos1 - pos2)
            if direction == "left":
                return diff in [1, -5]
            elif direction == "right":
                return diff in [-1, 5]
            else:
                raise Exception("This should never happen")

        def specific_side_of(cat1_pos: int | None, cat2_pos: int | None):
            return is_specific_side_of(cat1_pos, cat2_pos, self.direction)

        return (FunctionConstraint(specific_side_of), (self.cat1_name, self.cat2_name))

    def full_text(self) -> str:
        return f"{self.cat1_name} was sitting to the {self.direction} of {self.cat2_name}"


class AcrossFromCat(Clue):

    slug = "across_from_cat"

    @staticmethod
    def attempt_generate(cat_positions: dict[str, int | None]) -> Optional["AcrossFromCat"]: 
        cat1, cat2 = random_awake_cat_names(cat_positions, 2)
        if is_across_from(cat_positions[cat1], cat_positions[cat2]):
            return AcrossFromCat(cat1, cat2)
        return None

    def __init__(self, cat1_name: str, cat2_name: str):
        self.cat1_name = cat1_name
        self.cat2_name = cat2_name
    
    def __str__(self) -> str:
        return str((self.slug, self.cat1_name, self.cat2_name))
    
    def constraint(self) -> tuple[Constraint, tuple[str]]:
        def across_from(cat1_pos: int | None, cat2_pos: int | None):
            return is_across_from(cat1_pos, cat2_pos)

        return (FunctionConstraint(across_from), (self.cat1_name, self.cat2_name))

    def full_text(self) -> str:
        return f"{self.cat1_name} was sitting across from {self.cat2_name}"
    

class NotAcrossFromCat(Clue):

    slug = "not_across_from_cat"

    @staticmethod
    def attempt_generate(cat_positions: dict[str, int | None]) -> Optional["NotAcrossFromCat"]: 
        cat1, cat2 = random_awake_cat_names(cat_positions, 2)
        if not is_across_from(cat_positions[cat1], cat_positions[cat2]):
            return NotAcrossFromCat(cat1, cat2)
        return None

    def __init__(self, cat1_name: str, cat2_name: str):
        self.cat1_name = cat1_name
        self.cat2_name = cat2_name
    
    def __str__(self) -> str:
        return str((self.slug, self.cat1_name, self.cat2_name))
    
    def constraint(self) -> tuple[Constraint, tuple[str]]:
        def not_across_from(cat1_pos: int | None, cat2_pos: int | None):
            return not is_across_from(cat1_pos, cat2_pos)

        return (FunctionConstraint(not_across_from), (self.cat1_name, self.cat2_name))

    def full_text(self) -> str:
        return f"{self.cat1_name} was not sitting across from {self.cat2_name}"

class NextToItem(Clue):

    slug = "next_to_item"

    @staticmethod
    def attempt_generate(cat_positions: dict[str, int | None]) -> Optional["NextToItem"]: 
        cat, = random_awake_cat_names(cat_positions, 1)
        item, = random_items(1)

        if any(is_next_to(cat_positions[cat], ip) for ip in item_positions(item)):
            return NextToItem(cat, item)
        return None

    def __init__(self, cat_name: str, item_name: str):
        self.cat_name = cat_name
        self.item_name = item_name
    
    def __str__(self) -> str:
        return str((self.slug, self.cat_name, self.item_name))
    
    def constraint(self) -> tuple[Constraint, tuple[str]]:
        def next_to(cat_pos: int | None):
            return is_next_to_one_of(cat_pos, item_positions(self.item_name))

        return (FunctionConstraint(next_to), (self.cat_name,))

    def full_text(self) -> str:
        return f"{self.cat_name} was sitting next to {item_pronoun(self.item_name)} {self.item_name}"
    

class NotNextToItem(Clue):

    slug = "not_next_to_item"

    @staticmethod
    def attempt_generate(cat_positions: dict[str, int | None]) -> Optional["NextToItem"]: 
        cat, = random_awake_cat_names(cat_positions, 1)
        item, = random_items(1)

        if not any(is_next_to(cat_positions[cat], ip) for ip in item_positions(item)):
            return NotNextToItem(cat, item)
        return None

    def __init__(self, cat_name: str, item_name: str):
        self.cat_name = cat_name
        self.item_name = item_name
    
    def __str__(self) -> str:
        return str((self.slug, self.cat_name, self.item_name))
    
    def constraint(self) -> tuple[Constraint, tuple[str]]:
        def not_next_to(cat_pos: int | None):
            return not is_next_to_one_of(cat_pos, item_positions(self.item_name))

        return (FunctionConstraint(not_next_to), (self.cat_name,))

    def full_text(self) -> str:
        return f"{self.cat_name} was not sitting next to the {self.item_name}"


class InFrontOfItems(Clue):

    slug = "in_front_of_items"

    @staticmethod
    def attempt_generate(cat_positions: dict[str, int | None]) -> Optional["NextToCat"]: 
        cat, = random_awake_cat_names(cat_positions, 1)
        
        n_items = random.choices([1, 2], weights=[20, 80])[0]
        items = random_items(n_items)

        if is_in_front_of_items(cat_positions[cat], items):
            return InFrontOfItems(cat, items)
        return None

    def __init__(self, cat_name: str, item_names: list[str]):
        self.cat_name = cat_name
        self.item_names = item_names
    
    def __str__(self) -> str:
        return str((self.slug, self.cat_name, self.item_names))
    
    def constraint(self) -> tuple[Constraint, tuple[str]]:
        def in_front_of_items(cat_pos: int | None):
            return any(cat_pos == pos for pos, pos_items in enumerate(ALL_ITEMS) if set(self.item_names).issubset(set(pos_items)))

        return FunctionConstraint(in_front_of_items), (self.cat_name,)

    def full_text(self) -> str:
        item_str = ' and '.join(f"{item_pronoun(i)} {i}" for i in self.item_names)
        return f"{self.cat_name} was sitting in front of {item_str}"

class Sleeping(Clue):

    slug = "sleeping"

    def __init__(self, cat_name: str):
        self.cat_name = cat_name
    
    def __str__(self):
        return str((self.slug, self.cat_name))
    
    def constraint(self) -> tuple[Constraint, tuple[str]]:
        def is_asleep(cat_pos: int | None):
            return cat_pos is None

        return FunctionConstraint(is_asleep), (self.cat_name,)

    def full_text(self) -> str:
        return f"{self.cat_name} was sleeping"

class PuzzleGenerator:
    def __init__(self, clue_types: list[Clue]):
        self.clue_types = clue_types

    def _generate_solution(self) -> Tuple[Dict[str, int | None], List[str]]:
        num_awake_cats = random.randint(3, len(CATS))
        awake_cats = random.sample(CATS, num_awake_cats)
        sleeping_cats = [cat for cat in CATS if cat not in awake_cats]

        cat_positions = dict(zip(awake_cats, random.sample(range(N_POSITIONS), len(awake_cats))))
        return cat_positions, sleeping_cats
    
    def generate_clues(self, solution: dict[str, int], sleeping_cats: list[str]) -> dict[str, int]:
        clues = []
        for _ in range(50):
            clue_type = random.choice(self.clue_types)
            for _ in range(100):
                clue = clue_type.attempt_generate(solution)
                if clue:
                    clues.append(clue)
                    break

        for cat in sleeping_cats:
            clues.append(Sleeping(cat))

        return clues
    
    def solutions(self, clues: list[Clue]) -> list:
        problem = Problem()
        domain = POSITIONS + [None]

        for cat in CATS:
            problem.addVariable(cat, domain)

        problem.addConstraint(FunctionConstraint(all_different_except_none), CATS)
        
        for clue in clues:
            constraint, vars = clue.constraint()
            problem.addConstraint(constraint, vars)

        return problem.getSolutions()
    
    def _has_unique_solution(self, clues: list[Clue]) -> bool:
        return len(self.solutions(clues)) == 1

    def generate_puzzle(self) -> Dict:
        max_attempts = 10000
        for _ in range(max_attempts):
            cat_positions, sleeping_cats = self._generate_solution()
            clues = self.generate_clues(cat_positions, sleeping_cats)

            if self._has_unique_solution(clues):
                clues = self.remove_redundant_clues(clues)
                crime = self._choose_crime(clues, cat_positions)
                if crime is not None:
                    return {
                        "cat_positions": cat_positions,
                        "sleeping_cats": sleeping_cats,
                        "clues": clues,
                        "crime": crime,
                    }
        raise ValueError(f"Failed to generate a unique puzzle after {max_attempts} attempts.")

    def order_clues(self, clues: List[Tuple]) -> List[Tuple]:
        return sorted(clues, key=lambda clue: self.clue_difficulty.get(clue[0], float('inf')))

    def remove_redundant_clues(self, clues: List[Clue]) -> List[Tuple]:
        def remove_one(cs: List[Clue]) -> int | None:
            for i in range(len(cs)):
                # Don't remove sleeping clues
                if isinstance(cs[i], Sleeping):
                    continue
                remaining_clues = cs[:i] + cs[i+1:]
                if self._has_unique_solution(remaining_clues):
                    return i
            return None
        
        remaining_clues = clues
        random.shuffle(remaining_clues)
        while True:
            remove_i = remove_one(remaining_clues)
            if remove_i is not None:
                remaining_clues = remaining_clues[:remove_i] + remaining_clues[remove_i+1:]
            else:
                return remaining_clues

        raise Exception("This should never happen")

    def _choose_crime(self, clues: list[Clue], cat_positions: dict[str, int | None]) -> str | None:
        def is_cat_eligible(cat: str) -> bool:
            for clue in clues:
                # TODO: cats should be generically exposed on the Clue class
                if isinstance(clue, InFrontOfItems):
                    if clue.cat_name == cat:
                        return False
                if isinstance(clue, Sleeping):
                    if clue.cat_name == cat:
                        return False
            return True

        eligible_cats = [c for c in CATS if is_cat_eligible(c)]
        if len(eligible_cats) == 0:
            return None

        cat = random.choice(eligible_cats)
        cat_pos = cat_positions[cat]
        return CRIME_ITEMS[cat_pos]

        
NON_SLEEPING_CLUES = [
    InFrontOfItems,
    NextToCat,
    SpecificSideOfCat,
    NextToItem,
    NotNextToItem,
    AcrossFromCat,
    NotAcrossFromCat,
]
ALL_CLUES = [Sleeping] + NON_SLEEPING_CLUES

def import_clues(serialized_clues: list) -> list[Clue]:
    clues = []
    for c in serialized_clues:
        slug, *params = c
        clue_type = next(c for c in ALL_CLUES if c.slug == slug)
        clues.append(clue_type(*params))
    return clues


def order_clues(clues: list[Clue]) -> list[Clue]:
    priority_dict = {cls: idx for idx, cls in enumerate(ALL_CLUES)}
    def priority_key(clue):
        return priority_dict.get(type(clue), len(ALL_CLUES))
    return sorted(clues, key=priority_key)


def crime_image_path(crime: str) -> str:
    img_path = {
        "bird cage": "bird_cage.webp",
        "coffee": "coffee.webp",
        "shoes": "shoes.webp",
        "fish bowl": "fish_bowl.webp",
        "ball of yarn": "ball_of_yarn.webp",
        "plant": "plant.webp",
    }[crime]

    return "file://" + os.path.abspath(f"images/{img_path}")


def crime_title(crime: str) -> str:
    return {
        "bird cage": "Who ate the bird?",
        "coffee": "Who spilled the coffee?",
        "shoes": "Who ruined the shoes?",
        "fish bowl": "Who swallowed the fish?",
        "ball of yarn": "Who chased the yarn?",
        "plant": "Who bashed the plant?",
    }[crime]


def save_puzzle_img(clues: List[Clue], crime: str, save_dir: str):    
    clue_texts = [c.full_text() for c in order_clues(clues)]
    img_path = crime_image_path(crime)
    title = crime_title(crime).upper()

    svg_path = Path(save_dir) / "puzzle.svg"
    png_path = Path(save_dir) / "puzzle.png"

    dwg = svgwrite.Drawing(svg_path, size=('300px', '400px'))
    
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=10, ry=10, fill='black'))
    dwg.add(dwg.rect(insert=(10, 10), size=('280px', '380px'), rx=5, ry=5, fill='white'))
    
    dwg.add(dwg.text('BEGINNER', insert=(20, 40), fill='black', font_size='28px', font_weight='bold', font_family='Arial Black, sans-serif'))
    dwg.add(dwg.line(start=(20, 45), end=(150, 45), stroke='black', stroke_width=2))

    dwg.add(dwg.text(title, insert=(20, 90), fill='black', font_size='18px', font_weight='bold', font_family='Arial, sans-serif'))
    
    clip_path = dwg.defs.add(dwg.clipPath(id='circleClip'))
    clip_path.add(dwg.circle(center=(260, 40), r=25))
    
    image_group = dwg.g(clip_path='url(#circleClip)')
    image_group.add(dwg.image(href=img_path, insert=(235, 15), size=(50, 50)))
    dwg.add(image_group)
    dwg.add(dwg.circle(center=(260, 40), r=25, fill='none', stroke='black', stroke_width=2))
    
    y = 120
    for clue in clue_texts:
        wrapped_lines = textwrap.wrap(clue, width=40)
        for i, line in enumerate(wrapped_lines):
            dwg.add(dwg.text(f"{'• ' if i == 0 else '  '}{line}", insert=(20, y), fill='black', font_size='14px', font_family='Arial, sans-serif'))
            y += 20

    dwg.save()
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2, unsafe=True)



def save_solution_img(cat_positions: Dict[str, int], crime: str, save_dir: str):
    svg_path = Path(save_dir) / "solution.svg"
    png_path = Path(save_dir) / "solution.png"
    dwg = svgwrite.Drawing(svg_path, size=('300px', '400px'))

    # Background
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=10, ry=10, fill='black'))
    dwg.add(dwg.rect(insert=(10, 10), size=('280px', '380px'), rx=5, ry=5, fill='white'))
    dwg.add(dwg.ellipse(center=(150, 220), r=('120px', '160px'), fill='#F5DEB3'))

    crime_objects = ["bird cage", "coffee", "shoes", "fish bowl", "ball of yarn", "plant"]
    object_positions = [
        (150, 80),   # Top
        (240, 160),  # Top-right
        (240, 280),  # Bottom-right
        (150, 360),  # Bottom
        (60, 280),   # Bottom-left
        (60, 160),   # Top-left
    ]

    cat_positions_coords = [
        (80, 60),   # Top
        (240, 120),  # Top-right
        (240, 240),  # Bottom-right
        (80, 380),  # Bottom
        (50, 240),   # Bottom-left
        (50, 120),   # Top-left
    ]

    guilty_cat = next(cat for cat, pos in cat_positions.items() if pos == crime_objects.index(crime))

    # Add cats
    for cat, pos in cat_positions.items():
        if pos is not None:
            x, y = cat_positions_coords[pos]
            dwg.add(dwg.text(cat, insert=(x, y), fill='black', font_size='14px', font_weight='bold', font_family='Arial, sans-serif', text_anchor='middle'))

    # Add crime object images
    image_size = 60  # Diameter of the red circle
    for obj, (x, y) in zip(crime_objects, object_positions):
        image_path = crime_image_path(obj)
        dwg.add(dwg.image(href=str(image_path), insert=(x - image_size/2, y - image_size/2), size=(image_size, image_size)))

        if obj == crime:
            dwg.add(dwg.circle(center=(x, y), r=image_size/2, fill='none', stroke='red', stroke_width=2))

    # Title and guilty cat
    dwg.add(dwg.text('THE GUILTY CAT IS...', insert=(20, 40), fill='black', font_size='20px', font_weight='bold', font_family='Arial Black, sans-serif'))
    dwg.add(dwg.line(start=(20, 45), end=(280, 45), stroke='black', stroke_width=2))
    dwg.add(dwg.text(guilty_cat, insert=(280, 65), fill='red', font_size='20px', font_weight='bold', font_family='Arial, sans-serif', text_anchor='end'))

    dwg.save()
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2, unsafe=True)


def count_non_sleeping_clues(clues: list[Clue]) -> int:
    return len(clues) - sum(1 for c in clues if isinstance(c, Sleeping))


def clues_meet_requirements(clues: list[Clue], requirements: list[Callable]) -> bool:
    return all(r(clues) for r in requirements)


def generate_puzzle(clue_types: list, requirements: list[Callable]) -> dict:
    generator = PuzzleGenerator(clue_types=clue_types)
    while True:
        new_puzzle = generator.generate_puzzle()
            
        if clues_meet_requirements(new_puzzle['clues'], requirements):
            return new_puzzle


def generate_trivial_challenge() -> dict:
    return generate_puzzle(
        [
            InFrontOfItems,
            NextToCat,
            SpecificSideOfCat,
            NextToItem,
            AcrossFromCat,
        ],
        [
            lambda cs: count_non_sleeping_clues(cs) <= 6,
            lambda cs: sum(1 for c in cs if isinstance(c, InFrontOfItems) and len(c.item_names) > 1) >= 1,
            lambda cs: sum(1 for c in cs if isinstance(c, NextToItem)) > 0,
            lambda cs: sum(1 for c in cs if isinstance(c, NextToCat)) > 0,
            lambda cs: sum(1 for c in cs if isinstance(c, NotNextToItem) == 0),
            lambda cs: sum(1 for c in cs if isinstance(c, NotAcrossFromCat) == 0),
            lambda cs: sum(1 for c in cs if isinstance(c, SpecificSideOfCat) == 0),
        ]
    )


def generate_simple_challenge() -> dict:
    return generate_puzzle(
        NON_SLEEPING_CLUES,
        [
            lambda cs: count_non_sleeping_clues(cs) <= 6,
            lambda cs: sum(1 for c in cs if isinstance(c, InFrontOfItems) and len(c.item_names) > 1) == 1,
            lambda cs: sum(1 for c in cs if isinstance(c, NotNextToItem) <= 2),
        ],
    )


def generate_medium_challenge() -> dict:
    return generate_puzzle(
        NON_SLEEPING_CLUES,
        [
            lambda cs: count_non_sleeping_clues(cs) <= 6,
            lambda cs: sum(1 for c in cs if isinstance(c, InFrontOfItems) and len(c.item_names) > 1) <= 1,
            lambda cs: sum(1 for c in cs if isinstance(c, NotNextToItem) <= 2),
        ],
    )