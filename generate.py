import svgwrite
import os
import cairosvg
import random
import textwrap
from typing import List, Dict, Optional, Tuple
from uuid import uuid4
from constraint import Constraint, Problem, AllDifferentConstraint, FunctionConstraint


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
    ["bird cage"],
    ["bell ball", "paw print"],
    ["catnip"],
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
        return f"{self.cat_name} was sitting next to the {self.item_name}"
    

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
        item_str = ' and '.join(f"the {i}" for i in self.item_names)
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

class CatCrimesPuzzleGenerator:
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
        awake_cats = list(solution.keys())

        for _ in range(len(awake_cats)):
            choose_from_clue_types = self.clue_types
            n_in_front_ofs = sum(1 for c in clues if c.__class__ == InFrontOfItems)
            if n_in_front_ofs >= 2:
                choose_from_clue_types = [c for c in choose_from_clue_types if c != InFrontOfItems]

            clue_type = random.choice(choose_from_clue_types)
            for i in range(100):
                clue = clue_type.attempt_generate(solution)
                if clue:
                    clues.append(clue)

        # Add clues about sleeping cats
        for cat in sleeping_cats:
            clues.append(Sleeping(cat))

        #return self.remove_redundant_clues(clues, cat_positions)
        return clues
    
    def _solutions(self, clues: list[Clue]) -> list:
        problem = Problem()

        for cat in CATS:
            problem.addVariable(cat, POSITIONS + [None])
        problem.addConstraint(AllDifferentConstraint())
        
        for clue in clues:
            constraint, vars = clue.constraint()
            problem.addConstraint(constraint, vars)

        return problem.getSolutions()
    
    def _has_unique_solution(self, clues: list[Clue]) -> bool:
        return len(self._solutions(clues)) == 1

    def generate_puzzle(self) -> Dict:
        max_attempts = 10000
        for _ in range(max_attempts):
            cat_positions, sleeping_cats = self._generate_solution()
            clues = self.generate_clues(cat_positions, sleeping_cats)

            if self._has_unique_solution(clues):
                clues = self.remove_redundant_clues(clues)
                crime = self._choose_crime(clues, cat_positions)
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

    def _choose_crime(self, clues: list[Clue], cat_positions: dict[str, int | None]) -> str:
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
        cat = random.choice(eligible_cats)
        cat_pos = cat_positions[cat]
        return CRIME_ITEMS[cat_pos]


            

clue_types = [
    NextToCat,
    NextToItem,
    InFrontOfItems,
    AcrossFromCat,
    NotAcrossFromCat,
]

def import_clues(serialized_clues: list) -> list[Clue]:
    clues = []
    for c in serialized_clues:
        slug, *params = c
        clue_type = next(c for c in clue_types + [Sleeping] if c.slug == slug)
        clues.append(clue_type(*params))
    return clues


import sys
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    import datetime
    seed = int(datetime.datetime.now().timestamp())

print(f"{seed=}")
random.seed(seed)


def is_good_challenge(clues: list[Clue], max_n_clues: int | None = None) -> bool:
    n_non_sleeping_clues = len(clues) - sum(1 for c in clues if isinstance(c, Sleeping))
    if max_n_clues and n_non_sleeping_clues > max_n_clues:
        return False

    n_in_front_of = sum(1 for c in clues if isinstance(c, InFrontOfItems) and len(c.item_names) > 1)
    if n_in_front_of != 1:
        return False
    
    return True


def order_clues(clues: list[Clue]) -> list[Clue]:
    priorities = [
        Sleeping,
        InFrontOfItems,
        NextToCat,
        NextToItem,
        NotNextToItem,
        AcrossFromCat,
        NotAcrossFromCat,
    ]
    priority_dict = {cls: idx for idx, cls in enumerate(priorities)}
    def priority_key(clue):
        return priority_dict.get(type(clue), len(priorities))
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
        "shoes": "Who chewed the shoes?",
        "fish bowl": "Who ate the fish?",
        "ball of yarn": "Who chased the yarn?",
        "plant": "Who bashed the plant?",
    }[crime]


def save_puzzle(clues: List[Clue], crime: str):    
    clue_texts = [c.full_text() for c in order_clues(clues)]
    img_path = crime_image_path(crime)
    title = crime_title(crime).upper()

    uuid = uuid4()
    fcrime = crime.replace(' ', '_')
    svg_path = os.path.abspath(f'./outputs/svg/{fcrime}-{uuid}.svg')
    png_path = os.path.abspath(f'./outputs/png/{fcrime}-{uuid}.png')
    txt_path = os.path.abspath(f'./outputs/text/{fcrime}-{uuid}.txt')
    
    with open(txt_path, 'w') as f:
        f.write("\n".join(str(c) for c in clues))

    dwg = svgwrite.Drawing(svg_path, size=('300px', '400px'))
    
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=10, ry=10, fill='#4CAF50'))
    dwg.add(dwg.rect(insert=(10, 10), size=('280px', '380px'), rx=5, ry=5, fill='white'))
    
    dwg.add(dwg.text('BEGINNER', insert=(20, 40), fill='black', font_size='28px', font_weight='bold', font_family='Arial Black, sans-serif'))
    dwg.add(dwg.line(start=(20, 45), end=(150, 45), stroke='black', stroke_width=2))

    dwg.add(dwg.text(title, insert=(20, 90), fill='black', font_size='18px', font_weight='bold', font_family='Arial, sans-serif'))
    
    clip_path = dwg.defs.add(dwg.clipPath(id='circleClip'))
    clip_path.add(dwg.circle(center=(260, 40), r=25))
    
    image_group = dwg.g(clip_path='url(#circleClip)')
    image_group.add(dwg.image(href=img_path, insert=(235, 15), size=(50, 50)))
    dwg.add(image_group)
    dwg.add(dwg.circle(center=(260, 40), r=25, fill='none', stroke='red', stroke_width=2))
    
    y = 120
    for clue in clue_texts:
        wrapped_lines = textwrap.wrap(clue, width=40)
        for i, line in enumerate(wrapped_lines):
            dwg.add(dwg.text(f"{'• ' if i == 0 else '  '}{line}", insert=(20, y), fill='black', font_size='14px', font_family='Arial, sans-serif'))
            y += 20

    dwg.save()
    cairosvg.svg2png(url=svg_path, write_to=png_path, scale=2, unsafe=True)


generator = CatCrimesPuzzleGenerator(clue_types=[
    NextToCat,
    NextToItem,
    NotNextToItem,
    InFrontOfItems,
    AcrossFromCat,
    NotAcrossFromCat,
])
while True:
    new_puzzle = generator.generate_puzzle()
    if is_good_challenge(new_puzzle['clues'], max_n_clues=6):
        break

for c in order_clues(new_puzzle['clues']):
    print(c.full_text())

save_puzzle(new_puzzle['clues'], new_puzzle['crime'])

print("---")
for c in new_puzzle['clues']:
    print(str(c))
print("---")
print(new_puzzle["cat_positions"])

print("====")
clues = import_clues([
    ('sleeping', 'Sassy'),
    ('next_to_cat', 'Ginger', 'Mr. Mittens'),
    ('in_front_of_items', 'Duchess', ['ball of yarn', 'mouse']),
    ('next_to_item', 'Tom Cat', 'fish bowl'),
    ('across_from_cat', 'Pip Squeak', 'Ginger'),
    ('in_front_of_items', 'Ginger', ['sock', 'catnip'])
])
for c in clues:
    print(c.full_text())

print("===")
print(generator._solutions(clues))

