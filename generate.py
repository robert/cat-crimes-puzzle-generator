import os
from pathlib import Path
from uuid import uuid4
from lib import generate_medium_challenge, generate_simple_challenge, generate_trivial_challenge, order_clues, save_puzzle_img, save_solution_img


if __name__ == "__main__":
    for (fn, l, n) in [(generate_trivial_challenge, "a", 4), (generate_simple_challenge, "b", 4), (generate_medium_challenge, "c", 2)]:
        for i in range(n):
            new_puzzle = fn()

            for c in order_clues(new_puzzle['clues']):
                print(c.full_text())

            uuid = uuid4()
            print(f"uuid={str(uuid)}")

            d = os.path.abspath(f'./outputs/{l}-{i}-{uuid}')
            os.makedirs(d, exist_ok=True)

            with open(Path(d) / "puzzle.txt", 'w') as f:
                f.write("\n".join(str(c) for c in new_puzzle['clues']))

            save_puzzle_img(new_puzzle['clues'], new_puzzle['crime'], d)
            save_solution_img(new_puzzle['cat_positions'], new_puzzle['crime'], d)
