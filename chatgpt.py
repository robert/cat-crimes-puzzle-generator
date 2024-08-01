import json
import os
from uuid import uuid4

from lib import ALL_CLUES, PuzzleGenerator, import_clues, save_puzzle_img, save_solution_img


if __name__ == "__name__":
    output = ""

    for fpath in ["./generate.py", "./lib.py"]:
        output += f"===={fpath}===="
        with open(fpath) as f:
            output += f.read()

    output += """Here is code defining a puzzle game. Give me a puzzle in JSON, on a single line that I can paste into Python's input(), that will be accepted by the function `import_clues` that adheres to the rules in `generate_simple_challenge`. Explain how you expect a player to solve the puzzle. Ensure that the puzzle only has 1 unique solution.

    Paste the above prompt into ChatGPT."""

    print(output)

    generator = PuzzleGenerator(ALL_CLUES)
    while True:
        json_clues_data = input("""Paste ChatGPT output here > """)
        clues_data = json.loads(json_clues_data)

        clues = import_clues(clues_data)
        solutions = generator.solutions(clues)

        if len(solutions) == 1:
            print("Valid challenge! Saving...")
            print(solutions)
            break
        else:
            print(f"""Challenge not valid, not exactly 1 solution. Solutions: {solutions}

Paste the above error into ChatGPT
""")

    solution = solutions[0]
    crime = generator._choose_crime(clues, solution)

    uuid = uuid4()

    d = f"outputs/chatgpt-{uuid}"
    os.makedirs(d, exist_ok=True)

    save_puzzle_img(clues, crime, d)
    save_solution_img(solution, crime, d)

    print(f"Saved to {d}")