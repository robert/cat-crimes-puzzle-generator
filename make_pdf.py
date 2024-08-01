import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image

def create_puzzle_book(puzzles_root_dir, output_file):
    # Create the PDF canvas
    c = canvas.Canvas(output_file, pagesize=(A4[1], A4[0]))  # A4 landscape
    page_width, page_height = A4[1], A4[0]

    def add_title_page(title):
        c.setFont("Helvetica-Bold", 48)
        c.setFillColor(black)
        c.drawCentredString(page_width/2, page_height/2, title)
        c.showPage()

    def add_image_to_pdf(img_path, page_num, is_puzzle=False, solution_page=None):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Calculate scaling factor to fit image on page (with margins)
        scale = min((page_width - 2*inch) / img_width, (page_height - 2*inch) / img_height)
        
        # Calculate position to center the image
        x = (page_width - img_width * scale) / 2
        y = (page_height - img_height * scale) / 2
        
        c.drawImage(img_path, x, y, width=img_width*scale, height=img_height*scale)
        
        # Add page number (bigger)
        c.setFont("Helvetica-Bold", 24)
        c.setFillColor(black)
        c.drawString(page_width - 1.5*inch, 0.75 * inch, str(page_num))
        
        # Add "Solution on page X" for puzzle pages
        if is_puzzle and solution_page:
            c.setFont("Helvetica", 14)
            c.drawString(1*inch, 0.75 * inch, f"Solution on page {solution_page}")
        
        c.showPage()

    # Get sorted list of puzzle directories
    puzzle_dirs = sorted([d for d in os.listdir(puzzles_root_dir) 
                          if os.path.isdir(os.path.join(puzzles_root_dir, d))])

    # First pass: count valid puzzles and solutions to calculate solution pages
    valid_puzzles = []
    valid_solutions = []
    for puzzle_dir in puzzle_dirs:
        puzzle_path = os.path.join(puzzles_root_dir, puzzle_dir, 'puzzle.png')
        solution_path = os.path.join(puzzles_root_dir, puzzle_dir, 'solution.png')
        if os.path.exists(puzzle_path):
            valid_puzzles.append(puzzle_path)
        if os.path.exists(solution_path):
            valid_solutions.append(solution_path)

    # Calculate starting pages
    challenges_start_page = 2  # Account for "Challenges" title page
    solutions_start_page = challenges_start_page + len(valid_puzzles) + 1  # Account for "Solutions" title page

    add_title_page("Challenges")

    for i, puzzle_path in enumerate(valid_puzzles):
        puzzle_page = challenges_start_page + i
        solution_page = solutions_start_page + i + 1 if i < len(valid_solutions) else None
        add_image_to_pdf(puzzle_path, puzzle_page, is_puzzle=True, solution_page=solution_page)

    add_title_page("Solutions")

    for i, solution_path in enumerate(valid_solutions):
        solution_page = solutions_start_page + i + 1
        add_image_to_pdf(solution_path, solution_page)

    c.save()

puzzles_root_dir = 'outputs'
output_file = 'puzzles.pdf'

create_puzzle_book(puzzles_root_dir, output_file)