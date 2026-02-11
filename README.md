## Overview
This project implements a Computer Vision system for automatic scoring of the Qwirkle Connect game. The system analyzes a sequence of images, detects the board configuration, identifies newly placed pieces, and computes the score according to the official rules.

## Structure
- `abordare_template.py`: full pipeline implementation (board detection, piece detection, scoring, export).
- `templates/`: detection templates (e.g., digit 2).
- `README.txt`: run instructions (plain text).
- `README.md`: this document.

## Requirements
- `numpy==2.2.6`
- `opencv_python==4.12.0.88`

## How to run
1. Open a terminal in the project folder.
2. Run the script:
   ```bash
   python abordare_template.py
   ```

## Input
- Input images must be in a folder and follow the format: `<game_id>_<move>.jpg` (e.g., `1_00.jpg`, `1_01.jpg`, ...).
- The default folder is `antrenare`, but you can change it in:
  ```python
  proceseaza_datele(input_folder, output_folder)
  ```

## Output
- For each move, a text file is generated in `output_folder`: `<game_id>_<move>.txt`.
- The file contains lines for newly placed pieces and the move score on the last line.
- Default output: `antrenare_output` (or `output` if explicitly set).

### Output file example
```text
1A 12
1B 34
2C 25
7
```
Meaning:
- `1A 12` means row 1, column A, piece with shape id `1` and color code `2`.
- The last line `7` is the score for the move.

## Internal flow (short)
1. **Board detection**: extract the green board area using geometric and color-based detection, then apply perspective warp to a fixed 1440x1440 view.
2. **Piece detection**: segment shapes on the warped board, classify their shape and color, and map them to `(shape_id, color_code)`.
3. **Scoring**: compare the current board state to the previous one, determine newly placed pieces, compute per-move score, then export to text.

## Example call
The script already includes an example:
```python
proceseaza_datele('evaluare/fake_test', 'output')
```

## Notes
If you use an external evaluator, set the path to `output_folder` (e.g., `predictions_path_root = "output/"`).
