## Overview
This project implements an Computer Vision system for automatic scoring of the **Qwirkle Connect** board game. The system processes a sequence of game images, detects the green game board, identifies pieces (shapes and colors), tracks game state across moves, and computes scores according to official Qwirkle rules.

### Key Features
- **Robust board detection** using hybrid geometric and color-based approaches
- **Template matching** for shape recognition with geometric fallback
- **HSV color classification** for accurate piece identification
- **Automatic perspective correction** to standardized 1440x1440px board view
- **Configuration matrix** system for handling special game layouts
- **Delta detection** between consecutive moves for scoring
- **Qwirkle rule validation** including bonus points for completing lines of 6

## Project Structure
- `abordare_template.py`: Complete pipeline implementation with all modules
- `templates/`: Shape templates for piece detection:
  - `cerc.JPG` - Circle shape
  - `patrat.JPG` - Square shape
  - `romb.JPG` - Diamond/rhombus shape
  - `stea_4.JPG` - 4-pointed star
  - `stea_8.JPG` - 8-pointed star
  - `trifoi.JPG` - Clover/trefoil shape
- `README.txt`: Quick run instructions (plain text)
- `README.md`: This comprehensive documentation
- `DocumentatieCAVA.pdf`: Detailed technical documentation (Romanian)

## Requirements
- `numpy==2.2.6`
- `opencv_python==4.12.0.88`
- Python 3.x

## How to run
1. Open a terminal in the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python abordare_template.py
   ```
4. The script processes images from the input folder and generates output files.

## Input Format
- Input images must be in a folder with naming format: `<game_id>_<move>.jpg` 
  - Example: `1_00.jpg`, `1_01.jpg`, `1_02.jpg`, ...
- The default input folder is `antrenare`, configurable in:
  ```python
  proceseaza_datele(input_folder, output_folder)
  ```
- Images should show the complete Qwirkle game board with good lighting

## Output Format
- For each move, a text file is generated: `<game_id>_<move>.txt`
- Output location: `antrenare_output` by default (or custom `output_folder`)
- Each line represents a newly placed piece, the last line is the move score

### Output File Structure
```text
<row><col> <shape_id><color_code>
<row><col> <shape_id><color_code>
...
<score>
```

### Example Output File
```text
1A 12
1B 34
2C 25
7
```

**Interpretation:**
- `1A 12`: Row 1, Column A, Shape ID=1 (circle), Color Code=2 (red)
- `1B 34`: Row 1, Column B, Shape ID=3 (diamond), Color Code=4 (green)
- `2C 25`: Row 2, Column C, Shape ID=2 (square), Color Code=5 (yellow)
- `7`: Total score for this move

### Encoding System
**Shape IDs:**
- `1` = Circle (cerc)
- `2` = Square (patrat)
- `3` = Diamond (romb)
- `4` = 4-pointed star
- `5` = 8-pointed star
- `6` = Clover (trifoi)

**Color Codes:**
- `1` = Orange
- `2` = Red
- `3` = Purple
- `4` = Green
- `5` = Yellow
- `6` = Blue

## Internal flow (short)
1. **Board detection**: extract the green board area using geometric and color-based detection, then apply perspective warp to a fixed 1440x1440 view.
2. **Piece detection**: segment shapes on the warped board, classify their shape and color, and map them to `(shape_id, color_code)`.
3. **Scoring**: compare the current board state to the previous one, determine newly placed pieces, compute per-move score, then export to text.

## Detailed Pipeline Architecture

### Stage 1: Board Detection & Extraction (`extrage_careu`)

The system uses a **hybrid approach** with two complementary methods:

#### Method A: Geometric Detection (Primary)
1. **Preprocessing:**
   - Convert to grayscale
   - Gaussian blur (9x9 kernel)
   - Adaptive illumination normalization using CLAHE
   - Binary thresholding (threshold=60)

2. **Edge Detection:**
   - Canny edge detection (40-75 thresholds)
   - Morphological operations (dilation + erosion)
   - Closing and opening to clean contours

3. **Corner Extraction:**
   - Find all external contours
   - For each contour, identify 4 extreme points:
     - Top-left: minimum sum of coordinates
     - Bottom-right: maximum sum
     - Top-right: minimum difference
     - Bottom-left: maximum difference
   - Select contour with largest area

4. **Refinement (`rafineaza_colturi`):**
   - Expand detected region with configurable padding (default: 60px)
   - Apply HSV color filtering for green board area
   - Refine corner positions using mask-based contour analysis

#### Method B: Color-Based Fallback
- Used when geometric detection fails validation
- HSV color space filtering (Hue: 30-90°, Saturation: >40)
- Morphological closing with large kernel (15x15, 10 iterations)
- Minimum area rectangle fitting on largest green contour

#### Validation (`este_rezultat_valid`)
All detections must pass three validation checks:
1. **Area check:** Board area must be >15% of image area
2. **Aspect ratio:** Must be square-like (0.8 < ratio < 1.2)
3. **Color verification:** Mean HSV values must indicate green surface

#### Perspective Correction
- Order detected corners (TL, TR, BR, BL)
- Apply perspective transform to standardized 1440×1440px output
- Ensures consistent grid cell size (90px per cell for 16×16 grid)

### Stage 2: Piece Detection & Classification

#### A. Shape Segmentation (`detecteaza_forme_din_masca`)
1. **Color-based Masking:**
   - Create binary mask excluding pure green background
   - Grid-based scanning (16×16 cells, 90px each)
   - Detect occupied cells (>2000 non-zero pixels threshold)

2. **Contour Analysis:**
   - Find contours in each occupied cell
   - Filter by area (300-6000 px²) to remove noise
   - Select contour closest to cell center (<50px distance)

#### B. Shape Classification (Dual Approach)

**Method 1: Template Matching (`identifica_forma_template`)**
- Primary method with higher accuracy
- Process:
  1. Normalize symbol size to 70×70px
  2. Preprocess: Gaussian blur, Otsu's thresholding, morphological cleaning
  3. Match against 6 rotated templates (0°, 90°, 180°, 270°) for each shape
  4. Use TM_CCOEFF_NORMED correlation
  5. Accept if confidence > 0.8 threshold

**Method 2: Geometric Analysis (`classify_shape`)**
- Fallback when template matching fails
- Hu moments calculation for shape invariance
- Circularity metrics: `4π × Area / Perimeter²`
- Decision tree:
  - **Circle:** circularity > 0.80
  - **Square:** circularity 0.65-0.80, solidity > 0.85, convexity > 0.85
  - **Diamond:** circularity < 0.65, aspect ratio 0.8-1.2
  - **Stars:** Low convexity (<0.80) with vertex analysis
  - **Clover:** Specific contour pattern with defects

#### C. Color Classification (`classify_color`)
Uses HSV color space analysis:
1. Extract mean HSV values within piece contour
2. Apply decision tree based on HSV ranges:
   - **Orange:** H=10-25, S>100, V>100
   - **Red:** H=0-10 or 170-180, S>100, V>80
   - **Purple:** H=130-160, S>50, V>50
   - **Green:** H=40-85, S>80, V>60
   - **Yellow:** H=20-40, S>100, V>150
   - **Blue:** H=90-130, S>90, V>70

### Stage 3: Configuration Matrix (`detecteaza_cifra_2`)

Special handling for predefined Qwirkle board patterns:
- Detects "2" markers at specific board positions
- Identifies 4 possible diagonal configurations:
  - Careu1: Main diagonal (top-left quadrant)
  - Careu2: Secondary diagonal (top-right quadrant)
  - Careu3: Main diagonal (bottom-left quadrant)
  - Careu4: Secondary diagonal (bottom-right quadrant)
- Marks blocked cells (-1) and starting positions (1)
- Used for initialization of game state

### Stage 4: Scoring Engine (`calculeaza_scor_complex`)

#### Delta Detection
- Compare previous board state with current state
- Identify newly placed pieces (None → piece)
- Determine move orientation (horizontal/vertical)

#### Score Calculation Rules
1. **Line Score:**
   - Count connected pieces in primary direction
   - Award points equal to line length (if > 1)
   - **Qwirkle bonus:** +6 points if line length = 6

2. **Secondary Lines:**
   - For each new piece, check perpendicular direction
   - Add secondary line scores (if length > 1)
   - Apply Qwirkle bonus for completed 6-piece lines

3. **Edge Cases:**
   - Single isolated piece: Score = 0 (invalid move typically)
   - Multiple pieces must form continuous line

#### Output Generation
- Export format: `<row><column> <shape_id><color_code>`
- Convert grid coordinates to alphanumeric (A-P for columns, 1-16 for rows)
- Final line: Total score for the move

## Algorithm Performance Notes

### Illumination Handling (`uniformizeaza_iluminare`)
- Background normalization using large Gaussian kernel (61×61)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Clip limit: 2.5
  - Tile grid: 8×8
- Unsharp masking for edge enhancement (weight: 1.4/-0.4)

### Optimization Techniques
- Grid-based cell scanning reduces computation
- Template pre-loading at initialization
- Morphological operations tuned for board geometry
- Hierarchical validation (fast checks first)

### Robustness Features
- Hybrid detection with automatic fallback
- Multi-rotation template matching
- Geometric validation at multiple stages
- Configurable thresholds and padding parameters


## Example call
The script already includes an example:
```python
proceseaza_datele('evaluare/fake_test', 'output')
```

