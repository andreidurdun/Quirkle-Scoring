# %%
import cv2 as cv
import numpy as np
import os
import ctypes
import re

# %%
def show_image(title, image, max_scale=1.0, margin=80):
    h, w = image.shape[:2]
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    scale = min((screen_w - margin) / w, (screen_h - margin) / h, max_scale)
    if scale < 1.0:
        image = cv.resize(image, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# %%
def uniformizeaza_iluminare(gray, blur_kernel=61, eps=1e-6):

    # Fundal (iluminare lentă)
    background = cv.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    # Flatten (convertim la float)
    ratio = (gray.astype(np.float32) / (background.astype(np.float32) + eps))
    # Rescalare la 0..255
    ratio = cv.normalize(ratio, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # CLAHE pentru contrast local
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    local = clahe.apply(ratio)

    # Unsharp (claritate)
    blur_small = cv.GaussianBlur(local, (0,0), 2.0)
    sharp = cv.addWeighted(local, 1.4, blur_small, -0.4, 0)

    return sharp



def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def este_rezultat_valid(image, corners):
    
   
    area = cv.contourArea(corners)
    img_area = image.shape[0] * image.shape[1]
    
    if area < (0.15 * img_area):
        print(f"[Validator] FAIL: Arie prea mica ({area:.0f}).")
        return False

    # 2. Verificare Raport Aspect (Tabla e 1:1)
    ordered = order_points(corners)
    w = np.linalg.norm(ordered[0] - ordered[1])
    h = np.linalg.norm(ordered[0] - ordered[3])
    
    # Evitam impartirea la zero
    if h == 0: return False
    aspect_ratio = w / h
    
   
    if not (0.8 < aspect_ratio < 1.2):
        print(f"[Validator] FAIL: Nu e patrat (Ratio={aspect_ratio:.2f}).")
        return False

    #  Verificare Culoare (Verde)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    corners_int = np.int32(corners)
    cv.fillConvexPoly(mask, corners_int, 255)
    
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mean_val = cv.mean(hsv, mask=mask)
    hue, sat = mean_val[0], mean_val[1]
    
    
    is_green = (30 < hue < 95) and (sat > 40)
    
    if not is_green:
        print(f"[Validator] FAIL: Culoare gresita (Hue={hue:.1f}, Sat={sat:.1f}).")
        return False

    print("[Validator] OK: Geometrie si Culoare valide")
    return True



def detectie_geometrica_custom(image):
    # Procesare
    original_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_blur = cv.GaussianBlur(original_gray, (9,9), 0)
    try: image_blur = uniformizeaza_iluminare(image_blur)
    except: pass
    _, thresh = cv.threshold(image_blur, 60, 255, cv.THRESH_BINARY)
    
    canny = cv.Canny(thresh, 40, 75)
    dilate = cv.dilate(canny, np.ones((9,9), np.uint8), iterations=3)
    erode = cv.erode(dilate, np.ones((9,9), np.uint8), iterations=2)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    processed = cv.morphologyEx(erode, cv.MORPH_CLOSE, kernel, iterations=4)
    processed = cv.morphologyEx(processed, cv.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv.findContours(processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    best_corners = None
    
    
    for i in range(len(contours)):
        if len(contours[i]) > 3:
            pts = contours[i].squeeze()
            if pts.ndim != 2: continue
            
            sum_pts = pts.sum(axis=1)
            diff_pts = np.diff(pts, axis=1)

            tl = pts[np.argmin(sum_pts)]
            br = pts[np.argmax(sum_pts)]
            tr = pts[np.argmin(diff_pts)]
            bl = pts[np.argmax(diff_pts)]

            temp_corners = np.array([tl, tr, br, bl], dtype="float32")
            area = cv.contourArea(temp_corners)
            
            if area > max_area:
                max_area = area
                best_corners = temp_corners

    return best_corners



def detectie_culoare_fallback(image):

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # Interval verde 
    mask = cv.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15)) 
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=10) 
    
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    

    best_cnt = max(contours, key=cv.contourArea)

    rect = cv.minAreaRect(best_cnt)
    box = cv.boxPoints(rect)
    return np.float32(box)



def rafineaza_colturi(image, corners, padding=60):
  
    h_img, w_img = image.shape[:2]

    corners = order_points(corners) 

    corners = corners.copy()
    expanded_corners = corners.copy()

   
    expanded_corners[0][0] = max(0, corners[0][0] - padding)
    expanded_corners[0][1] = max(0, corners[0][1] - padding)

    # Top-Right (x+p, y-p)
    expanded_corners[1][0] = min(w_img, corners[1][0] + padding)
    expanded_corners[1][1] = max(0, corners[1][1] - padding)

    # Bottom-Right (x+p, y+p)
    expanded_corners[2][0] = min(w_img, corners[2][0] + padding)
    expanded_corners[2][1] = min(h_img, corners[2][1] + padding)

    # Bottom-Left (x-p, y+p)
    expanded_corners[3][0] = max(0, corners[3][0] - padding)
    expanded_corners[3][1] = min(h_img, corners[3][1] + padding)


    mask_roi = np.zeros((h_img, w_img), dtype=np.uint8)
    
    mask_roi = np.zeros(image.shape[:2], dtype=np.uint8)
    corners_int = np.int32(expanded_corners)
    cv.fillConvexPoly(mask_roi, corners_int, 255)

   
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Masca Verde
    mask_green = cv.inRange(hsv, np.array([30, 40, 40]), np.array([95, 255, 255]))
    
    mask_dark = cv.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))

    mask_content = cv.bitwise_or(mask_green, mask_dark)
    
    mask_refined = cv.bitwise_and(mask_roi, mask_content)
    #show_image("Masca Rafinată", mask_refined)  
   
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    mask_refined = cv.morphologyEx(mask_refined, cv.MORPH_CLOSE, kernel, iterations=10)
    #show_image("Masca Rafinată Curățată", mask_refined)
 
    contours, _ = cv.findContours(mask_refined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return corners 
        
    best_cnt = max(contours, key=cv.contourArea)
    
   
    pts = best_cnt.squeeze()
    if pts.ndim != 2 or len(pts) < 4:
        return corners # Fallback
        
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1)

    tl = pts[np.argmin(sum_pts)]
    br = pts[np.argmax(sum_pts)]
    tr = pts[np.argmin(diff_pts)]
    bl = pts[np.argmax(diff_pts)]

    refined_corners = np.array([tl, tr, br, bl], dtype="float32")
    
  
    if cv.contourArea(refined_corners) < 0.8 * cv.contourArea(corners):
        return corners
        
  
    return refined_corners



def extrage_careu(image):
    original = image.copy()
    final_corners = None
    metoda = ""

    
    corners_geom = detectie_geometrica_custom(original)
    
    if corners_geom is not None:
        corners_geom = rafineaza_colturi(original, corners_geom)
       
        if este_rezultat_valid(original, corners_geom):
            final_corners = corners_geom
            metoda = "Geometrica (OK)"
        
    
    if final_corners is None:
        corners_color = detectie_culoare_fallback(original)
        
       
        if corners_color is not None and este_rezultat_valid(original, corners_color):
            final_corners = corners_color
            metoda = "Culoare (Fallback)"
        

   
    if final_corners is None:
        print("EROARE: Nu am putut detecta tabla.")
        return original

   
    print(f"SUCCES: Folosim metoda '{metoda}'")
    
    ordered_pts = order_points(final_corners)
    width = 1440
    height = 1440
    dst_pts = np.array([[0,0], [width,0], [width,height], [0,height]], dtype="float32")

    M = cv.getPerspectiveTransform(ordered_pts, dst_pts)
    result = cv.warpPerspective(original, M, (width, height))
    
    return result

    

# %%
def detecteaza_cifra_2(image_board, template_path):
   
    img_rgb = image_board.copy()
    
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(template_path, 0) 
    #show_image("Template Cifra 2", template)
    if template is None:
        print("EROARE: Nu am putut incarca sablonul pentru cifra 2")
        return img_rgb

  
    target_size = 80 
    template = cv.resize(template, (target_size, target_size))

    w, h = template.shape[::-1]
    rectangles = [] 

    
    # 0 = 90 clockwise, 1 = 180, 2 = 90 counter-clockwise
    rotations = [None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
    
    current_template = template
    
    for i, rotate_code in enumerate(rotations):
     
        if rotate_code is not None:
            current_template = cv.rotate(template, rotate_code)
        curr_w, curr_h = current_template.shape[::-1]

        res = cv.matchTemplate(img_gray, current_template, cv.TM_CCOEFF_NORMED)
        
        threshold = 0.7
        
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            rectangles.append([int(pt[0]), int(pt[1]), int(curr_w), int(curr_h)])


    rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.2)

    print(f"Am detectat {len(rectangles)} cifre de 2.")

    coordonate = []
 
    for (x, y, w, h) in rectangles:
        i = y // 90
        j = x // 90
        coordonate.append((i,j))


        cv.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #show_image("Detecții Cifra 2", img_rgb)
    

    return coordonate



def draw_grid(image, grid_size=16, cell_px=90, color=(50, 50, 50), thickness=1):
    h, w = image.shape[:2]
    for i in range(grid_size + 1):
        y = i * cell_px
        cv.line(image, (0, y), (w, y), color, thickness)
    for i in range(grid_size + 1):
        x = i * cell_px
        cv.line(image, (x, 0), (x, h), color, thickness)


def classify_color(hsv_img, mask_symbol):

    mean_val = cv.mean(hsv_img, mask=mask_symbol)
    hue, sat, val = mean_val[:3]
    if sat < 40 and val > 60: return "Alb"
    if (hue >= 0 and hue <= 5) or (hue >= 170 and hue <= 179):
        if 3 < hue <= 5 and sat < 90: return "Portocaliu"
        return "Rosu"
    elif 5 < hue <= 18: return "Portocaliu"
    elif 18 < hue <= 32: return "Galben"
    elif 32 < hue <= 85: return "Verde"
    elif 85 < hue <= 130: return "Albastru"
    return "Rosu"

def classify_shape(contour):

    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0 or area < 100: return "Unknown"
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    hull = cv.convexHull(contour)
    solidity = area / cv.contourArea(hull) if cv.contourArea(hull) > 0 else 0
    (_, _), radius = cv.minEnclosingCircle(contour)
    fill_ratio = area / (np.pi * radius**2 + 1e-6)
    
    hull_idx = cv.convexHull(contour, returnPoints=False)
    major_defects = 0
    if hull_idx is not None and len(hull_idx) > 3:
        try:
            defects = cv.convexityDefects(contour, hull_idx)
            if defects is not None:
                for i in range(defects.shape[0]):
                    if defects[i, 0][3] > 0.035 * perimeter * 256: major_defects += 1
        except: pass

    if circularity > 0.82: return "Cerc"
    if major_defects >= 3:
        if major_defects >= 6: return "Stea_8"
        if solidity > 0.70 and fill_ratio > 0.55: return "Trifoi"
        return "Stea_4"
    else:
        rect = cv.minAreaRect(contour)
        (x, y), (w, h), angle = rect
        ar = min(w, h) / max(w, h)
        if ar < 0.65: return "Unknown"
        angle = abs(angle)
        if w < h: angle = abs(angle - 90)
        if angle > 45: angle = 90 - angle
        if angle < 20: return "Patrat"
        elif angle > 25: return "Romb"
        else: return "Patrat" if solidity > 0.94 else "Romb"
    return "Unknown"



# %%


def incarca_templateuri(folder="templates"):

    templates = {}
    if not os.path.exists(folder):
        print(f"[Warning] Folderul '{folder}' nu exista! Se va folosi doar logica geometrica.")
        return templates

    print(f"Incarc template-uri din '{folder}'...")
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(folder, f)
        
            img = cv.imread(path, 0)
            if img is not None:

                _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
                name = os.path.splitext(f)[0]
                templates[name] = img
                print(f" -> {name}")
    return templates


def preprocesare_simbol(img_binar, target_size=(64, 64)):

    coords = cv.findNonZero(img_binar)
    if coords is None:
        return np.zeros(target_size, dtype=np.uint8)
        
    x, y, w, h = cv.boundingRect(coords)
    crop = img_binar[y:y+h, x:x+w]
    

    t_w, t_h = target_size
    pad = 6
    max_dim = max(w, h)
    scale = (min(t_w, t_h) - 2*pad) / max_dim
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv.resize(crop, (new_w, new_h), interpolation=cv.INTER_AREA)
   
    canvas = np.zeros(target_size, dtype=np.uint8)
    

    off_x = (t_w - new_w) // 2
    off_y = (t_h - new_h) // 2
    
    canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized
    
    return canvas

def identifica_forma_template(roi_binar, templates, threshold=0.55):
    if not templates: return "Unknown"
    

    STD_SIZE = 64
   
    roi_centered = preprocesare_simbol(roi_binar, target_size=(STD_SIZE, STD_SIZE))
    

    roi_padded = cv.copyMakeBorder(roi_centered, 8, 8, 8, 8, cv.BORDER_CONSTANT, value=0)
 
    roi_blur = cv.GaussianBlur(roi_padded, (5, 5), 0)

    best_score = -1
    best_label = "Unknown"

    for label, templ_raw in templates.items():
      
        if templ_raw.shape != (STD_SIZE, STD_SIZE):
             templ_proc = preprocesare_simbol(templ_raw, (STD_SIZE, STD_SIZE))
        else:
             templ_proc = templ_raw

    
        templ_blur = cv.GaussianBlur(templ_proc, (5, 5), 0)
        
     
        curr_t = templ_blur
        for _ in range(4):
       
            res = cv.matchTemplate(roi_blur, curr_t, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            score = max_val
            
            if score > best_score:
                best_score = score
                best_label = label
            
            curr_t = cv.rotate(curr_t, cv.ROTATE_90_CLOCKWISE)

    print(f"   [Template] Best: {best_label} ({best_score:.2f})")
    
    if best_score >= threshold:
        return best_label
    
    return "Unknown"


TEMPLATES_DICT = incarca_templateuri("templates")


def detecteaza_forme_din_masca(mask_tiles, hsv_full_image, cell_px=90, grid_size=16):
 
    rezultate = []
    debug_img = cv.cvtColor(mask_tiles, cv.COLOR_GRAY2BGR)
    draw_grid(debug_img, grid_size, cell_px, color=(100, 100, 100))

    pad = 0

    for row in range(grid_size):
        for col in range(grid_size):
            x1_grid, y1_grid = col * cell_px, row * cell_px
            x2_grid, y2_grid = x1_grid + cell_px, y1_grid + cell_px
            center_x, center_y = x1_grid + cell_px//2, y1_grid + cell_px//2

            x1_p = max(0, x1_grid - pad)
            y1_p = max(0, y1_grid - pad)
            x2_p = min(mask_tiles.shape[1], x2_grid + pad)
            y2_p = min(mask_tiles.shape[0], y2_grid + pad)

            roi_mask = mask_tiles[y1_p:y2_p, x1_p:x2_p]
            
            # Verifc am daca celula e goala
            if cv.countNonZero(roi_mask) < 2000: 
                continue 

            roi_processed = roi_mask.copy()
            contours, _ = cv.findContours(roi_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            best_cnt = None
            min_dist = 9999

            if contours:
                for cnt in contours:
                    area = cv.contourArea(cnt)
                    # Eliminam zgomotul foarte mic sau contururi prea mari
                    if area < 300 or area > 6000: continue

                    M = cv.moments(cnt)
                    if M["m00"] == 0: continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    cx_global = x1_p + cx
                    cy_global = y1_p + cy
                    # gasim cel mai apropiat contur de centrul celulei
                    dist = np.sqrt((cx_global - center_x)**2 + (cy_global - center_y)**2)
                    if dist < 50 and dist < min_dist:
                        min_dist = dist
                        best_cnt = cnt

            if best_cnt is not None:
            
                x, y, w, h = cv.boundingRect(best_cnt)
                
                # Extragem simbolul pentru template matching
                symbol_crop = roi_processed[y:y+h, x:x+w]
                 #verificam daca piesa exista
                shape_name = classify_shape(best_cnt)
                if shape_name == "Unknown":
                    continue  

         
                shape_name = identifica_forma_template(symbol_crop, TEMPLATES_DICT, threshold=0.8)
                
                if shape_name == "Unknown":
                    shape_name = classify_shape(best_cnt)
                     

           
                if shape_name != "Unknown":
                    roi_hsv_ext = hsv_full_image[y1_p:y2_p, x1_p:x2_p]
                    mask_sym = np.zeros_like(roi_mask)
                    cv.drawContours(mask_sym, [best_cnt], -1, 255, -1)
                    color_name = classify_color(roi_hsv_ext, mask_sym)

                    gx, gy = x1_p + x, y1_p + y
                    
                    rezultate.append({
                        "row": row, "col": col,
                        "shape": shape_name, "color": color_name,
                        "bbox": (gx, gy, w, h)
                    })
                    
                 
                    cv.rectangle(debug_img, (gx, gy), (gx+w, gy+h), (0, 255, 0), 2)
                    label_text = f"{shape_name[:4]}"
                    cv.putText(debug_img, label_text, (gx, gy - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                
                cv.rectangle(debug_img, (x1_grid+20, y1_grid+20), (x2_grid-20, y2_grid-20), (0, 0, 255), 1)

    #show_image("Debug Template Matching", debug_img)
    return rezultate




def extrage_forme_din_imagine(img_warped, cell_px=90, grid_size=16):
    if img_warped is None: return []

    draw_grid(img_warped, grid_size, cell_px, color=(230, 230, 230), thickness=2)
    
    hsv_full = cv.cvtColor(cv.medianBlur(img_warped, 3), cv.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0], np.uint8)
    upper_black = np.array([179, 255, 80], np.uint8) 

    mask_tiles = cv.inRange(hsv_full, lower_black, upper_black)

    mask_tiles = cv.bitwise_not(mask_tiles)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

    mask_tiles = cv.morphologyEx(mask_tiles, cv.MORPH_OPEN, kernel, iterations=2)
    
    # show_image("Masca Inversata (Piese Negre)", mask_tiles)

    forme = detecteaza_forme_din_masca(mask_tiles, hsv_full, cell_px, grid_size)


    viz = img_warped.copy()
    for f in forme:
        x, y, w, h = f["bbox"]
        cv.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(viz, f"{f['shape'][:4]}", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # show_image("Rezultat Final", viz)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return forme




# %% 



MAP_FORMA = {
    "Cerc": 1, "cerc": 1,
    "Trifoi": 2, "trifoi": 2, "cruce": 2,
    "Romb": 3, "romb": 3, "diamant": 3,
    "Patrat": 4, "patrat": 4,
    "Stea_4": 5, "stea_4": 5, "stea4": 5,
    "Stea_8": 6, "stea_8": 6, "stea8": 6
}

MAP_CULOARE = {
    "Rosu": "R", "Albastru": "B", "Verde": "G",
    "Galben": "Y", "Portocaliu": "O", "Alb": "W"
}


def detecteaza_pozitia_tablelor(careu, template_path):
    
    pozitii_2 = detecteaza_cifra_2(careu, template_path)
    
    tabla = np.zeros((16,16), dtype=int)

    if (1,1) in pozitii_2 and (6,6) in pozitii_2:
        tabla[1,1] = 2
        tabla[6,6] = 2
        
        tabla[1,6] = -1
        tabla[2,5] = -1
        tabla[3,4] = -1
        tabla[4,3] = -1
        tabla[5,2] = -1
        tabla[6,1] = -1

        for i in range(2, 7):
            for j in range(1, 6):
                if tabla[i][j] == -1:
                    tabla[i-1][j] = 1
                    tabla[i][j+1] = 1

        careu1 = "secundara"
        print("Careu1: secundara")
    elif (1,6) in pozitii_2 and (6,1) in pozitii_2:
        tabla[1,6] = 2
        tabla[6,1] = 2

        tabla[1,1] = -1
        tabla[2,2] = -1
        tabla[3,3] = -1
        tabla[4,4] = -1
        tabla[5,5] = -1
        tabla[6,6] = -1

        for i in range(1, 6):
            for j in range(1, 6):
                if tabla[i,j] == -1:
                    tabla[i+1,j] = 1
                    tabla[i,j+1] = 1

        careu1 = "principala"
        print("Careu1: principala")
    else:
        careu1 = "necunoscuta"
        print("Careu1: necunoscuta")

    if (1,9) in pozitii_2 and (6,14) in pozitii_2:
        tabla[1,9] = 2
        tabla[6,14] = 2

        tabla[1,14] = -1
        tabla[2,13] = -1
        tabla[3,12] = -1
        tabla[4,11] = -1
        tabla[5,10] = -1
        tabla[6,9] = -1

        for i in range(2, 7):
            for j in range(9, 14):
                if tabla[i][j] == -1:
                    tabla[i-1][j] = 1
                    tabla[i][j+1] = 1

        careu2 = "secundara"
        print("Careu2: secundara")
    elif (1,14) in pozitii_2 and (6,9) in pozitii_2:
        tabla[1,14] = 2
        tabla[6,9] = 2

        tabla[1,9] = -1
        tabla[2,10] = -1
        tabla[3,11] = -1
        tabla[4,12] = -1
        tabla[5,13] = -1
        tabla[6,14] = -1

        for i in range(1, 6):
            for j in range(9, 14):
                if tabla[i,j] == -1:
                    tabla[i+1,j] = 1
                    tabla[i,j+1] = 1

        careu2 = "principala"
        print("Careu2: principala")
    else:
        careu2 = "necunoscuta"
        print("Careu2: necunoscuta")
    
    if (9,1) in pozitii_2 and (14,6) in pozitii_2:
        tabla[9,1] = 2
        tabla[14,6] = 2

        tabla[9,6] = -1
        tabla[10,5] = -1
        tabla[11,4] = -1
        tabla[12,3] = -1
        tabla[13,2] = -1
        tabla[14,1] = -1

        for i in range(10, 15):          
                for j in range(1, 6):        
                    if tabla[i][j] == -1:
                        tabla[i-1][j] = 1   
                        tabla[i][j+1] = 1

        careu3 = "secundara"
        print("Careu3: secundara")
    elif (9,6) in pozitii_2 and (14,1) in pozitii_2:
        tabla[9,6] = 2
        tabla[14,1] = 2

        tabla[9,1] = -1
        tabla[10,2] = -1
        tabla[11,3] = -1
        tabla[12,4] = -1
        tabla[13,5] = -1
        tabla[14,6] = -1
        
        for i in range(9, 14):
            for j in range(1, 6):
                if tabla[i,j] == -1:
                    tabla[i+1,j] = 1
                    tabla[i,j+1] = 1

        careu3 = "principala"
        print("Careu3: principala")
    else:
        careu3 = "necunoscuta"
        print("Careu3: necunoscuta")

    if (9,9) in pozitii_2 and (14,14) in pozitii_2:
        tabla[9,9] = 2
        tabla[14,14] = 2

        tabla[9,14] = -1
        tabla[10,13] = -1
        tabla[11,12] = -1
        tabla[12,11] = -1
        tabla[13,10] = -1
        tabla[14,9] = -1

        for i in range(10, 15):
            for j in range(9, 14):
                if tabla[i][j] == -1:
                    tabla[i-1][j] = 1
                    tabla[i][j+1] = 1

        careu4 = "secundara"
        print("Careu4: secundara")
    elif (9,14) in pozitii_2 and (14,9) in pozitii_2:
        tabla[9,14] = 2
        tabla[14,9] = 2

        tabla[9,9] = -1
        tabla[10,10] = -1
        tabla[11,11] = -1
        tabla[12,12] = -1
        tabla[13,13] = -1
        tabla[14,14] = -1

        for i in range(9, 14):
            for j in range(9, 14):
                if tabla[i,j] == -1:
                    tabla[i+1,j] = 1
                    tabla[i,j+1] = 1

        careu4 = "principala"
        print("Careu4: principala")
    else:
        careu4 = "necunoscuta"
        print("Careu4: necunoscuta")

    #print(tabla)
    return tabla
    


def calculeaza_scor_complex(tabla_veche, tabla_noua, config_matrix):

    rows, cols = tabla_noua.shape
    
    piese_noi = []
    for r in range(rows):
        for c in range(cols):
            val_veche = tabla_veche[r, c]
            val_noua = tabla_noua[r, c]
            if (val_veche is None) and (val_noua is not None):
                piese_noi.append((r, c))

    if not piese_noi: return 0

    def get_line_length(start_r, start_c, dr, dc):
        length = 1 
        # Pozitiv
        cr, cc = start_r + dr, start_c + dc
        while 0 <= cr < rows and 0 <= cc < cols:
            if tabla_noua[cr, cc] is not None:
                length += 1; cr += dr; cc += dc
            else: break
        # Negativ
        cr, cc = start_r - dr, start_c - dc
        while 0 <= cr < rows and 0 <= cc < cols:
            if tabla_noua[cr, cc] is not None:
                length += 1; cr -= dr; cc -= dc
            else: break
        return length

    scor_total = 0
    rs = [p[0] for p in piese_noi]
    cs = [p[1] for p in piese_noi]
    is_horizontal = len(set(rs)) == 1
    is_vertical = len(set(cs)) == 1
    
    ref_r, ref_c = piese_noi[0]

    # Calcul Linii
    if is_horizontal:
        l = get_line_length(ref_r, ref_c, 0, 1)
        if l > 1: 
            scor_total += l
            if l == 6: scor_total += 6
        for (r, c) in piese_noi:
            l_sec = get_line_length(r, c, 1, 0)
            if l_sec > 1:
                scor_total += l_sec
                if l_sec == 6: scor_total += 6
                
    elif is_vertical:
        l = get_line_length(ref_r, ref_c, 1, 0)
        if l > 1:
            scor_total += l
            if l == 6: scor_total += 6
        for (r, c) in piese_noi:
            l_sec = get_line_length(r, c, 0, 1)
            if l_sec > 1:
                scor_total += l_sec
                if l_sec == 6: scor_total += 6


    for (r, c) in piese_noi:
        if config_matrix[r, c] == 2:
            scor_total += 2 
        elif config_matrix[r, c] == 1:
            scor_total += 1

    # Caz prima mutare (1 piesă)
    if scor_total == 0 and len(piese_noi) > 0:
        return len(piese_noi)

    return scor_total



def get_row_col_string(r, c):
    return str(r + 1), chr(ord('A') + c)

def construieste_matricea_curenta(cale_imagine):
    img = cv.imread(cale_imagine)
    if img is None: return None

  
    tabla_warped = extrage_careu(img)

    piese_detectate = extrage_forme_din_imagine(tabla_warped)
    
    matrice_stare = np.full((16, 16), None, dtype=object)
    
    for p in piese_detectate:
        r, c = p['row'], p['col']
        shape_name = p['shape']
        color_name = p['color']
        
        shape_id = MAP_FORMA.get(shape_name, MAP_FORMA.get(shape_name.lower()))
        color_code = MAP_CULOARE.get(color_name)
        
        if shape_id and color_code:
            matrice_stare[r, c] = (shape_id, color_code)
            
    return matrice_stare




def proceseaza_datele(input_folder="antrenare", output_folder="antrenare_output"):

    if not os.path.exists(input_folder):
        print(f"Eroare: Folderul {input_folder} nu exista.")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fisiere = os.listdir(input_folder)
    jocuri_ids = set()
    for f in fisiere:
        match = re.match(r"(\d+)_\d+\.jpg", f)
        if match:
            jocuri_ids.add(int(match.group(1)))
            
    lista_jocuri = sorted(list(jocuri_ids))
    print(f"Jocuri detectate: {lista_jocuri}")

    for joc in lista_jocuri:
        print(f"\n--- Procesare Jocul {joc} ---")
        
   
        file_init = f"{joc}_00.jpg"
        path_init = os.path.join(input_folder, file_init)
        
        img_init_raw = cv.imread(path_init)
        if img_init_raw is None: continue
            
        careu_init = extrage_careu(img_init_raw)
        config_matrix = detecteaza_pozitia_tablelor(careu_init, '2_template.jpg')
        
        piese_init = extrage_forme_din_imagine(careu_init)
        stare_anterioara = np.full((16, 16), None, dtype=object)
        for p in piese_init:
            # if config_matrix[p['row'], p['col']] != -1:
            #     continue
            sid = MAP_FORMA.get(p['shape'], MAP_FORMA.get(p['shape'].lower()))
            cid = MAP_CULOARE.get(p['color'])
            if sid and cid: stare_anterioara[p['row'], p['col']] = (sid, cid)

        scor_total_joc = 0
        

        for mutare in range(1, 101):
            mutare_str = f"{mutare:02d}"
            file_img = f"{joc}_{mutare_str}.jpg"
            file_txt = f"{joc}_{mutare_str}.txt"
            
            path_img = os.path.join(input_folder, file_img)
            path_txt = os.path.join(output_folder, file_txt)
            
            if not os.path.exists(path_img):
                break 
                
            print(f"  > Procesez {mutare_str}...", end=" ")
            stare_curenta = construieste_matricea_curenta(path_img)
            
            if stare_curenta is None:
                continue
            
            # Piesele nu dispar
            mask_piese_vechi = (stare_anterioara != None)
            # Unde aveam piesa si acum nu avem, pastram piesa veche
            stare_curenta[mask_piese_vechi & (stare_curenta == None)] = stare_anterioara[mask_piese_vechi & (stare_curenta == None)]

            

            # Calculam scor
            scor_runda = calculeaza_scor_complex(stare_anterioara, stare_curenta, config_matrix)
            scor_total_joc += scor_runda

 
            linii_fisier = []
            
            for r in range(16):
                for c in range(16):
                    if stare_anterioara[r, c] is None and stare_curenta[r, c] is not None:
                        ro, co = get_row_col_string(r, c) 
                        sid, ccode = stare_curenta[r, c] 
              
                        linie_piesa = f"{ro}{co} {sid}{ccode}"
                        linii_fisier.append(linie_piesa)

            with open(path_txt, "w") as f:
                if linii_fisier:
                    for line in linii_fisier:
                        f.write(line + "\n")
                
                f.write(str(scor_runda))
            

            stare_anterioara = stare_curenta


proceseaza_datele('evaluare/fake_test', 'output')






