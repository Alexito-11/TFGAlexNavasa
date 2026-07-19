# quantification_methods.py
# extreu descriptors geometrics "manuals" (forma, curvatura, distancies, gruix) de
# les mascares GT de l'ACDC, per comparar-los despres amb les features de la xarxa

import skimage.io as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial import distance
import os
import nibabel as nib
import math

# PARÀMETRES AJUSTABLES
VAR_THRESHOLD = 0.02  # Mateix valor que config.py de la xarxa neuronal

# True  → usa sempre el GT com a imatge (consistent amb la xarxa neuronal)
# False → usa sempre la imatge real (salta el frame si no existeix)
USE_GT_AS_IMAGE = True


def get_seg_area_pixels(input_img, ground_img):
    # nomes agafa els pixels de la imatge que cauen dins l'etiqueta 2 del GT
    height1, width1 = input_img.shape
    height2, width2 = ground_img.shape
    if width1 == width2 and height1 == height2:
        segmented_pixels = input_img[np.where(ground_img == 2)]
        print("Total Segmentated Pixel:", len(segmented_pixels))
        return segmented_pixels
    else:
        return "Error: The Diminsion of the Input Image and Ground Image is Not same"


def intensity_statistics(input_image, ground_image):
    # estadistiques d'intensitat dins la regio marcada pel GT (no s'usa amb USE_GT_AS_IMAGE=True)
    intensity_values = input_image[np.where(ground_image)]
    mean_intensity   = np.mean(intensity_values)
    median_intensity = np.median(intensity_values)
    std_intensity    = np.std(intensity_values)
    min_intensity    = np.min(intensity_values)
    max_intensity    = np.max(intensity_values)
    return {
        "Mean":          mean_intensity,
        "Median":        median_intensity,
        "STD":           std_intensity,
        "Min_Intensity": min_intensity,
        "Max_Intensity": max_intensity
    }


def shape_description(segmented_image):
    # binaritza amb otsu i treu descriptors de forma basics amb regionprops
    _, binary_mask = cv2.threshold(np.uint8(segmented_image), 0, 1,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask_int = binary_mask.astype(np.uint8)
    props = measure.regionprops(binary_mask_int)
    perimeter    = props[0].perimeter
    area         = props[0].area
    circularity  = (4 * np.pi * area) / (perimeter ** 2)
    eccentricity = props[0].eccentricity
    solidity     = props[0].solidity
    convex_area  = props[0].area_convex
    convexity    = area / convex_area
    return {
        "perimeter":    perimeter,
        "area":         area,
        "circularity":  circularity,
        "eccentricity": eccentricity,
        "solidity":     solidity,
        "convexity":    convexity
    }


def texture_feature(segmented_image):
    # descriptors de textura via GLCM (no s'usa amb USE_GT_AS_IMAGE=True, veure nota mes avall)
    grayco_M    = segmented_image.astype(np.uint8)
    levels      = grayco_M.max() + 1
    glcm        = graycomatrix(grayco_M, distances=[5], angles=[0],
                               levels=levels, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return {"contrast": contrast, "Homogeneity": homogeneity}


def irregular_shape_Centroid_radius(segmented_image):
    # calcula el radi mitja del contorn respecte al centroide (util per formes irregulars)
    contours, _ = cv2.findContours(np.uint8(segmented_image),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"centroid_radius": 0.0}
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        return {"centroid_radius": 0.0}
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])
    distances  = [np.linalg.norm(p[0] - [centroid_x, centroid_y])
                  for p in largest_contour]
    return {"centroid_radius": np.mean(distances)}


def calculate_distances(center_point, boundary_points):
    # helper generic, no cridat directament dins build_metrics_dataframe
    return [np.linalg.norm(center_point - p) for p in boundary_points]


def distanceMap_Center(segmented_image):
    # busca el "centre" com el punt mes allunyat del contorn (centre inscrit), i mesura
    # la distancia min/max del contorn respecte aquest centre
    contours, _ = cv2.findContours(np.uint8(segmented_image),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"min_distance": 0.0, "max_distance": 0.0}
    segmented_contour = max(contours, key=cv2.contourArea)
    if segmented_contour.shape[0] < 2:
        return {"min_distance": 0.0, "max_distance": 0.0}
    mask = np.zeros(segmented_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [segmented_contour], -1, 255, thickness=cv2.FILLED)
    segmented_points = np.argwhere(mask == 255)
    if segmented_points.size == 0:
        return {"min_distance": 0.0, "max_distance": 0.0}
    boundary_rc      = segmented_contour.squeeze().reshape(-1, 2)[:, [1, 0]]
    dist_all         = distance.cdist(segmented_points, boundary_rc)
    distance_min_map = np.zeros_like(segmented_image, dtype=float)
    distance_min_map[segmented_points[:, 0], segmented_points[:, 1]] = dist_all.min(axis=1)
    center           = np.column_stack(np.unravel_index(
        distance_min_map.argmax(), distance_min_map.shape))[0]
    dists_center     = np.linalg.norm(boundary_rc - center, axis=1)
    return {
        "min_distance": float(dists_center.min()),
        "max_distance": float(dists_center.max())
    }


def mean_contour_curvature(binary_mask):
    # curvatura mitjana del contorn: angle entre vectors consecutius dividit per la longitud d'arc
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.nan
    cnt = contours[0][:, 0, :]
    n   = len(cnt)
    if n < 3:
        return np.nan
    kappa_vals = []
    for i in range(n):
        p_prev = cnt[i - 1].astype(float)
        p_curr = cnt[i].astype(float)
        p_next = cnt[(i + 1) % n].astype(float)
        v1, v2 = p_curr - p_prev, p_next - p_curr
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_ang = np.dot(v1, v2) / (n1 * n2)
        ang     = math.acos(np.clip(cos_ang, -1.0, 1.0))
        ds      = 0.5 * (n1 + n2)
        kappa_vals.append(ang / ds)
    return np.nanmean(kappa_vals) if kappa_vals else np.nan


def get_acdc(path):
    # recorre tots els pacients, llegeix Info.cfg (ED/ES frame, grup, alçada, pes)
    # i carrega nomes els frames ED i ES amb la seva mascara GT, slice a slice
    all_imgs        = []
    all_gt          = []
    info            = []
    slice_numbers   = []
    patient_groups  = []
    patient_heights = []
    patient_weights = []

    for patient_folder in sorted(os.listdir(path)):
        patient_path = os.path.join(path, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        info_cfg_path = os.path.join(patient_path, "Info.cfg")
        ed_frame    = -1
        es_frame    = -1
        height      = np.nan
        weight      = np.nan
        group_label = "Unknown"

        if os.path.exists(info_cfg_path):
            with open(info_cfg_path, "r") as f:
                for line in f.read().splitlines():
                    if line.startswith("ED:"):
                        ed_frame    = int(line.split(":")[1].strip())
                    elif line.startswith("ES:"):
                        es_frame    = int(line.split(":")[1].strip())
                    elif line.startswith("Group:"):
                        group_label = line.split(":")[1].strip()
                    elif line.startswith("Height:"):
                        height      = float(line.split(":")[1].strip())
                    elif line.startswith("Weight:"):
                        weight      = float(line.split(":")[1].strip())
        else:
            print(f"Warning: Info.cfg not found for {patient_folder}.")

        # agrupo els fitxers per numero de frame, separant imatge real i mascara gt
        frame_files_map = {}
        for file in os.listdir(patient_path):
            if ".gz" in file and "frame" in file:
                frame_num = int(file.split("frame")[1][:2])
                if frame_num not in frame_files_map:
                    frame_files_map[frame_num] = {'img_file': None, 'gt_file': None}
                if "_gt" not in file:
                    frame_files_map[frame_num]['img_file'] = file
                else:
                    frame_files_map[frame_num]['gt_file'] = file

        for frame_num in sorted(frame_files_map.keys()):
            # Processar NOMÉS els frames ED i ES
            if frame_num != ed_frame and frame_num != es_frame:
                continue

            phase_type = "ED" if frame_num == ed_frame else "ES"
            gt_file    = frame_files_map[frame_num].get('gt_file')
            img_file   = frame_files_map[frame_num].get('img_file')

            if not gt_file:
                print(f"Warning: GT missing for {patient_folder} frame {frame_num}.")
                continue

            gt_nifti  = nib.load(os.path.join(patient_path, gt_file))
            gt_data   = gt_nifti.get_fdata()

            # segons USE_GT_AS_IMAGE, la "imatge" pot ser directament la mascara GT
            # o la imatge real de resonancia
            if USE_GT_AS_IMAGE:
                img_data = gt_data
            else:
                if not img_file:
                    print(f"Warning: Image missing for {patient_folder} frame {frame_num}.")
                    continue
                img_nifti = nib.load(os.path.join(patient_path, img_file))
                img_data  = img_nifti.get_fdata()

            num_slices = gt_data.shape[2]
            for idx in range(num_slices):
                all_imgs.append(img_data[:, :, idx])
                all_gt.append(gt_data[:, :, idx])
                info.append(f"{patient_folder}_{phase_type}")
                patient_groups.append(group_label)
                patient_heights.append(height)
                patient_weights.append(weight)
                slice_numbers.append(idx)

    print("Done loading ACDC data.")
    return [all_imgs, all_gt, info, slice_numbers,
            patient_groups, patient_heights, patient_weights]


def build_metrics_dataframe(acdc_data, GT_LABEL):
    # per cada slice calcula tots els descriptors geometrics de l'etiqueta GT_LABEL
    # (1=VD, 2=miocardi, 3=VE segons conveni ACDC) i els retorna en un dataframe
    input_imgs, g_imgs, info_slices, slice_numbers, \
        patient_groups, patient_heights, patient_weights = acdc_data

    all_metrics_data = []
    slices_descartades = 0

    for i in range(len(input_imgs)):
        input_slice          = input_imgs[i]
        ground_truth         = g_imgs[i]
        patient_id_from_info = info_slices[i]
        current_slice_no     = slice_numbers[i]

        # FILTRE DE VARIÀNCIA (igual que la xarxa neuronal)
        if input_slice.var() < VAR_THRESHOLD:
            slices_descartades += 1
            continue

        # nomes conservo els pixels de l'etiqueta GT_LABEL, la resta a 0, i binaritzo
        segmented_region = np.where(ground_truth == GT_LABEL, input_slice, 0)
        _, segmented_region = cv2.threshold(np.uint8(segmented_region), 0, 1,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        segmented_region = segmented_region.astype(np.uint8)

        props = measure.regionprops(segmented_region)
        if not props:
            continue

        region       = props[0]
        perimeter    = region.perimeter
        area         = region.area
        circularity  = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan
        eccentricity = region.eccentricity
        solidity     = region.solidity
        convex_area  = region.area_convex
        convexity    = area / convex_area if convex_area > 0 else np.nan
        major_len    = region.axis_major_length
        minor_len    = region.axis_minor_length
        elongation   = major_len / minor_len if minor_len > 0 else np.nan

        mask_gt = (ground_truth == GT_LABEL).astype(np.uint8)
        if np.sum(mask_gt) == 0:
            continue

        # nomes calculo gruix per l'etiqueta 2 (miocardi), te sentit fisic nomes per aquesta estructura
        if GT_LABEL == 2:
            dmap          = cv2.distanceTransform(mask_gt * 255, cv2.DIST_L2, 3)
            thickness_map = 2.0 * dmap
            t_vals        = thickness_map[mask_gt.astype(bool)]
            thickness_dict = {
                'thickness_max':  t_vals.max()  if t_vals.size else np.nan,
                'thickness_min':  t_vals.min()  if t_vals.size else np.nan,
                'thickness_mean': t_vals.mean() if t_vals.size else np.nan,
                'thickness_std':  t_vals.std()  if t_vals.size else np.nan
            }
        else:
            thickness_dict = {}

        mean_curvature           = mean_contour_curvature(mask_gt)
        centroid_results         = irregular_shape_Centroid_radius(mask_gt)
        centroid_radius          = centroid_results["centroid_radius"]
        distance_results         = distanceMap_Center(mask_gt)
        min_distance_to_boundary = distance_results["min_distance"]
        max_distance_to_boundary = distance_results["max_distance"]

        # NOTA: els descriptors d'intensitat (Mean, Median, STD, Min/Max Intensity)
        # i de textura basats en GLCM (contrast, Homogeneity) s'han eliminat perquè,
        # en treballar amb la màscara GT com a imatge, tots els píxels d'una mateixa
        # estructura tenen el mateix valor (1, 2 o 3). Això fa que aquests descriptors
        # siguin constants per construcció i no aportin cap informació discriminativa.

        current_slice_data = {
            'Patient_Information': patient_id_from_info,
            'Slice_No':            f"Slice_{current_slice_no}",
            'centroid_radius':     centroid_radius,
            'min_distance':        min_distance_to_boundary,
            'max_distance':        max_distance_to_boundary,
            'perimeter':           perimeter,
            'area':                area,
            'circularity':         circularity,
            'eccentricity':        eccentricity,
            'solidity':            solidity,
            'convexity':           convexity,
            'curvature':           mean_curvature,
            'Elongation':          elongation,
            'Patient_group':       patient_groups[i],
            'Height':              patient_heights[i],
            'Weight':              patient_weights[i]
        }
        all_metrics_data.append(current_slice_data)

    print(f"    Slices descartades per variància < {VAR_THRESHOLD}: {slices_descartades}")
    return pd.DataFrame(all_metrics_data)
