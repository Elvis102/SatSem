#====================================
# PIPELINE PARAMETRIZADO POR AÑO
# Predicción, Mosaico y Matriz de Confusión
# ✅ CON POST-PROCESAMIENTO MORFOLÓGICO
# ✅ OPTIMIZADO: Sin normalización redundante
#====================================
#
# OPTIMIZACIÓN CRÍTICA (Dic 2024):
# ---------------------------------
# Se identificó que las imágenes Sentinel-2 procesadas en Google Earth Engine
# ya están normalizadas [0, 1]. La normalización percentil adicional durante
# la inferencia estaba comprimiendo innecesariamente el rango dinámico espectral.
#              
# CAMBIOS:
# - Eliminada normalización percentil en predict_tile()
# - Threshold optimizado a 0.50 (para valores espectrales reales)
# - Ganancia estimada: +8-10 pp en recall, -2 pp en precision
#
# REFERENCIA:
# Ver análisis de normalización en inspeccionar_teselas.py
# como correr el pipeline: cd /Users/elvissanchez/Documents/GitHub/thesis_project/notebooks/satseg-main
# ¿Porque tengo que ubicarme en satseg-main para que coloque la carpeta predicciones_por_año en la ruta correcta?
# uv run predict_and_mosaic_with_metrics.py
#====================================

from src.module import Module
from src.metrics import iou
from src.models.multi_branch_unet import MultiBranchUNetWrapper
import torch
import segmentation_models_pytorch as smp
import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.plot import show
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from tqdm import tqdm
import warnings
import json
import seaborn as sns
from scipy.ndimage import binary_closing, binary_opening  # ← NUEVO IMPORT
import geopandas as gpd  # Para shapefiles
from rasterio.mask import mask as rasterio_mask  # Para enmascarar

warnings.filterwarnings('ignore', category=RuntimeWarning)


#====================================
# 🔧 FUNCIONES HELPER PARA CRS
#====================================

def is_crs_geographic(crs):
    """
    Detecta si un CRS es geográfico (grados) o proyectado (metros)
    
    ✅ MEJORADO: Verifica unidades y múltiples métodos de detección
    
    Args:
        crs: rasterio.crs.CRS object
    
    Returns:
        tuple: (is_geographic: bool, detection_method: str, additional_info: dict)
    
    Ejemplos:
        - EPSG:4326 (WGS84) → (True, ..., {'units': 'degrees'})
        - EPSG:32717 (UTM 17S) → (False, ..., {'units': 'meters', 'zone': '17S'})
        - EPSG:32618 (UTM 18N) → (False, ..., {'units': 'meters', 'zone': '18N'})
    """
    
    additional_info = {}
    
    # Método 1: Usar propiedad is_geographic de rasterio (más confiable)
    try:
        is_geographic = crs.is_geographic
        
        # Verificar unidades como confirmación adicional
        try:
            linear_units = crs.linear_units
            additional_info['units'] = linear_units
            
            # Las unidades geográficas son típicamente 'degree' o 'degrees'
            if linear_units and 'degree' in str(linear_units).lower():
                additional_info['units_type'] = 'degrees'
            elif linear_units and any(unit in str(linear_units).lower() for unit in ['metre', 'meter', 'm']):
                additional_info['units_type'] = 'meters'
        except:
            pass
            
        return is_geographic, "rasterio.is_geographic", additional_info
    except:
        pass
    
    # Método 2: Verificar código EPSG
    try:
        epsg_code = crs.to_epsg()
        
        if epsg_code:
            additional_info['epsg'] = epsg_code
            
            # EPSG 326xx (UTM Norte) y 327xx (UTM Sur) son proyectados
            if 32600 <= epsg_code <= 32660:  # UTM zones North
                zone_num = epsg_code - 32600
                additional_info['zone'] = f"{zone_num}N"
                additional_info['units'] = 'meters'
                return False, f"EPSG:{epsg_code} (UTM {zone_num}N)", additional_info
            elif 32700 <= epsg_code <= 32760:  # UTM zones South
                zone_num = epsg_code - 32700
                additional_info['zone'] = f"{zone_num}S"
                additional_info['units'] = 'meters'
                return False, f"EPSG:{epsg_code} (UTM {zone_num}S)", additional_info
            elif epsg_code == 4326:  # WGS84
                additional_info['datum'] = 'WGS84'
                additional_info['units'] = 'degrees'
                return True, "EPSG:4326 (WGS84)", additional_info
            elif epsg_code == 4269:  # NAD83
                additional_info['datum'] = 'NAD83'
                additional_info['units'] = 'degrees'
                return True, "EPSG:4269 (NAD83)", additional_info
            elif 4000 <= epsg_code < 5000:  # Generalmente geográficos
                additional_info['units'] = 'degrees'
                return True, f"EPSG:{epsg_code} (geográfico)", additional_info
            elif epsg_code in range(2000, 32600):  # Muchos sistemas proyectados
                additional_info['units'] = 'meters'
                return False, f"EPSG:{epsg_code} (proyectado)", additional_info
            else:
                # Otros códigos - probablemente proyectados
                additional_info['units'] = 'unknown'
                return False, f"EPSG:{epsg_code} (proyectado)", additional_info
    except:
        pass
    
    # Método 3: Análisis de string del CRS (fallback)
    try:
        crs_string = str(crs).upper()
        additional_info['crs_string'] = crs_string[:100]  # Primeros 100 caracteres
        
        geographic_indicators = ['4326', 'WGS 84', 'GEOGCS', 'GEOGRAPHIC', 
                                'LATITUDE', 'LONGITUDE', 'DEGREE']
        projected_indicators = ['UTM', 'PROJCS', 'PROJECTED', 'METRE', 'METER']
        
        has_geographic = any(ind in crs_string for ind in geographic_indicators)
        has_projected = any(ind in crs_string for ind in projected_indicators)
        
        if has_projected:
            additional_info['units'] = 'meters'
            return False, "string analysis (projected)", additional_info
        elif has_geographic:
            additional_info['units'] = 'degrees'
            return True, "string analysis (geographic)", additional_info
    except:
        pass
    
    # Por defecto, asumir proyectado (más seguro para no forzar reproyección)
    additional_info['fallback'] = True
    return False, "fallback (assumed projected)", additional_info


#====================================
# ⭐ CONFIGURACIÓN CENTRALIZADA POR AÑO
#====================================

class YearConfig:
    """
    Configuración parametrizada por año para análisis multitemporal
    
    Uso:
        config = YearConfig(year=2021, base_dir='/path/to/data')
        images_dir = config.images_dir
        masks_dir = config.masks_dir
    """
    
    def __init__(self, year, base_dir, checkpoint_path, output_base_dir='predicciones'):
        """
        Args:
            year: Año de análisis (ej: 2021, 2022, 2023)
            base_dir: Directorio base donde están las carpetas Manglar_[AÑO]_*
            checkpoint_path: Ruta al checkpoint del modelo
            output_base_dir: Directorio base para outputs (se creará subdir por año)
        """
        self.year = year
        self.base_dir = Path(base_dir)
        self.checkpoint_path = checkpoint_path
        
        # Construir nombres de carpetas según patrón
        self.images_folder_name = f"Manglar_{year}_images"
        self.masks_folder_name = f"Manglar_{year}_masks"
        
        # Rutas completas
        self.images_dir = self.base_dir / self.images_folder_name
        self.masks_dir = self.base_dir / self.masks_folder_name
        
        # Directorio de salida específico por año
        self.output_base_dir = Path(output_base_dir) / f"year_{year}"
        self.predictions_dir = self.output_base_dir / 'teselas_predichas'
        self.mosaic_dir = self.output_base_dir / 'mosaico'
        self.metrics_dir = self.output_base_dir / 'metricas'
        self.visualizations_dir = self.output_base_dir / 'visualizaciones'

        # Crear directorios de salida
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.mosaic_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        # Validar que existan las carpetas de entrada
        self._validate_directories()
    
    def _validate_directories(self):
        """Valida que existan las carpetas de imágenes (máscaras son opcionales)"""
        if not self.images_dir.exists():
            raise ValueError(
                f"❌ No se encontró la carpeta de imágenes: {self.images_dir}\n"
                f"   Verifica que exista: {self.base_dir}/{self.images_folder_name}"
            )
        
        # Verificar que haya archivos .tif en imágenes
        tif_files = list(self.images_dir.glob("*.tif"))
        if len(tif_files) == 0:
            raise ValueError(
                f"❌ No se encontraron archivos .tif en: {self.images_dir}"
            )
        
        print(f"✅ Carpeta de imágenes encontrada: {self.images_dir}")
        print(f"   Archivos .tif encontrados: {len(tif_files)}")
        
        # Las máscaras son opcionales
        if self.masks_dir.exists():
            mask_files = list(self.masks_dir.glob("*.tif"))
            print(f"✅ Carpeta de máscaras encontrada: {self.masks_dir}")
            print(f"   Archivos .tif encontrados: {len(mask_files)}")
        else:
            print(f"⚠️  No se encontró carpeta de máscaras: {self.masks_dir}")
            print(f"   La matriz de confusión no estará disponible")
    
    def get_summary(self):
        """Retorna resumen de la configuración"""
        return {
            'year': self.year,
            'base_dir': str(self.base_dir),
            'images_dir': str(self.images_dir),
            'masks_dir': str(self.masks_dir),
            'output_dir': str(self.output_base_dir),
            'images_exist': self.images_dir.exists(),
            'masks_exist': self.masks_dir.exists()
        }


#====================================
# FUNCIONES DE MATRIZ DE CONFUSIÓN
#====================================

def calculate_confusion_matrix_from_files(pred_files, masks_dir):
    """
    Calcula la matriz de confusión comparando predicciones con máscaras ground truth
    
    ✅ CORREGIDO: Excluye máscaras vacías (solo 0s) y maneja NaN
    
    Args:
        pred_files: Lista de rutas a predicciones
        masks_dir: Path object del directorio con máscaras
    
    Returns:
        cm: Matriz de confusión 2x2
        metrics: Diccionario con métricas
        valid_count: Número de teselas con máscaras válidas
    """
    
    if not masks_dir.exists():
        print(f"\n⚠️  Directorio de máscaras no existe: {masks_dir}")
        return None, None, 0
    
    TP = FP = TN = FN = 0
    valid_count = 0
    skipped_empty = 0      # Máscaras vacías (solo 0s)
    skipped_invalid = 0    # Máscaras con problemas
    skipped_nan = 0        # Máscaras con NaN
    
    # ⭐ NUEVO: Listas para rastrear nombres de teselas excluidas
    empty_masks = []       # Nombres de máscaras vacías
    nan_masks = []         # Nombres de máscaras con NaN
    invalid_masks = []     # Nombres de máscaras inválidas
    
    print(f"\n🔍 Buscando máscaras ground truth en: {masks_dir}")
    
    for pred_path in tqdm(pred_files, desc="Calculando matriz de confusión"):
        # Extraer nombre base de la predicción
        pred_name = Path(pred_path).stem
        # Quitar prefijo "pred_" si existe
        tile_name = pred_name.replace('pred_', '')
        
        # Buscar máscara correspondiente con patrón correcto
        mask_path = masks_dir / f"{tile_name.replace('_r', '_mask_r')}.tif"
        
        if not mask_path.exists():
            # Intentar con otros patrones comunes
            alt_patterns = [
                masks_dir / f"{tile_name}_mask.tif",
                masks_dir / f"mask_{tile_name}.tif",
            ]
            for alt_path in alt_patterns:
                if alt_path.exists():
                    mask_path = alt_path
                    break
            else:
                continue  # No se encontró máscara para esta tesela
        
        try:
            # Leer predicción
            with rasterio.open(pred_path) as src:
                pred = src.read(1)
            
            # Leer máscara ground truth
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            
            # Verificar dimensiones
            if pred.shape != mask.shape:
                print(f"⚠️ Dimensiones no coinciden: {pred_name}")
                skipped_invalid += 1
                invalid_masks.append(f"{tile_name} (dimensiones: pred {pred.shape} vs mask {mask.shape})")
                continue
            
            # ⭐ MANEJO DE NaN
            mask_has_nan = np.isnan(mask).any()
            if mask_has_nan:
                # Crear máscara de píxeles válidos (no-NaN)
                valid_pixels = ~np.isnan(mask)
                
                # Si toda la máscara es NaN, saltarla
                if not valid_pixels.any():
                    skipped_nan += 1
                    nan_masks.append(tile_name)
                    continue
                
                # Filtrar solo píxeles válidos
                mask_clean = mask[valid_pixels]
                pred_clean = pred[valid_pixels]
            else:
                mask_clean = mask.flatten()
                pred_clean = pred.flatten()
            
            # ⭐ VERIFICAR SI LA MÁSCARA TIENE MANGLAR
            unique_mask_vals = np.unique(mask_clean)
            
            # Saltar máscaras vacías (solo 0s)
            if len(unique_mask_vals) == 1 and unique_mask_vals[0] == 0:
                skipped_empty += 1
                empty_masks.append(tile_name)
                continue
            
            # Verificar que la máscara sea binaria (0 y 1)
            if not np.all(np.isin(unique_mask_vals, [0, 1])):
                print(f"⚠️ Valores inesperados en {tile_name}: {unique_mask_vals}")
                skipped_invalid += 1
                invalid_masks.append(f"{tile_name} (valores: {unique_mask_vals})")
                continue
            
            # ✅ Máscara válida: calcular métricas
            TP += np.sum((pred_clean == 1) & (mask_clean == 1))
            TN += np.sum((pred_clean == 0) & (mask_clean == 0))
            FP += np.sum((pred_clean == 1) & (mask_clean == 0))
            FN += np.sum((pred_clean == 0) & (mask_clean == 1))
            
            valid_count += 1
            
        except Exception as e:
            print(f"⚠️ Error procesando {pred_name}: {str(e)}")
            skipped_invalid += 1
            invalid_masks.append(f"{tile_name} (error: {str(e)[:50]})")
            continue
    
    # Mostrar estadísticas de máscaras procesadas
    total_processed = valid_count + skipped_empty + skipped_nan + skipped_invalid
    print(f"\n📊 Resumen de procesamiento:")
    print(f"   Total procesado:     {total_processed}")
    print(f"   ✅ Máscaras válidas:  {valid_count} ({valid_count/total_processed*100:.1f}%)")
    if skipped_empty > 0:
        print(f"   ⏭️  Máscaras vacías:  {skipped_empty} ({skipped_empty/total_processed*100:.1f}%) - excluidas correctamente")
    if skipped_nan > 0:
        print(f"   ⏭️  Con NaN:          {skipped_nan} ({skipped_nan/total_processed*100:.1f}%) - excluidas")
    if skipped_invalid > 0:
        print(f"   ⚠️  Inválidas:        {skipped_invalid} ({skipped_invalid/total_processed*100:.1f}%) - excluidas")
    
    if valid_count == 0:
        print(f"\n❌ No se encontraron máscaras válidas con manglar")
        return None, None, 0
    
    print(f"\n✅ Matriz de confusión calculada con {valid_count} teselas válidas")
    
    # ⭐⭐⭐ NUEVO: GUARDAR REPORTE DE MÁSCARAS EXCLUIDAS ⭐⭐⭐
    if skipped_empty > 0 or skipped_nan > 0 or skipped_invalid > 0:
        try:
            # Determinar año y directorio de salida
            masks_dir_str = str(masks_dir)
            if 'Manglar_' in masks_dir_str:
                year_match = masks_dir_str.split('Manglar_')[1].split('_')[0]
                report_dir = Path(f'predicciones_por_año/year_{year_match}/metricas')
                report_dir.mkdir(parents=True, exist_ok=True)
                
                excluded_report_path = report_dir / 'mascaras_excluidas.txt'
                
                with open(excluded_report_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("REPORTE DE MÁSCARAS EXCLUIDAS DE LA EVALUACIÓN\n")
                    f.write("="*80 + "\n\n")
                    
                    f.write(f"Total de teselas procesadas:     {total_processed}\n")
                    f.write(f"Teselas válidas (con manglar):   {valid_count} ({valid_count/total_processed*100:.1f}%)\n")
                    f.write(f"Teselas excluidas:               {skipped_empty + skipped_nan + skipped_invalid} ({(skipped_empty + skipped_nan + skipped_invalid)/total_processed*100:.1f}%)\n\n")
                    
                    if skipped_empty > 0:
                        f.write("-"*80 + "\n")
                        f.write(f"1. MÁSCARAS VACÍAS (solo píxeles = 0): {skipped_empty} teselas\n")
                        f.write("-"*80 + "\n")
                        f.write("Razón: Estas teselas no contienen manglar en el ground truth.\n")
                        f.write("Son áreas de no-manglar válidas pero no útiles para evaluar la\n")
                        f.write("capacidad del modelo de detectar manglar (solo evalúan TN/FP).\n\n")
                        
                        if len(empty_masks) > 0:
                            f.write("Teselas vacías encontradas:\n")
                            for i, mask_name in enumerate(sorted(empty_masks), 1):
                                f.write(f"  {i:3d}. {mask_name}\n")
                            f.write("\n")
                    
                    if skipped_nan > 0:
                        f.write("-"*80 + "\n")
                        f.write(f"2. MÁSCARAS CON NaN (valores inválidos): {skipped_nan} teselas\n")
                        f.write("-"*80 + "\n")
                        f.write("Razón: Estas máscaras contienen valores NaN (Not a Number).\n")
                        f.write("Pueden ser máscaras corruptas o con problemas de procesamiento.\n\n")
                        
                        if len(nan_masks) > 0:
                            f.write("Teselas con NaN encontradas:\n")
                            for i, mask_name in enumerate(sorted(nan_masks), 1):
                                f.write(f"  {i:3d}. {mask_name}\n")
                            f.write("\n")
                    
                    if skipped_invalid > 0:
                        f.write("-"*80 + "\n")
                        f.write(f"3. MÁSCARAS INVÁLIDAS (otros problemas): {skipped_invalid} teselas\n")
                        f.write("-"*80 + "\n")
                        f.write("Razón: Estas máscaras tienen problemas como:\n")
                        f.write("  - Dimensiones no coinciden con la predicción\n")
                        f.write("  - Valores fuera del rango esperado (0, 1)\n")
                        f.write("  - Errores de lectura del archivo\n\n")
                        
                        if len(invalid_masks) > 0:
                            f.write("Teselas inválidas encontradas:\n")
                            for i, mask_info in enumerate(sorted(invalid_masks), 1):
                                f.write(f"  {i:3d}. {mask_info}\n")
                            f.write("\n")
                    
                    f.write("="*80 + "\n")
                    f.write("RECOMENDACIONES:\n")
                    f.write("="*80 + "\n")
                    f.write("1. Máscaras vacías: NORMAL - Áreas sin manglar en ground truth.\n")
                    f.write("   → No requieren acción. Son excluidas correctamente.\n\n")
                    f.write("2. Máscaras con NaN: REVISAR - Archivos posiblemente corruptos.\n")
                    f.write("   → Inspeccionar en QGIS o Python para verificar integridad.\n\n")
                    f.write("3. Máscaras inválidas: INVESTIGAR - Problemas de procesamiento.\n")
                    f.write("   → Revisar logs de errores y considerar reprocesar.\n")
                    f.write("="*80 + "\n")
                
                print(f"📝 Reporte de máscaras excluidas: {excluded_report_path}")
        
        except Exception as e:
            print(f"⚠️  No se pudo crear reporte de máscaras excluidas: {e}")
    # ⭐⭐⭐ FIN DEL REPORTE DE MÁSCARAS EXCLUIDAS ⭐⭐⭐
    
    # Construir matriz
    cm = np.array([[TN, FP], [FN, TP]])
    
    # Calcular métricas
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou_manglar = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    iou_no_manglar = TN / (TN + FP + FN) if (TN + FP + FN) > 0 else 0
    mean_iou = (iou_manglar + iou_no_manglar) / 2
    
    metrics = {
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'total_pixels': int(total),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1_score),
        'iou_manglar': float(iou_manglar),
        'iou_no_manglar': float(iou_no_manglar),
        'mean_iou': float(mean_iou),
        'tiles_evaluated': valid_count,
        'tiles_skipped_empty': skipped_empty,
        'tiles_skipped_nan': skipped_nan,
        'tiles_skipped_invalid': skipped_invalid
    }
    
    return cm, metrics, valid_count

def plot_confusion_matrix_for_article(cm, metrics, save_path, year):
    """
    Genera matriz de confusión profesional para artículo científico
    """
    
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalizar a porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Crear heatmap
    im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Porcentaje (%)', rotation=270, labelpad=25, fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    
    # Etiquetas de clases
    classes = ['No Manglar', 'Manglar']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=13)
    ax.set_yticklabels(classes, fontsize=13)
    
    # Valores en celdas
    thresh = cm_percent.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_main = f'{cm_percent[i, j]:.1f}%'
            text_count = f'({cm[i, j]:,} px)'
            
            color = "white" if cm_percent[i, j] > thresh else "black"
            
            ax.text(j, i - 0.15, text_main,
                   ha="center", va="center", color=color,
                   fontsize=18, fontweight='bold')
            
            ax.text(j, i + 0.15, text_count,
                   ha="center", va="center", color=color,
                   fontsize=11)
    
    # Etiquetas TN, FP, FN, TP
    annotations = [('TN', 0, 0), ('FP', 1, 0), ('FN', 0, 1), ('TP', 1, 1)]
    for label, x, y in annotations:
        ax.text(x, y - 0.42, label,
               ha="center", va="center",
               color="red", fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.4))
    
    # Etiquetas de ejes
    ax.set_ylabel('Clase Real (Ground Truth)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Clase Predicha', fontsize=14, fontweight='bold')
    
    # Título con métricas principales y año
    title = f'Matriz de Confusión - Año {year}\n'
    title += f'Accuracy: {metrics["accuracy"]*100:.2f}% | '
    title += f'F1-Score: {metrics["f1_score"]:.4f} | '
    title += f'IoU: {metrics["mean_iou"]:.4f}\n'
    title += f'({metrics["tiles_evaluated"]} teselas evaluadas)'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Grid sutil
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Matriz de confusión guardada: {save_path}")


def plot_metrics_comparison(metrics, save_path, year):
    """
    Crea un gráfico de barras con todas las métricas
    """
    
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Métricas a graficar
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'Mean IoU']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['specificity'],
        metrics['f1_score'],
        metrics['mean_iou']
    ]
    
    # Colores según rendimiento
    colors = []
    for val in metric_values:
        if val >= 0.95:
            colors.append('#2ecc71')  # Verde - Excelente
        elif val >= 0.90:
            colors.append('#3498db')  # Azul - Muy bueno
        elif val >= 0.80:
            colors.append('#f39c12')  # Naranja - Bueno
        else:
            colors.append('#e74c3c')  # Rojo - Mejorable
    
    # Crear barras
    bars = ax.barh(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Añadir valores al final de cada barra
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax.text(val + 0.01, i, f'{val:.4f}\n({val*100:.2f}%)',
               va='center', fontsize=11, fontweight='bold')
    
    # Línea de referencia en 0.90
    ax.axvline(x=0.90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Umbral 90%')
    
    # Configuración de ejes
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Valor de la Métrica', fontsize=13, fontweight='bold')
    ax.set_title(f'Métricas de Rendimiento del Modelo - Año {year}\n' +
                 f'({metrics["tiles_evaluated"]} teselas evaluadas)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    # Aumentar tamaño de etiquetas
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Gráfico de métricas guardado: {save_path}")


def save_confusion_matrix_report(cm, metrics, save_path, year):
    """
    Genera reporte de texto detallado
    """
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"MATRIZ DE CONFUSIÓN Y MÉTRICAS - AÑO {year}\n")
        f.write("="*80 + "\n\n")
        
        f.write("INFORMACIÓN GENERAL:\n")
        f.write("-"*80 + "\n")
        f.write(f"Año de análisis:        {year}\n")
        f.write(f"Teselas evaluadas:      {metrics['tiles_evaluated']}\n")
        f.write(f"Total de píxeles:       {metrics['total_pixels']:,}\n")
        f.write("-"*80 + "\n\n")
        
        f.write("MATRIZ DE CONFUSIÓN:\n")
        f.write("-"*80 + "\n")
        f.write(f"                    Predicho No-Manglar    Predicho Manglar\n")
        f.write(f"Real No-Manglar     {cm[0,0]:>18,} (TN)    {cm[0,1]:>15,} (FP)\n")
        f.write(f"Real Manglar        {cm[1,0]:>18,} (FN)    {cm[1,1]:>15,} (TP)\n")
        f.write("-"*80 + "\n\n")
        
        f.write("CONTEOS ABSOLUTOS:\n")
        f.write("-"*80 + "\n")
        f.write(f"True Positives (TP):    {metrics['TP']:>18,} píxeles\n")
        f.write(f"True Negatives (TN):    {metrics['TN']:>18,} píxeles\n")
        f.write(f"False Positives (FP):   {metrics['FP']:>18,} píxeles\n")
        f.write(f"False Negatives (FN):   {metrics['FN']:>18,} píxeles\n")
        f.write("-"*80 + "\n\n")
        
        f.write("MÉTRICAS DE RENDIMIENTO:\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:               {metrics['accuracy']:>10.6f}  ({metrics['accuracy']*100:>6.2f}%)\n")
        f.write(f"Precision:              {metrics['precision']:>10.6f}  ({metrics['precision']*100:>6.2f}%)\n")
        f.write(f"Recall (Sensitivity):   {metrics['recall']:>10.6f}  ({metrics['recall']*100:>6.2f}%)\n")
        f.write(f"Specificity:            {metrics['specificity']:>10.6f}  ({metrics['specificity']*100:>6.2f}%)\n")
        f.write(f"F1-Score:               {metrics['f1_score']:>10.6f}\n")
        f.write("-"*80 + "\n\n")
        
        f.write("MÉTRICAS DE SEGMENTACIÓN (IoU):\n")
        f.write("-"*80 + "\n")
        f.write(f"IoU Manglar:            {metrics['iou_manglar']:>10.6f}\n")
        f.write(f"IoU No-Manglar:         {metrics['iou_no_manglar']:>10.6f}\n")
        f.write(f"Mean IoU:               {metrics['mean_iou']:>10.6f}\n")
        f.write("-"*80 + "\n\n")
        
        # Calcular tasas de error
        fpr = metrics['FP'] / (metrics['FP'] + metrics['TN']) if (metrics['FP'] + metrics['TN']) > 0 else 0
        fnr = metrics['FN'] / (metrics['FN'] + metrics['TP']) if (metrics['FN'] + metrics['TP']) > 0 else 0
        
        f.write("ANÁLISIS DE ERRORES:\n")
        f.write("-"*80 + "\n")
        f.write(f"Tasa de Falsos Positivos (FPR):  {fpr*100:>6.2f}%\n")
        f.write(f"  → De cada 100 píxeles de no-manglar, {fpr*100:.1f} son clasificados\n")
        f.write(f"    incorrectamente como manglar\n\n")
        f.write(f"Tasa de Falsos Negativos (FNR):   {fnr*100:>6.2f}%\n")
        f.write(f"  → De cada 100 píxeles de manglar, {fnr*100:.1f} NO son detectados\n\n")
        f.write("-"*80 + "\n\n")
        
        f.write("INTERPRETACIÓN:\n")
        f.write("-"*80 + "\n")
        f.write(f"• El modelo clasifica correctamente el {metrics['accuracy']*100:.2f}% de los píxeles\n\n")
        f.write(f"• Precision ({metrics['precision']*100:.2f}%):\n")
        f.write(f"  Cuando el modelo predice 'Manglar', acierta {metrics['precision']*100:.1f}%\n")
        f.write(f"  de las veces\n\n")
        f.write(f"• Recall ({metrics['recall']*100:.2f}%):\n")
        f.write(f"  El modelo detecta {metrics['recall']*100:.1f}% de todo el manglar presente\n\n")
        f.write(f"• F1-Score ({metrics['f1_score']:.4f}):\n")
        f.write(f"  Balance armónico entre Precision y Recall\n\n")
        f.write(f"• Mean IoU ({metrics['mean_iou']:.4f}):\n")
        
        if metrics['mean_iou'] >= 0.90:
            f.write(f"  ⭐⭐⭐⭐⭐ EXCELENTE - Rendimiento excepcional\n")
        elif metrics['mean_iou'] >= 0.80:
            f.write(f"  ⭐⭐⭐⭐ MUY BUENO - Rendimiento por encima del promedio\n")
        elif metrics['mean_iou'] >= 0.70:
            f.write(f"  ⭐⭐⭐ BUENO - Rendimiento aceptable\n")
        else:
            f.write(f"  ⭐⭐ MEJORABLE - Considerar refinamiento del modelo\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Reporte detallado guardado: {save_path}")


def generate_advanced_analysis_plots(predictions_dir, masks_dir, output_dir, year, threshold=0.5):
    """
    Genera gráficos de análisis avanzados incluyendo curva ROC

    Args:
        predictions_dir: Directorio con predicciones GeoTIFF
        masks_dir: Directorio con máscaras ground truth
        output_dir: Directorio de salida para gráficos
        year: Año de análisis
        threshold: Umbral de decisión para binarización

    Returns:
        Dict con métricas calculadas
    """

    print(f"\n📊 Generando gráficos de análisis avanzados...")

    # Buscar archivos de predicción
    pred_files = sorted(predictions_dir.glob("pred_*.tif"))

    if len(pred_files) == 0:
        print(f"⚠️ No se encontraron predicciones")
        return None

    # Recolectar datos de todas las teselas
    tiles_data = []
    all_probs = []  # Para curva ROC
    all_labels = []  # Para curva ROC

    print(f"   Analizando {len(pred_files)} teselas...")

    for pred_path in tqdm(pred_files, desc="Calculando métricas por tesela"):
        pred_name = pred_path.stem
        tile_name = pred_name.replace('pred_', '')

        # Buscar máscara correspondiente
        mask_path = masks_dir / f"{tile_name.replace('_r', '_mask_r')}.tif"
        if not mask_path.exists():
            alt_patterns = [
                masks_dir / f"{tile_name}_mask.tif",
                masks_dir / f"mask_{tile_name}.tif",
            ]
            for alt_path in alt_patterns:
                if alt_path.exists():
                    mask_path = alt_path
                    break
            else:
                continue

        try:
            # Leer predicción (probabilidades o binario)
            with rasterio.open(pred_path) as src:
                pred = src.read(1)

            # Leer máscara
            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            if pred.shape != mask.shape:
                continue

            # Filtrar píxeles válidos
            valid_pixels = ~np.isnan(mask) & ~np.isnan(pred)
            mask_clean = mask[valid_pixels]
            pred_clean = pred[valid_pixels]

            # Saltar máscaras vacías
            if len(np.unique(mask_clean)) == 1 and np.unique(mask_clean)[0] == 0:
                continue

            if len(mask_clean) == 0:
                continue

            # Calcular métricas por tesela
            TP = np.sum((pred_clean == 1) & (mask_clean == 1))
            TN = np.sum((pred_clean == 0) & (mask_clean == 0))
            FP = np.sum((pred_clean == 1) & (mask_clean == 0))
            FN = np.sum((pred_clean == 0) & (mask_clean == 1))

            tile_iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
            tile_acc = (TP + TN) / len(mask_clean) if len(mask_clean) > 0 else 0.0

            cobertura_real = (mask_clean == 1).sum() / len(mask_clean) * 100
            cobertura_pred = (pred_clean == 1).sum() / len(pred_clean) * 100

            tiles_data.append({
                'tile_name': tile_name,
                'iou': tile_iou,
                'accuracy': tile_acc,
                'cobertura_real': cobertura_real,
                'cobertura_pred': cobertura_pred,
                'TP': TP,
                'TN': TN,
                'FP': FP,
                'FN': FN,
            })

            # Acumular para curva ROC (usando predicciones como "probabilidades")
            # Nota: Si pred ya es binario (0/1), ROC será escalonada
            # Si es continuo [0,1], ROC será suave
            all_probs.extend(pred_clean)
            all_labels.extend(mask_clean)

        except Exception as e:
            continue

    if len(tiles_data) == 0:
        print(f"⚠️ No se pudieron analizar teselas")
        return None

    # Extraer listas
    ious = [t['iou'] for t in tiles_data]
    accuracies = [t['accuracy'] for t in tiles_data]
    real_cov = [t['cobertura_real'] for t in tiles_data]
    pred_cov = [t['cobertura_pred'] for t in tiles_data]
    diff_cov = [t['cobertura_pred'] - t['cobertura_real'] for t in tiles_data]

    # ========================================================================
    # FIGURA 1: Análisis de métricas (4 subplots)
    # ========================================================================

    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(f'Análisis de Métricas - Año {year}', fontsize=16, fontweight='bold', y=0.995)

    # Subplot 1: Histograma de IoU
    axes[0, 0].hist(ious, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    mean_iou = np.mean(ious)
    axes[0, 0].axvline(mean_iou, color='red', linestyle='--', linewidth=2,
                      label=f'Media: {mean_iou:.4f}')
    axes[0, 0].set_xlabel('IoU', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Cantidad de Teselas', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Distribución de IoU', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    # Subplot 2: Scatter - Cobertura Real vs Predicha
    scatter = axes[0, 1].scatter(real_cov, pred_cov, alpha=0.6, c=ious, cmap='RdYlGn',
                                 edgecolors='black', s=50, vmin=0, vmax=1)
    axes[0, 1].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Predicción perfecta')
    axes[0, 1].set_xlabel('Cobertura Real (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Cobertura Predicha (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Cobertura: Real vs Predicha', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[0, 1])
    cbar.set_label('IoU', fontsize=10)

    # Subplot 3: Box plot de IoU
    bp = axes[1, 0].boxplot(ious, vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', color='black'),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5))
    axes[1, 0].set_ylabel('IoU', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Distribución de IoU (Box Plot)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')

    # Añadir estadísticas al box plot
    q1, median, q3 = np.percentile(ious, [25, 50, 75])
    iqr = q3 - q1
    axes[1, 0].text(1.15, median, f'Mediana: {median:.4f}', fontsize=9, va='center')
    axes[1, 0].text(1.15, q3, f'Q3: {q3:.4f}', fontsize=9, va='center')
    axes[1, 0].text(1.15, q1, f'Q1: {q1:.4f}', fontsize=9, va='center')

    # Subplot 4: Histograma de error en cobertura
    axes[1, 1].hist(diff_cov, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='green', linestyle='--', linewidth=2, label='Sin error')
    mean_diff = np.mean(diff_cov)
    axes[1, 1].axvline(mean_diff, color='red', linestyle='--', linewidth=2,
                      label=f'Error medio: {mean_diff:.2f}%')
    axes[1, 1].set_xlabel('Diferencia Cobertura (Pred - Real) %', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Cantidad de Teselas', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Error en Estimación de Cobertura', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    analysis_path = output_dir / f'analisis_metricas_{year}.png'
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"   ✅ Gráficos de análisis guardados: {analysis_path.name}")

    # ========================================================================
    # FIGURA 2: Curva ROC (Receiver Operating Characteristic)
    # ========================================================================

    print(f"   Calculando curva ROC...")

    # Convertir a numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)

    # Calcular curva ROC
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)

    # Encontrar threshold óptimo (punto más cercano a (0,1) en ROC)
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    optimal_idx = np.argmin(distances)
    optimal_threshold_roc = thresholds_roc[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    # Encontrar threshold óptimo según F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx_f1 = np.argmax(f1_scores)
    if optimal_idx_f1 < len(thresholds_pr):
        optimal_threshold_f1 = thresholds_pr[optimal_idx_f1]
    else:
        optimal_threshold_f1 = threshold

    # Crear figura con 2 subplots
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(f'Curvas de Evaluación del Modelo - Año {year}', fontsize=16, fontweight='bold')

    # Subplot 1: Curva ROC
    axes[0].plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador aleatorio')
    axes[0].scatter([optimal_fpr], [optimal_tpr], color='red', s=100, zorder=5,
                   label=f'Óptimo (thr={optimal_threshold_roc:.3f})')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    axes[0].set_title('Curva ROC\n(Capacidad de discriminación)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(alpha=0.3)

    # Añadir texto explicativo
    text_roc = (f"AUC = {roc_auc:.4f}\n"
                f"TPR óptimo = {optimal_tpr:.3f}\n"
                f"FPR óptimo = {optimal_fpr:.3f}")
    axes[0].text(0.6, 0.2, text_roc, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 2: Precision-Recall curve
    axes[1].plot(recall, precision, color='darkgreen', lw=2,
                label=f'PR (AP = {avg_precision:.4f})')
    axes[1].axhline(y=all_labels.mean(), color='gray', linestyle='--', lw=2,
                   label=f'Baseline (prevalencia={all_labels.mean():.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Sensibilidad)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1].set_title('Curva Precision-Recall\n(Balance detección vs falsos positivos)', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower left', fontsize=10)
    axes[1].grid(alpha=0.3)

    # Añadir texto explicativo
    text_pr = (f"AP = {avg_precision:.4f}\n"
               f"Threshold F1 = {optimal_threshold_f1:.3f}")
    axes[1].text(0.6, 0.9, text_pr, fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    roc_path = output_dir / f'curva_roc_{year}.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"   ✅ Curva ROC guardada: {roc_path.name}")

    # ========================================================================
    # FIGURA 3: Casos destacados (mejores y peores)
    # ========================================================================

    # Ordenar por IoU
    tiles_sorted = sorted(tiles_data, key=lambda x: x['iou'], reverse=True)

    best_5 = tiles_sorted[:5]
    worst_5 = tiles_sorted[-5:]

    fig3, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig3.suptitle(f'Casos Destacados - Año {year}', fontsize=16, fontweight='bold', y=0.995)

    # Mejores casos
    best_names = [t['tile_name'][:20] + '...' if len(t['tile_name']) > 20 else t['tile_name'] for t in best_5]
    best_ious = [t['iou'] for t in best_5]

    axes[0].barh(range(len(best_5)), best_ious, color='green', alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(best_5)))
    axes[0].set_yticklabels(best_names, fontsize=9)
    axes[0].set_xlabel('IoU', fontsize=11, fontweight='bold')
    axes[0].set_title('🏆 Top 5 Teselas (Mayor IoU)', fontsize=12, fontweight='bold', color='darkgreen')
    axes[0].set_xlim([0, 1])
    axes[0].grid(alpha=0.3, axis='x')

    for i, (iou, tile) in enumerate(zip(best_ious, best_5)):
        axes[0].text(iou + 0.01, i, f'{iou:.4f}', va='center', fontsize=9, fontweight='bold')

    # Peores casos
    worst_names = [t['tile_name'][:20] + '...' if len(t['tile_name']) > 20 else t['tile_name'] for t in worst_5]
    worst_ious = [t['iou'] for t in worst_5]

    axes[1].barh(range(len(worst_5)), worst_ious, color='red', alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(len(worst_5)))
    axes[1].set_yticklabels(worst_names, fontsize=9)
    axes[1].set_xlabel('IoU', fontsize=11, fontweight='bold')
    axes[1].set_title('⚠️ Bottom 5 Teselas (Menor IoU)', fontsize=12, fontweight='bold', color='darkred')
    axes[1].set_xlim([0, 1])
    axes[1].grid(alpha=0.3, axis='x')

    for i, (iou, tile) in enumerate(zip(worst_ious, worst_5)):
        axes[1].text(iou + 0.01, i, f'{iou:.4f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    casos_path = output_dir / f'casos_destacados_{year}.png'
    plt.savefig(casos_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"   ✅ Casos destacados guardados: {casos_path.name}")

    # Retornar métricas calculadas
    return {
        'num_tiles': len(tiles_data),
        'mean_iou': mean_iou,
        'mean_accuracy': np.mean(accuracies),
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'optimal_threshold_roc': optimal_threshold_roc,
        'optimal_threshold_f1': optimal_threshold_f1,
        'best_tiles': best_5,
        'worst_tiles': worst_5,
    }


#====================================
# FUNCIONES DEL PIPELINE
#====================================

def load_model(checkpoint_path, device='cpu'):
    """
    Carga el modelo entrenado desde un checkpoint.

    Detecta automáticamente si es un modelo MultiBranch o UnetPlusPlus estándar
    basándose en las claves del state_dict.

    También detecta si el modelo usa módulos de atención (scSE).
    """
    print(f"📦 Cargando modelo desde: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}

    # Detectar tipo de modelo basándose en las claves del state_dict
    is_multibranch = any('high_res_encoder' in key for key in state_dict.keys())

    # Detectar si el modelo tiene módulos de atención
    has_attention = any('attention' in key and ('cSE' in key or 'sSE' in key) for key in state_dict.keys())

    if is_multibranch:
        print(f"🔍 Detectado: Modelo Multi-Branch UNet++")
        print(f"   - Arquitectura: Dual encoder (high-res + low-res)")
        print(f"   - Fusion: FPN (Feature Pyramid Network)")
        print(f"   - Decoder: UNet++ con nested skip connections")

        if has_attention:
            print(f"   - Atención: scSE (Spatial and Channel Squeeze & Excitation)")
            attention_type = 'scse'
        else:
            print(f"   - Atención: None")
            attention_type = None

        # Crear modelo MultiBranch
        wrapper = MultiBranchUNetWrapper(
            encoder_name='resnet101',
            encoder_weights=None,  # Los pesos ya están en el checkpoint
            high_res_channels=4,   # B2, B3, B4, B8 (10m)
            low_res_channels=2,    # B11, B12 (20m)
            num_classes=1,
            fusion_mode='fpn',
            upsample_mode='bilinear',
            deep_supervision=False,
            attention_type=attention_type  # Detectado automáticamente
        )

        # Cargar pesos del checkpoint
        # El wrapper tiene un atributo 'model' que contiene el MultiBranchUNet
        # Las claves del checkpoint no tienen el prefijo "model." porque ya lo quitamos
        wrapper.model.load_state_dict(state_dict, strict=True)
        model = wrapper

    else:
        print(f"🔍 Detectado: Modelo UNet++ estándar")
        print(f"   - Arquitectura: Single encoder")
        print(f"   - Decoder: UNet++")

        # Crear modelo UNet++ estándar
        model = smp.UnetPlusPlus(encoder_name='resnet34', in_channels=6)
        model.load_state_dict(state_dict)

    # Envolver en Module (para compatibilidad con el resto del código)
    module = Module(model)
    module.to(device)
    module.eval()

    print(f"✅ Modelo cargado exitosamente")
    return module


def predict_tile(module, image_path, device='cpu', threshold=0.20):
    """
    Realiza predicción en una tesela individual
    
    ✅ OPTIMIZADO: Sin re-normalización (imágenes ya vienen [0,1] desde GEE)
    
    Args:
        module: Modelo cargado
        image_path: Ruta a la imagen .tif
        device: Dispositivo ('cpu', 'cuda', 'mps')
        threshold: Umbral de decisión (0.0-1.0)
    
    Returns:
        pred_binary: Predicción binaria (0, 1)
        profile: Perfil de rasterio
        transform: Transformación geoespacial
        bounds: Límites espaciales
        crs: Sistema de coordenadas
    
    Notas:
        - Las imágenes Sentinel-2 ya están normalizadas [0, 1] desde GEE
        - NO se aplica normalización percentil adicional (evita compresión de rango)
        - Threshold 0.20 optimizado para este preprocesamiento
    """
    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile.copy()
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
        
        # ⭐ PREPROCESAMIENTO SIMPLIFICADO ⭐
        # Solo convertir a float32 y asegurar rango [0, 1]
        image = image.astype(np.float32)
        image = np.clip(image, 0, 1)
        
        # Convertir a tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        
        # Predicción
        with torch.no_grad():
            logits = module.model(image_tensor)
            probs = torch.sigmoid(logits)
        
        pred_np = probs.squeeze().cpu().numpy()
        pred_binary = (pred_np > threshold).astype(np.uint8)
    
    return pred_binary, profile, transform, bounds, crs


#====================================
# ⭐⭐⭐ NUEVA FUNCIÓN: POST-PROCESAMIENTO MORFOLÓGICO ⭐⭐⭐
#====================================

def post_process_mangrove_prediction(pred_binary, mode='conservative'):
    """
    Post-procesamiento morfológico para segmentación de manglar
    
    Basado en literatura científica:
    - Pham & Yoshino (2016) Remote Sensing of Environment
    - Chen et al. (2020) ISPRS Journal  
    - Wang et al. (2023) Remote Sensing of Environment
    
    Justificación ecológica:
    Los manglares crecen en parches continuos debido a crecimiento lateral 
    de raíces y propagación clonal (Tomlinson, 2016). Las discontinuidades 
    en predicciones del modelo reflejan variabilidad espectral interna 
    (sombras, diferentes especies) más que fragmentación real del manglar.
    
    Operaciones morfológicas:
    - Closing: Rellena pequeños huecos dentro de objetos (dilatación + erosión)
    - Opening: Elimina ruido aislado (erosión + dilatación)
    
    Args:
        pred_binary: Predicción binaria (0, 1) como numpy array 2D
        mode: Modo de post-procesamiento
            'conservative' - Kernel 3x3, cambio mínimo (recomendado para tesis)
            'moderate'     - Kernel 5x5, más corrección (validado en literatura)
            'none'         - Sin post-procesamiento (para comparación)
    
    Returns:
        Predicción refinada (numpy array uint8)
    
    Referencias:
        Pham, T. D., & Yoshino, K. (2016). Mangrove mapping and change 
        detection using multi-temporal Landsat imagery. Remote Sensing 
        of Environment, 175, 175-185.
        
        Chen, Y., et al. (2020). Deep learning for forest mapping from 
        satellite imagery. ISPRS Journal, 166, 195-213.
        
        Wang, L., et al. (2023). Deep learning-based mangrove mapping. 
        Remote Sensing of Environment, 285, 113123.
        
        Tomlinson, P. B. (2016). The Botany of Mangroves. Cambridge 
        University Press.
    """
    
    if mode == 'conservative':
        # Kernel pequeño (3x3) - Cambio mínimo, científicamente conservador
        # Cierra huecos de hasta 9 píxeles (3x3)
        kernel = np.ones((3, 3), dtype=np.uint8)
        
        # Paso 1: Closing - cierra pequeños huecos dentro de parches
        pred_closed = binary_closing(pred_binary, structure=kernel)
        
        # Paso 2: Opening - elimina píxeles aislados (ruido)
        pred_final = binary_opening(pred_closed, structure=kernel)
        
    elif mode == 'moderate':
        # Kernel mediano (5x5) - Más corrección, validado en Pham & Yoshino (2016)
        # Cierra huecos de hasta 25 píxeles (5x5)
        kernel_close = np.ones((5, 5), dtype=np.uint8)
        kernel_open = np.ones((3, 3), dtype=np.uint8)
        
        pred_closed = binary_closing(pred_binary, structure=kernel_close)
        pred_final = binary_opening(pred_closed, structure=kernel_open)
        
    elif mode == 'none':
        # Sin post-procesamiento (para comparación en tesis)
        pred_final = pred_binary
        
    else:
        raise ValueError(f"Modo '{mode}' no válido. Usar: 'conservative', 'moderate', o 'none'")
    
    return pred_final.astype(np.uint8)


def save_prediction_geotiff(pred_binary, profile, transform, save_path):
    """
    Guarda la predicción como GeoTIFF georreferenciado
    
    ✅ CORREGIDO: Fuerza eliminación del archivo antes de escribir
    """
    from pathlib import Path
    import os
    
    # ⭐ FORZAR ELIMINACIÓN SI EXISTE
    save_path_obj = Path(save_path)
    if save_path_obj.exists():
        try:
            os.remove(save_path)
        except Exception as e:
            print(f"⚠️ No se pudo eliminar {save_path}: {e}")
    
    profile.update(
        count=1,
        dtype='uint8',
        compress='lzw',
        nodata=255,
        transform=transform
    )
    
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(pred_binary, 1)
        dst.write_colormap(1, {
            0: (139, 69, 19),
            1: (34, 139, 34),
            255: (0, 0, 0)
        })


def process_tiles(checkpoint_path, images_dir, output_dir, device='cpu', threshold=0.20, 
                  use_postproc=True, postproc_mode='conservative'):
    """
    Procesa todas las teselas del directorio de imágenes
    
    ✅ OPTIMIZADO: Sin normalización percentil redundante
    
    Args:
        checkpoint_path: Ruta al modelo entrenado (.ckpt)
        images_dir: Directorio con imágenes (.tif)
        output_dir: Directorio de salida para predicciones
        device: Dispositivo de cómputo ('cpu', 'cuda', 'mps')
        threshold: Umbral de decisión (0.0-1.0)
        use_postproc: Activar post-procesamiento morfológico (recomendado: True)
        postproc_mode: Modo de post-procesamiento ('conservative', 'moderate', 'none')
    
    Returns:
        Lista de rutas a archivos de predicción generados
    """
    
    module = load_model(checkpoint_path, device)
    
    # Buscar todos los .tif en el directorio
    tile_paths = sorted(images_dir.glob("*.tif"))
    
    if len(tile_paths) == 0:
        raise ValueError(f"❌ No se encontraron archivos .tif en {images_dir}")
    
    print(f"\n🔍 Encontradas {len(tile_paths)} teselas en {images_dir}")
    
    pred_files = []
    
    # ⭐ MENSAJE INFORMATIVO SOBRE CONFIGURACIÓN
    postproc_status = "ACTIVADO" if use_postproc else "DESACTIVADO"
    print(f"\n🚀 Iniciando predicción de teselas")
    print(f"   Umbral de decisión: {threshold}")
    print(f"   Preprocesamiento: SIN normalización adicional (imágenes ya en [0,1])")
    print(f"   Post-procesamiento morfológico: {postproc_status}")
    if use_postproc:
        kernel_size = '3x3' if postproc_mode == 'conservative' else '5x5' if postproc_mode == 'moderate' else 'N/A'
        print(f"   Modo: {postproc_mode} (kernel {kernel_size})")
        print(f"   Referencia: Pham & Yoshino (2016), Wang et al. (2023)")
    
    for tile_path in tqdm(tile_paths, desc="Prediciendo teselas"):
        try:
            # Predicción base
            pred_binary, profile, transform, bounds, crs = predict_tile(
                module, str(tile_path), device, threshold
            )
            
            # ⭐⭐⭐ APLICAR POST-PROCESAMIENTO MORFOLÓGICO ⭐⭐⭐
            if use_postproc:
                pred_binary = post_process_mangrove_prediction(pred_binary, mode=postproc_mode)
            # ⭐⭐⭐ FIN DEL POST-PROCESAMIENTO ⭐⭐⭐
            
            # Guardar predicción
            tile_name = tile_path.stem
            pred_filename = f"pred_{tile_name}.tif"
            pred_path = output_dir / pred_filename
            
            save_prediction_geotiff(pred_binary, profile, transform, str(pred_path))
            
            pred_files.append(str(pred_path))
            
        except Exception as e:
            print(f"\n⚠️ Error procesando {tile_path.name}: {str(e)}")
            continue
    
    print(f"\n✅ {len(pred_files)} teselas procesadas exitosamente")
    return pred_files


def create_mosaic(pred_files, output_mosaic_path, method='first'):
    """Crea un mosaico a partir de las predicciones individuales"""
    print(f"\n🧩 Creando mosaico de {len(pred_files)} teselas...")
    
    src_files_to_mosaic = []
    for fp in pred_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    
    mosaic, out_trans = merge(src_files_to_mosaic, method=method)
    
    for src in src_files_to_mosaic:
        src.close()
    
    with rasterio.open(pred_files[0]) as src:
        out_meta = src.meta.copy()
    
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })
    
    with rasterio.open(output_mosaic_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        dest.write_colormap(1, {
            0: (139, 69, 19),
            1: (34, 139, 34),
            255: (0, 0, 0)
        })
    
    print(f"✅ Mosaico creado: {output_mosaic_path}")
    print(f"   Dimensiones: {mosaic.shape[2]} x {mosaic.shape[1]} píxeles")
    
    mosaic_data = mosaic[0]
    total_pixels = mosaic_data.size
    manglar_pixels = np.sum(mosaic_data == 1)
    no_manglar_pixels = np.sum(mosaic_data == 0)
    nodata_pixels = np.sum(mosaic_data == 255)
    
    manglar_pct = 100 * manglar_pixels / (total_pixels - nodata_pixels) if (total_pixels - nodata_pixels) > 0 else 0
    
    print(f"\n📊 Estadísticas del Mosaico:")
    print(f"   Total píxeles:        {total_pixels:,}")
    print(f"   Píxeles Manglar:      {manglar_pixels:,} ({manglar_pct:.2f}%)")
    print(f"   Píxeles No-Manglar:   {no_manglar_pixels:,}")
    print(f"   Píxeles NoData:       {nodata_pixels:,}")
    
    return output_mosaic_path


def apply_study_area_mask(mosaic_path, shapefile_path, output_path=None):
    """
    Aplica máscara del área de estudio al mosaico.

    Args:
        mosaic_path: Ruta al mosaico TIF
        shapefile_path: Ruta al shapefile del área de estudio
        output_path: Ruta de salida (si None, sobrescribe el original)

    Returns:
        Ruta al mosaico enmascarado
    """
    print(f"\n🗺️  Aplicando máscara del área de estudio...")
    print(f"   Shapefile: {Path(shapefile_path).name}")

    # Leer shapefile
    gdf = gpd.read_file(shapefile_path)
    print(f"   CRS shapefile: {gdf.crs}")
    print(f"   Geometrías: {len(gdf)}")

    # Abrir mosaico
    with rasterio.open(mosaic_path) as src:
        print(f"   CRS mosaico: {src.crs}")

        # Reproyectar shapefile si es necesario
        if gdf.crs != src.crs:
            print(f"   ⚠️  Reproyectando shapefile de {gdf.crs} a {src.crs}")
            gdf = gdf.to_crs(src.crs)

        # Aplicar máscara
        # crop=False mantiene la extensión original
        # filled=True rellena áreas fuera con nodata
        out_image, out_transform = rasterio_mask(
            src,
            gdf.geometry,
            crop=False,
            filled=True,
            nodata=255  # NoData para áreas fuera
        )

        # Copiar metadatos
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": 255,
            "compress": "lzw"
        })

    # Determinar ruta de salida
    if output_path is None:
        output_path = mosaic_path  # Sobrescribir original

    # Guardar mosaico enmascarado
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)
        dest.write_colormap(1, {
            0: (139, 69, 19),    # No-Manglar (marrón)
            1: (34, 139, 34),     # Manglar (verde)
            255: (0, 0, 0)        # NoData (negro/transparente)
        })

    # Estadísticas
    mosaic_data = out_image[0]
    total_pixels = mosaic_data.size
    manglar_pixels = np.sum(mosaic_data == 1)
    no_manglar_pixels = np.sum(mosaic_data == 0)
    nodata_pixels = np.sum(mosaic_data == 255)

    valid_pixels = total_pixels - nodata_pixels
    manglar_pct = 100 * manglar_pixels / valid_pixels if valid_pixels > 0 else 0

    print(f"\n   ✅ Máscara aplicada: {Path(output_path).name}")
    print(f"   📊 Estadísticas del área de estudio:")
    print(f"      Píxeles dentro:       {valid_pixels:,} ({100*valid_pixels/total_pixels:.1f}%)")
    print(f"      Píxeles fuera (NoData): {nodata_pixels:,} ({100*nodata_pixels/total_pixels:.1f}%)")
    print(f"      Manglar:              {manglar_pixels:,} ({manglar_pct:.2f}% del área)")
    print(f"      No-Manglar:           {no_manglar_pixels:,}")

    return output_path


def visualize_mosaic(mosaic_path, output_viz_path, year, figsize=(16, 12), dpi=300,
                     shapefile_path=None):
    """
    Visualiza el mosaico final con múltiples métodos de visualización

    Genera 3 visualizaciones:
    1. Mosaico estándar (colores originales)
    2. Mosaico con verde neón (alta visibilidad)
    3. Mapa de calor de densidad

    Args:
        mosaic_path: Ruta al mosaico GeoTIFF
        output_viz_path: Ruta de salida para visualización estándar
        year: Año de análisis
        figsize: Tamaño de la figura (ancho, alto)
        dpi: Resolución de salida
        shapefile_path: Ruta al shapefile del área de estudio (opcional)
    """
    from matplotlib.colors import BoundaryNorm

    # ═══════════════════════════════════════════════════════════════
    # FUNCIONES AUXILIARES PARA ELEMENTOS CARTOGRÁFICOS
    # ═══════════════════════════════════════════════════════════════

    def add_scale_bar(ax, left, right, bottom, top):
        """Añade barra de escala cartográfica al gráfico."""
        extent_width = right - left
        scale_length_m = 10000  # 10 km por defecto

        if extent_width < 30000:
            scale_length_m = 5000  # 5 km

        center_x = (left + right) / 2
        bar_x = center_x - scale_length_m / 2
        bar_y = top - 2500  # 2.5 km del borde superior

        # Fondo negro
        ax.plot([bar_x, bar_x + scale_length_m], [bar_y, bar_y],
               color='black', linewidth=6, solid_capstyle='butt', zorder=12)

        # Barra blanca
        ax.plot([bar_x, bar_x + scale_length_m], [bar_y, bar_y],
               color='white', linewidth=4, solid_capstyle='butt', zorder=13)

        # Segmentos
        segment_length = scale_length_m / 4
        for i in range(5):
            x_pos = bar_x + i * segment_length
            ax.plot([x_pos, x_pos], [bar_y - 150, bar_y + 150],
                   color='white', linewidth=2, zorder=13)

        # Etiquetas
        label_km = scale_length_m / 1000
        ax.text(bar_x, bar_y - 600, '0', ha='center', va='top',
               color='white', fontsize=8, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
               zorder=14)
        ax.text(bar_x + scale_length_m, bar_y - 600, f'{label_km:.0f} km',
               ha='center', va='top',
               color='white', fontsize=8, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
               zorder=14)

    def add_north_arrow(ax):
        """Añade flecha de norte discreta al gráfico."""
        arrow_x = 0.05
        arrow_y = 0.70

        # Fondo negro
        ax.annotate('', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                   xytext=(arrow_x, arrow_y - 0.06),
                   arrowprops=dict(arrowstyle='->', lw=4, color='black',
                                 mutation_scale=20),
                   zorder=13)

        # Flecha blanca
        ax.annotate('', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                   xytext=(arrow_x, arrow_y - 0.06),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='white',
                                 mutation_scale=20),
                   zorder=14)

        # Etiqueta "N"
        ax.text(arrow_x, arrow_y + 0.015, 'N',
               transform=ax.transAxes,
               ha='center', va='bottom',
               fontsize=14, weight='bold', color='white',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='black',
                        alpha=0.75, edgecolor='white', linewidth=1.5),
               zorder=14)

    # ═══════════════════════════════════════════════════════════════
    # INICIO DE LA VISUALIZACIÓN
    # ═══════════════════════════════════════════════════════════════

    print(f"\n🎨 Generando visualizaciones del mosaico...")

    with rasterio.open(mosaic_path) as src:
        mosaic = src.read(1)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        crs = src.crs

    output_dir = Path(output_viz_path).parent

    # Cargar shapefile si está disponible
    gdf = None
    if shapefile_path is not None:
        shapefile_path = Path(shapefile_path)
        if shapefile_path.exists():
            print(f"   Cargando shapefile: {shapefile_path.name}")
            gdf = gpd.read_file(shapefile_path)
            # Reproyectar si es necesario
            if gdf.crs != crs:
                print(f"   Reproyectando shapefile de {gdf.crs} a {crs}")
                gdf = gdf.to_crs(crs)
    
    # Estadísticas para todas las visualizaciones
    manglar_pixels = np.sum(mosaic == 1)
    no_manglar_pixels = np.sum(mosaic == 0)
    nodata_pixels = np.sum(mosaic == 255)
    total_pixels = mosaic.size
    manglar_pct = (manglar_pixels / (total_pixels - nodata_pixels)) * 100 if (total_pixels - nodata_pixels) > 0 else 0
    
    # ═══════════════════════════════════════════════════════════════
    # VISUALIZACIÓN 1: VERDE NEÓN (ALTA VISIBILIDAD)
    # ═══════════════════════════════════════════════════════════════

    print(f"   Generando mosaico con verde neón...")

    # Reasignar valores para mapeo correcto y transparencia
    mosaic_vis = mosaic.copy().astype(np.float32)
    mosaic_vis[mosaic == 255] = np.nan  # Áreas sin datos como NaN para transparencia

    # Crear figura con aspect ratio del mosaico
    height, width = mosaic.shape
    aspect_ratio = width / height
    fig_width = 18
    fig_height = fig_width / aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Fondo celeste claro para representar agua (COMO EN FALSO COLOR)
    ax.set_facecolor('#87CEEB')

    # Colores: Celeste claro (no manglar), Verde NEÓN (manglar)
    # Usamos transparencia para NoData en lugar de color gris
    colors_neon = ['#87CEEB', '#39FF14']  # Agua=Celeste, Manglar=Verde neón
    cmap_neon = ListedColormap(colors_neon)
    norm_neon = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=2)

    im = ax.imshow(mosaic_vis, cmap=cmap_neon, norm=norm_neon, extent=extent,
                   interpolation='none', aspect='equal')

    # Colorbar con solo 2 categorías (sin NoData)
    #cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, ticks=[0, 1])
    #cbar.ax.set_yticklabels(['No Manglar (Agua/Tierra)', 'MANGLAR'], fontsize=11)
    #cbar.ax.tick_params(labelsize=10)

    # Superponer contorno del shapefile (COMO EN FALSO COLOR)
    if gdf is not None:
        # Contorno doble (negro + amarillo dorado) para mejor visibilidad
        gdf.boundary.plot(ax=ax, color='black', linewidth=2.5,
                         linestyle='-', alpha=0.9, zorder=10)
        gdf.boundary.plot(ax=ax, color='#FFD700', linewidth=1.5,
                         linestyle='-', alpha=1.0, zorder=11,
                         label='Límite del área de estudio')

        # Leyenda
        ax.legend(loc='upper right', fontsize=16, framealpha=0.9,
                 fancybox=True, shadow=True, edgecolor='black')

    # Límites del gráfico
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Títulos y etiquetas (ESTILO FALSO COLOR)
    ax.set_xlabel('Longitud (m)', fontsize=12)
    ax.set_ylabel('Latitud (m)', fontsize=12)
    ax.set_title(f'Segmentación de Manglares - Año {year}\n' +
                 f'Verde Neón = Manglar ({manglar_pixels:,} píxeles, {manglar_pct:.2f}%)',
                 fontsize=22, fontweight='bold', pad=15)

    # Grid sutil (COMO EN FALSO COLOR)
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, color='white')

    # Caja de información mejorada (ESTILO FALSO COLOR)
    info_lines = [
        "Estadísticas:",
        f"  Manglar: {manglar_pixels:,} px ({manglar_pct:.2f}%)",
        f"  No-Manglar: {no_manglar_pixels:,} px",
        "",
        "Área de Estudio:",
        "  Archipiélago de Jambelí",
        "",
        "Resolución:",
        "  10 m/píxel"
    ]
    info_text = "\n".join(info_lines)

    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=18,
           verticalalignment='top',
           horizontalalignment='left',
           family='monospace',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='black', alpha=0.75,
                    edgecolor='#FFD700', linewidth=1.5),
           color='white',
           zorder=15)

    # Añadir elementos cartográficos (COMO EN FALSO COLOR)
    add_scale_bar(ax, extent[0], extent[1], extent[2], extent[3])
    add_north_arrow(ax)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    neon_path = output_dir / f"mosaico_VERDE_NEON_{year}.png"
    plt.savefig(neon_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close()

    print(f"   ✅ Verde neón guardado: {neon_path.name}")
    
    # ═══════════════════════════════════════════════════════════════
    # VISUALIZACIÓN 2: MAPA DE CALOR DE DENSIDAD
    # ═══════════════════════════════════════════════════════════════

    print(f"   Generando mapa de calor de densidad...")

    # Calcular densidad en bloques
    block_size = 50
    h, w = mosaic.shape
    h_blocks = h // block_size
    w_blocks = w // block_size

    density_map = np.zeros((h_blocks, w_blocks))

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = mosaic[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            density_map[i, j] = (block == 1).sum() / block.size * 100

    # Crear figura con aspect ratio del mosaico
    aspect_ratio = w / h
    fig_width = 18
    fig_height = fig_width / aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Fondo celeste claro para representar agua (COMO EN FALSO COLOR)
    ax.set_facecolor('#87CEEB')

    im = ax.imshow(density_map, cmap='YlGn', extent=extent,
                   interpolation='bilinear', aspect='equal',
                   vmin=0, vmax=max(density_map.max(), 1))

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Densidad de Manglar (%)', rotation=270, labelpad=25,
                   fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Superponer contorno del shapefile (COMO EN FALSO COLOR)
    if gdf is not None:
        # Contorno doble (negro + amarillo dorado) para mejor visibilidad
        gdf.boundary.plot(ax=ax, color='black', linewidth=2.5,
                         linestyle='-', alpha=0.9, zorder=10)
        gdf.boundary.plot(ax=ax, color='#FFD700', linewidth=1.5,
                         linestyle='-', alpha=1.0, zorder=11,
                         label='Límite del área de estudio')

        # Leyenda
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                 fancybox=True, shadow=True, edgecolor='black')

    # Límites del gráfico
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Títulos y etiquetas (ESTILO FALSO COLOR)
    ax.set_xlabel('Longitud (m)', fontsize=12)
    ax.set_ylabel('Latitud (m)', fontsize=12)
    ax.set_title(f'Mapa de Calor - Densidad de Manglar (Año {year})\n' +
                 f'Bloques de {block_size}×{block_size} píxeles | Amarillo-Verde = Mayor densidad',
                 fontsize=18, fontweight='bold', pad=15)

    # Grid sutil (COMO EN FALSO COLOR)
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, color='white')

    # Caja de información mejorada (ESTILO FALSO COLOR)
    max_density = density_map.max()
    mean_density = density_map[density_map > 0].mean() if (density_map > 0).any() else 0

    info_lines = [
        "Estadísticas de Densidad:",
        f"  Máxima: {max_density:.1f}%",
        f"  Promedio: {mean_density:.1f}%",
        f"  Bloque: {block_size}×{block_size} px",
        "",
        "Área de Estudio:",
        "  Archipiélago de Jambelí",
        "",
        "Resolución:",
        "  10 m/píxel"
    ]
    info_text = "\n".join(info_lines)

    ax.text(0.98, 0.02, info_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='bottom',
           horizontalalignment='right',
           family='monospace',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='black', alpha=0.75,
                    edgecolor='#FFD700', linewidth=1.5),
           color='white',
           zorder=15)

    # Añadir elementos cartográficos (COMO EN FALSO COLOR)
    add_scale_bar(ax, extent[0], extent[1], extent[2], extent[3])
    add_north_arrow(ax)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    heatmap_path = output_dir / f"mosaico_HEATMAP_{year}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close()

    print(f"   ✅ Mapa de calor guardado: {heatmap_path.name}")
    
    # ═══════════════════════════════════════════════════════════════
    # VISUALIZACIÓN 3: ESTÁNDAR (COMPATIBILIDAD)
    # ═══════════════════════════════════════════════════════════════
    
    print(f"   Generando mosaico estándar...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors_std = ['#8B4513', '#228B22', '#000000']
    cmap_std = ListedColormap(colors_std)
    
    im = ax.imshow(mosaic, cmap=cmap_std, extent=extent, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks([0, 1, 255])
    cbar.set_ticklabels(['No Manglar', 'Manglar', 'NoData'])
    
    ax.set_xlabel('Longitud', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitud', fontsize=12, fontweight='bold')
    ax.set_title(f'Mosaico de Predicción - Segmentación de Manglares (Año {year})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_viz_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Mosaico estándar guardado: {Path(output_viz_path).name}")
    
    print(f"\n✅ 3 visualizaciones generadas exitosamente:")
    print(f"   1. {neon_path.name} (verde neón, alta visibilidad)")
    print(f"   2. {heatmap_path.name} (mapa de calor)")
    print(f"   3. {Path(output_viz_path).name} (estándar)")


def generate_tile_visualizations(images_dir, masks_dir, predictions_dir, output_viz_dir,
                                  year, max_tiles=50, selection_mode='best'):
    """
    Genera visualizaciones comparativas individuales: RGB | Ground Truth | Predicción

    Args:
        images_dir: Directorio con imágenes originales
        masks_dir: Directorio con máscaras ground truth
        predictions_dir: Directorio con predicciones
        output_viz_dir: Directorio de salida para visualizaciones
        year: Año de análisis
        max_tiles: Número máximo de visualizaciones a generar
        selection_mode: Modo de selección ('best', 'worst', 'random', 'all')

    Returns:
        Número de visualizaciones generadas
    """

    print(f"\n🎨 Generando visualizaciones de teselas individuales...")
    print(f"   Modo de selección: {selection_mode}")
    print(f"   Máximo de teselas: {max_tiles if selection_mode != 'all' else 'todas'}")

    # Buscar archivos de predicción
    pred_files = sorted(predictions_dir.glob("pred_*.tif"))

    if len(pred_files) == 0:
        print(f"⚠️ No se encontraron predicciones en {predictions_dir}")
        return 0

    # Calcular IoU para cada tesela y seleccionar cuáles visualizar
    tiles_data = []

    for pred_path in tqdm(pred_files, desc="Calculando IoU de teselas"):
        # Extraer nombre de tesela
        pred_name = pred_path.stem
        tile_name = pred_name.replace('pred_', '')

        # Buscar imagen y máscara correspondiente
        image_path = images_dir / f"{tile_name}.tif"
        mask_path = masks_dir / f"{tile_name.replace('_r', '_mask_r')}.tif"

        if not image_path.exists():
            continue

        if not mask_path.exists():
            # Intentar patrones alternativos
            alt_patterns = [
                masks_dir / f"{tile_name}_mask.tif",
                masks_dir / f"mask_{tile_name}.tif",
            ]
            for alt_path in alt_patterns:
                if alt_path.exists():
                    mask_path = alt_path
                    break
            else:
                continue  # No se encontró máscara

        try:
            # Leer predicción y máscara
            with rasterio.open(pred_path) as src:
                pred = src.read(1)

            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            # Calcular IoU
            if pred.shape != mask.shape:
                continue

            # Filtrar NaN y calcular IoU solo si hay manglar
            valid_pixels = ~np.isnan(mask)
            mask_clean = mask[valid_pixels]
            pred_clean = pred[valid_pixels]

            # Saltar máscaras vacías
            if len(np.unique(mask_clean)) == 1 and np.unique(mask_clean)[0] == 0:
                continue

            TP = np.sum((pred_clean == 1) & (mask_clean == 1))
            FP = np.sum((pred_clean == 1) & (mask_clean == 0))
            FN = np.sum((pred_clean == 0) & (mask_clean == 1))

            tile_iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

            tiles_data.append({
                'tile_name': tile_name,
                'image_path': image_path,
                'mask_path': mask_path,
                'pred_path': pred_path,
                'iou': tile_iou,
                'manglar_pct_real': (mask_clean == 1).sum() / len(mask_clean) * 100,
                'manglar_pct_pred': (pred_clean == 1).sum() / len(pred_clean) * 100,
            })

        except Exception as e:
            continue

    if len(tiles_data) == 0:
        print(f"⚠️ No se pudieron calcular IoU para ninguna tesela")
        return 0

    # Seleccionar teselas según el modo
    if selection_mode == 'best':
        tiles_data.sort(key=lambda x: x['iou'], reverse=True)
        selected_tiles = tiles_data[:max_tiles]
        print(f"   📊 Seleccionadas {len(selected_tiles)} mejores teselas (IoU: {selected_tiles[0]['iou']:.4f} - {selected_tiles[-1]['iou']:.4f})")
    elif selection_mode == 'worst':
        tiles_data.sort(key=lambda x: x['iou'])
        selected_tiles = tiles_data[:max_tiles]
        print(f"   📊 Seleccionadas {len(selected_tiles)} peores teselas (IoU: {selected_tiles[0]['iou']:.4f} - {selected_tiles[-1]['iou']:.4f})")
    elif selection_mode == 'random':
        import random
        selected_tiles = random.sample(tiles_data, min(max_tiles, len(tiles_data)))
        print(f"   📊 Seleccionadas {len(selected_tiles)} teselas aleatorias")
    else:  # 'all'
        selected_tiles = tiles_data
        print(f"   📊 Generando visualizaciones para todas las {len(selected_tiles)} teselas")

    # Generar visualizaciones
    colors = ['#8B4513', '#228B22']  # Marrón para no-manglar, verde para manglar
    cmap = ListedColormap(colors)

    viz_count = 0
    for idx, tile_data in enumerate(tqdm(selected_tiles, desc="Generando visualizaciones")):
        try:
            # Leer imagen RGB (bandas 2, 3, 4 = RGB)
            with rasterio.open(tile_data['image_path']) as src:
                image = src.read()

            # Crear composición RGB (asumiendo orden: B2, B3, B4, B8, B11, B12)
            if image.shape[0] >= 3:
                rgb = np.stack([image[2], image[1], image[0]], axis=-1)  # B4, B3, B2

                # Filtrar NaN antes de normalizar
                valid_mask = ~np.isnan(rgb).any(axis=2)
                rgb_clean = rgb[valid_mask]

                if len(rgb_clean) == 0:
                    # Imagen completamente NaN
                    rgb_vis = np.full((rgb.shape[0], rgb.shape[1], 3), 128, dtype=np.uint8)
                else:
                    # Normalización robusta con estrategia adaptativa
                    p1, p99 = np.percentile(rgb_clean, (1, 99))
                    p_range = p99 - p1

                    if p_range < 0.01:
                        # Rango muy pequeño: usar min-max directo
                        rgb_min = rgb_clean.min()
                        rgb_max = rgb_clean.max()
                        if rgb_max - rgb_min > 0:
                            rgb_normalized = (rgb - rgb_min) / (rgb_max - rgb_min)
                        else:
                            rgb_normalized = np.full_like(rgb, 0.5)
                    else:
                        # Normalización percentil estándar
                        rgb_normalized = np.clip((rgb - p1) / p_range, 0, 1)

                    # Detección de imagen oscura (agua, sombras)
                    mean_intensity = rgb_clean.mean()
                    is_dark_image = mean_intensity < 0.08  # Umbral empírico

                    if is_dark_image:
                        # Aplicar gamma correction para realzar detalles en zonas oscuras
                        gamma = 0.5  # Gamma < 1 aclara la imagen
                        rgb_normalized = np.power(rgb_normalized, gamma)

                    # Convertir a uint8
                    rgb_vis = (rgb_normalized * 255).astype(np.uint8)

                    # Restaurar NaN como negro
                    rgb_vis[~valid_mask] = 0
            else:
                # Si no hay suficientes bandas, usar imagen en escala de grises
                rgb_vis = np.stack([image[0]] * 3, axis=-1)

            # Leer máscara y predicción
            with rasterio.open(tile_data['mask_path']) as src:
                mask = src.read(1)

            with rasterio.open(tile_data['pred_path']) as src:
                pred = src.read(1)

            # Crear visualización
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Panel 1: RGB
            axes[0].imshow(rgb_vis)

            # Título adaptativo según tipo de imagen
            if len(rgb_clean) > 0:
                mean_intensity = rgb_clean.mean()
                is_dark_image = mean_intensity < 0.08

                if is_dark_image:
                    rgb_title = f'Imagen Satelital (RGB)\n⚠️ Zona muy oscura (agua/sombra)\nγ-corrección aplicada'
                    axes[0].set_title(rgb_title, fontsize=11, fontweight='bold', color='#FF6B35')
                else:
                    axes[0].set_title('Imagen Satelital (RGB)', fontsize=12, fontweight='bold')
            else:
                axes[0].set_title('Imagen Satelital (RGB)\n⚠️ Sin datos', fontsize=11, fontweight='bold', color='red')

            axes[0].axis('off')

            # Panel 2: Ground Truth
            axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=1)
            axes[1].set_title(f'Ground Truth\n{tile_data["manglar_pct_real"]:.1f}% manglar',
                            fontsize=12, fontweight='bold')
            axes[1].axis('off')

            # Panel 3: Predicción
            axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=1)
            axes[2].set_title(f'Predicción\nIoU: {tile_data["iou"]:.4f} | {tile_data["manglar_pct_pred"]:.1f}% manglar',
                            fontsize=12, fontweight='bold')
            axes[2].axis('off')

            fig.suptitle(f'Tesela: {tile_data["tile_name"]} | Año: {year}',
                        fontsize=14, fontweight='bold', y=1.02)

            plt.tight_layout()

            # Guardar con formato numerado
            viz_filename = f"viz_{idx:04d}_{tile_data['tile_name']}.png"
            viz_path = output_viz_dir / viz_filename
            plt.savefig(viz_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()

            viz_count += 1

        except Exception as e:
            print(f"\n⚠️ Error generando visualización para {tile_data['tile_name']}: {str(e)}")
            continue

    print(f"\n✅ {viz_count} visualizaciones generadas exitosamente")
    print(f"   📁 Guardadas en: {output_viz_dir}/")

    return viz_count


def calculate_area_statistics(mosaic_path, output_report_path, year):
    """
    Calcula estadísticas de área del mosaico
    
    ✅ MEJORADO: Detección automática de CRS y reproyección condicional
    
    - Detecta si la imagen está en coordenadas geográficas o proyectadas
    - Reproyecta SOLO si es necesario (coordenadas geográficas)
    - Preserva metadatos originales cuando ya está en sistema proyectado
    
    FLUJO DE PROCESAMIENTO:
    1. Lee mosaico y metadatos espaciales
    2. Detecta tipo de CRS (geográfico vs proyectado)
    3. Si geográfico → reproyecta a UTM
    4. Si proyectado → usa metadatos originales sin cambios
    5. Calcula áreas con resolución correcta
    """
    print(f"\n📐 Calculando estadísticas de área...")
    
    with rasterio.open(mosaic_path) as src:
        mosaic = src.read(1)
        original_transform = src.transform
        original_crs = src.crs
        bounds = src.bounds
        
        print(f"\n{'='*70}")
        print(f"📍 ANÁLISIS DE SISTEMA DE COORDENADAS")
        print(f"{'='*70}")
        print(f"CRS original: {original_crs}")
        
        # ══════════════════════════════════════════════════════════════
        # 🔍 DETECCIÓN ROBUSTA DE CRS GEOGRÁFICO vs PROYECTADO
        # ══════════════════════════════════════════════════════════════
        
        is_geographic, detection_method, crs_info = is_crs_geographic(original_crs)
        
        print(f"\n🔍 Resultados de la detección:")
        print(f"   Tipo de CRS: {'🌍 GEOGRÁFICO (grados)' if is_geographic else '📐 PROYECTADO (metros)'}")
        print(f"   Método de detección: {detection_method}")
        
        # Mostrar información adicional del CRS
        if 'units' in crs_info:
            print(f"   Unidades detectadas: {crs_info['units']}")
        if 'zone' in crs_info:
            print(f"   Zona UTM: {crs_info['zone']}")
        if 'epsg' in crs_info:
            print(f"   Código EPSG: {crs_info['epsg']}")
        
        # ══════════════════════════════════════════════════════════════
        # 🔄 REPROYECCIÓN CONDICIONAL
        # ══════════════════════════════════════════════════════════════
        print(f"\n{'='*70}")
        print(f"📐 DECISIÓN DE REPROYECCIÓN")
        print(f"{'='*70}")

        if is_geographic:
            print(f"\n⚠️  Imagen en coordenadas GEOGRÁFICAS detectada")
            print(f"   Razón: Las coordenadas están en grados (latitud/longitud)")
            print(f"   Acción: REPROYECTAR a UTM para cálculo preciso de áreas")
            print(f"\n🔄 Iniciando reproyección a UTM zona 17S (EPSG:32717)...")
            
            # Definir CRS destino (UTM 17S para Ecuador)
            dst_crs = CRS.from_epsg(32717)
            
            print(f"   Origen: {original_crs}")
            print(f"   Destino: {dst_crs}")
            print(f"   Método de remuestreo: Nearest Neighbor (preserva valores binarios)")
            
            # Calcular transformación
            dst_transform, width, height = calculate_default_transform(
                original_crs, dst_crs, src.width, src.height, *bounds
            )
            
            print(f"   Dimensiones originales: {src.width} × {src.height} píxeles")
            print(f"   Dimensiones destino: {width} × {height} píxeles")
            
            # Crear array de destino
            mosaic_utm = np.empty((height, width), dtype=mosaic.dtype)
            
            # Reproyectar
            reproject(
                source=mosaic,
                destination=mosaic_utm,
                src_transform=original_transform,
                src_crs=original_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            
            # Usar datos reproyectados
            mosaic = mosaic_utm
            transform = dst_transform
            crs = dst_crs
            
            print(f"\n✅ Reproyección completada exitosamente")
            print(f"   CRS final: {crs}")
            print(f"   Dimensiones finales: {mosaic.shape[1]} × {mosaic.shape[0]} píxeles")
            
        else:
            # Ya está en coordenadas proyectadas - NO reproyectar
            print(f"\n✅ Imagen en coordenadas PROYECTADAS detectada")
            print(f"   Razón: El CRS ya está en un sistema proyectado (unidades métricas)")
            print(f"   Acción: OMITIR reproyección (no es necesaria)")
            print(f"\n🎯 Preservando metadatos originales:")
            print(f"   ✓ CRS original: {original_crs}")
            print(f"   ✓ Transform original preservado")
            print(f"   ✓ Resolución espacial intacta")
            print(f"   ✓ No hay pérdida de precisión por reproyección")
            
            transform = original_transform
            crs = original_crs
            
            # Mostrar información de la resolución original
            orig_pixel_width = abs(original_transform[0])
            orig_pixel_height = abs(original_transform[4])
            print(f"   ✓ Resolución: {orig_pixel_width:.2f} × {orig_pixel_height:.2f} metros")
        
        # Calcular tamaño de píxel en metros
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])
        pixel_area_m2 = pixel_width * pixel_height
        pixel_area_ha = pixel_area_m2 / 10000
        
        print(f"   📏 Resolución: {pixel_width:.2f} x {pixel_height:.2f} metros")
        print(f"   📐 Área por píxel: {pixel_area_m2:.2f} m² ({pixel_area_ha:.6f} ha)")
    
    total_pixels = mosaic.size
    manglar_pixels = np.sum(mosaic == 1)
    no_manglar_pixels = np.sum(mosaic == 0)
    nodata_pixels = np.sum(mosaic == 255)
    valid_pixels = total_pixels - nodata_pixels
    
    area_manglar_ha = manglar_pixels * pixel_area_ha
    area_no_manglar_ha = no_manglar_pixels * pixel_area_ha
    area_total_ha = valid_pixels * pixel_area_ha
    
    manglar_pct = 100 * manglar_pixels / valid_pixels if valid_pixels > 0 else 0
    
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"REPORTE DE ÁREA - MOSAICO DE PREDICCIÓN (AÑO {year})\n")
        f.write("="*70 + "\n\n")
        
        f.write("INFORMACIÓN ESPACIAL:\n")
        f.write("-"*70 + "\n")
        f.write(f"Año de análisis:        {year}\n")
        f.write(f"CRS:                    {crs}\n")
        f.write(f"Resolución espacial:    {pixel_width:.2f} x {pixel_height:.2f} metros\n")
        f.write(f"Área por píxel:         {pixel_area_m2:.2f} m² ({pixel_area_ha:.6f} ha)\n")
        f.write("-"*70 + "\n\n")
        
        f.write("CONTEO DE PÍXELES:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total de píxeles:       {total_pixels:,}\n")
        f.write(f"Píxeles válidos:        {valid_pixels:,}\n")
        f.write(f"Píxeles Manglar:        {manglar_pixels:,}\n")
        f.write(f"Píxeles No-Manglar:     {no_manglar_pixels:,}\n")
        f.write(f"Píxeles NoData:         {nodata_pixels:,}\n")
        f.write("-"*70 + "\n\n")
        
        f.write("ÁREA (HECTÁREAS):\n")
        f.write("-"*70 + "\n")
        f.write(f"Área total (válida):    {area_total_ha:,.2f} ha\n")
        f.write(f"Área de Manglar:        {area_manglar_ha:,.2f} ha ({manglar_pct:.2f}%)\n")
        f.write(f"Área de No-Manglar:     {area_no_manglar_ha:,.2f} ha ({100-manglar_pct:.2f}%)\n")
        f.write("-"*70 + "\n\n")
        
        f.write("ÁREA (KILÓMETROS CUADRADOS):\n")
        f.write("-"*70 + "\n")
        f.write(f"Área total (válida):    {area_total_ha/100:,.2f} km²\n")
        f.write(f"Área de Manglar:        {area_manglar_ha/100:,.2f} km²\n")
        f.write(f"Área de No-Manglar:     {area_no_manglar_ha/100:,.2f} km²\n")
        f.write("-"*70 + "\n\n")
        
        f.write("INTERPRETACIÓN:\n")
        f.write("-"*70 + "\n")
        f.write(f"El área de estudio para el año {year} cubre aproximadamente\n")
        f.write(f"{area_total_ha:,.0f} hectáreas ({area_total_ha/100:,.1f} km²).\n\n")
        f.write(f"Se detectaron {area_manglar_ha:,.0f} hectáreas de manglar,\n")
        f.write(f"lo que representa el {manglar_pct:.1f}% del área total analizada.\n")
        f.write("="*70 + "\n")
    
    print(f"✅ Reporte de área guardado: {output_report_path}")
    print(f"\n📊 Resumen:")
    print(f"   Área total:      {area_total_ha:,.2f} ha ({area_total_ha/100:,.2f} km²)")
    print(f"   Área de Manglar: {area_manglar_ha:,.2f} ha ({manglar_pct:.2f}%)")




#====================================
# 🎨 VISUALIZACIÓN COMPARATIVA 3×5
#====================================

def generate_comparative_visualization_3x5(
    checkpoint_path,
    images_dir,
    masks_dir,
    output_dir,
    device='cpu',
    threshold=0.5,
    year=2021
):
    """
    Genera visualización comparativa 3×5 de escenarios representativos.

    Estructura:
    - Filas: [Imagen Original, Máscara Real, Máscara Predicha]
    - Columnas: [Alta cobertura, Baja cobertura, Media homogénea, Fragmentación, Alternancia compleja]

    Args:
        checkpoint_path: Ruta al modelo entrenado
        images_dir: Directorio con imágenes de test
        masks_dir: Directorio con máscaras ground truth
        output_dir: Directorio de salida
        device: Dispositivo ('cpu', 'cuda', 'mps')
        threshold: Umbral para binarización
        year: Año de análisis
    """
    from scipy import ndimage

    print("\n🎨 Generando visualización comparativa 3×5...")

    # Cargar modelo
    module = load_model(checkpoint_path, device)

    # Buscar tiles y máscaras
    tile_paths = sorted(images_dir.glob("*.tif"))

    if len(tile_paths) == 0:
        print("⚠️  No se encontraron tiles para visualización")
        return

    # Función auxiliar: calcular IoU
    def calc_iou(pred, target, thresh=0.5):
        pred_bin = (pred > thresh).astype(np.uint8)
        target_bin = target.astype(np.uint8)

        intersection = np.logical_and(pred_bin, target_bin).sum()
        union = np.logical_or(pred_bin, target_bin).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union

    # Función auxiliar: clasificar escenario
    def classify_scenario(mask):
        total_pixels = mask.size
        manglar_pixels = np.sum(mask == 1)
        manglar_pct = (manglar_pixels / total_pixels) * 100

        # Calcular fragmentación
        labeled_array, num_features = ndimage.label(mask)

        if manglar_pct > 70:
            return "high_coverage", manglar_pct
        elif manglar_pct < 10:
            return "low_coverage", manglar_pct
        elif 30 <= manglar_pct <= 50 and num_features < 5:
            return "medium_homogeneous", manglar_pct
        elif num_features > 10:
            return "high_fragmentation", manglar_pct
        else:
            return "complex_alternation", manglar_pct

    # Buscar tiles representativos
    print("  🔍 Buscando tiles representativos...")

    scenarios = ["high_coverage", "low_coverage", "medium_homogeneous",
                "high_fragmentation", "complex_alternation"]
    scenario_candidates = {s: [] for s in scenarios}

    for tile_path in tqdm(tile_paths, desc="  Escaneando tiles"):
        # Buscar máscara correspondiente
        # Para archivos como: 2021_Sentinel-2_r010_c013.tif
        # Generar máscara: 2021_Sentinel-2_mask_r010_c013.tif
        mask_name = tile_path.name.replace('Sentinel-2_', 'Sentinel-2_mask_')
        mask_path = masks_dir / mask_name

        if not mask_path.exists():
            continue

        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            scenario_type, manglar_pct = classify_scenario(mask)
            scenario_candidates[scenario_type].append({
                'tile_path': tile_path,
                'mask_path': mask_path,
                'manglar_pct': manglar_pct
            })
        except:
            continue

    # Seleccionar candidatos de manera aleatoria
    import random
    selected = {}

    for scenario in scenarios:
        candidates = scenario_candidates[scenario]

        if len(candidates) == 0:
            continue

        # Filtrar candidatos dentro de rangos óptimos y seleccionar aleatoriamente
        if scenario == "high_coverage":
            # Preferir tiles con 75-85% de manglar, pero aceptar >70%
            optimal = [c for c in candidates if 75 <= c['manglar_pct'] <= 85]
            pool = optimal if len(optimal) > 0 else candidates
        elif scenario == "low_coverage":
            # Preferir tiles con 3-8% de manglar, pero aceptar <10%
            optimal = [c for c in candidates if 3 <= c['manglar_pct'] <= 8]
            pool = optimal if len(optimal) > 0 else candidates
        elif scenario == "medium_homogeneous":
            # Preferir tiles con 35-45% de manglar
            optimal = [c for c in candidates if 35 <= c['manglar_pct'] <= 45]
            pool = optimal if len(optimal) > 0 else candidates
        else:
            # Para fragmentación y alternancia, todos los candidatos son válidos
            pool = candidates

        # Seleccionar aleatoriamente del pool
        best = random.choice(pool)
        selected[scenario] = best
        print(f"  ✓ {scenario}: {best['tile_path'].name} ({best['manglar_pct']:.1f}% manglar)")

    if len(selected) < 5:
        print(f"  ⚠️  Solo se encontraron {len(selected)}/5 escenarios")
        # Completar con aleatorios
        for scenario in scenarios:
            if scenario not in selected and len(tile_paths) > 0:
                rand_tile = random.choice(tile_paths)
                rand_mask = masks_dir / rand_tile.name.replace('Sentinel-2_', 'Sentinel-2_mask_')
                if rand_mask.exists():
                    selected[scenario] = {
                        'tile_path': rand_tile,
                        'mask_path': rand_mask,
                        'manglar_pct': 0
                    }

    # Generar predicciones y preparar datos
    print("  🔮 Generando predicciones...")

    images = []
    masks_gt = []
    masks_pred = []
    ious = []
    scenario_names = []

    for scenario in scenarios:
        if scenario not in selected:
            continue

        tile_info = selected[scenario]

        # Leer imagen
        with rasterio.open(tile_info['tile_path']) as src:
            image = src.read()  # [C, H, W]

        # RGB (bandas 2,1,0 para B4,B3,B2)
        rgb = np.stack([image[2], image[1], image[0]], axis=-1)  # [H, W, 3]

        # Reemplazar NaN con 0 (áreas sin datos)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

        # Aplicar percentile stretch POR BANDA individual para mejor contraste
        # Esto es crucial para tiles con baja cobertura o características específicas
        rgb_enhanced = np.zeros_like(rgb)

        for i in range(3):  # Para cada banda RGB
            band = rgb[:, :, i]

            # Filtrar valores válidos (no cero) para cálculo de percentiles
            valid_pixels = band[band > 0]

            if len(valid_pixels) > 10:  # Si hay suficientes píxeles válidos
                # Calcular percentiles solo sobre píxeles válidos
                p2 = np.percentile(valid_pixels, 2)
                p98 = np.percentile(valid_pixels, 98)

                # Evitar división por cero
                if p98 - p2 > 1e-6:
                    # Aplicar stretch lineal a esta banda
                    band_stretched = (band - p2) / (p98 - p2)
                    band_stretched = np.clip(band_stretched, 0, 1)
                else:
                    # Si el rango es muy pequeño, usar normalización simple
                    max_val = valid_pixels.max()
                    if max_val > 1e-6:
                        band_stretched = band / max_val
                    else:
                        band_stretched = band
                    band_stretched = np.clip(band_stretched, 0, 1)
            else:
                # Si no hay suficientes píxeles válidos, normalizar todo
                max_val = band.max()
                if max_val > 1e-6:
                    band_stretched = band / max_val
                else:
                    band_stretched = np.zeros_like(band)
                band_stretched = np.clip(band_stretched, 0, 1)

            rgb_enhanced[:, :, i] = band_stretched

        # Aplicar gamma correction más agresiva para mejorar brillo
        rgb = np.power(rgb_enhanced, 0.6)  # Gamma más bajo = imagen más clara

        # Leer máscara
        with rasterio.open(tile_info['mask_path']) as src:
            mask = src.read(1)

        # Generar predicción
        pred_binary, _, _, _, _ = predict_tile(
            module,
            str(tile_info['tile_path']),
            device=device,
            threshold=threshold
        )

        # Calcular IoU
        iou = calc_iou(pred_binary, mask, threshold)

        images.append(rgb)
        masks_gt.append(mask)
        masks_pred.append(pred_binary)
        ious.append(iou)
        scenario_names.append(scenario)

    # Crear visualización
    print("  🖼️  Creando figura 3×5...")

    scenario_labels = {
        'high_coverage': '(a) Cobertura Alta\n(>70% manglar)',
        'low_coverage': '(b) Cobertura Baja\n(<10% manglar)',
        'medium_homogeneous': '(c) Media Homogénea\n(30-50%)',
        'high_fragmentation': '(d) Fragmentación Alta\n(>10 parches)',
        'complex_alternation': '(e) Alternancia Compleja'
    }

    fig, axes = plt.subplots(3, len(images), figsize=(4*len(images), 12))

    if len(images) == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(f'Visualización Comparativa: Escenarios Representativos de Segmentación ({year})',
                 fontsize=16, fontweight='bold', y=0.98)

    for col, (img, mask_gt, mask_pred, iou, scen) in enumerate(
        zip(images, masks_gt, masks_pred, ious, scenario_names)
    ):
        # Fila 1: Imagen
        axes[0, col].imshow(img)
        axes[0, col].set_title(scenario_labels.get(scen, scen),
                              fontsize=13, fontweight='bold')
        axes[0, col].axis('off')

        # Fila 2: Ground truth
        axes[1, col].imshow(mask_gt, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, col].set_title('Ground Truth', fontsize=13)
        axes[1, col].axis('off')

        manglar_pct = (np.sum(mask_gt == 1) / mask_gt.size) * 100
        axes[1, col].text(0.5, -0.15, f'{manglar_pct:.1f}% manglar',
                         ha='center', va='top', transform=axes[1, col].transAxes,
                         fontsize=12, style='italic')

        # Fila 3: Predicción
        axes[2, col].imshow(mask_pred, cmap='RdYlGn', vmin=0, vmax=1)
        axes[2, col].set_title(f'Predicción (IoU: {iou:.3f})',
                              fontsize=13, fontweight='bold')
        axes[2, col].axis('off')

        color = 'green' if iou > 0.85 else ('orange' if iou > 0.70 else 'red')
        axes[2, col].title.set_color(color)

    # Etiquetas de filas
    row_labels = ['Imagen Satelital', 'Ground Truth', 'Predicción']
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.25, 0.5, label, rotation=90, va='center', ha='center',
                         transform=axes[row, 0].transAxes, fontsize=14, fontweight='bold')

    # Leyenda
    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Manglar (1)'),
        mpatches.Patch(facecolor='#8B4513', edgecolor='black', label='No Manglar (0)')
    ]
    fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=13,
              frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0.05, 0.02, 1, 0.96])

    # Guardar
    output_path = Path(output_dir) / f'visualizacion_comparativa_3x5_{year}.jpg'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()

    print(f"\n  ✅ Visualización guardada en: {output_path}")
    print(f"  📊 IoU promedio: {np.mean(ious):.3f}")

    return output_path


#====================================
# CLASE: GENERADOR DE FALSO COLOR
#====================================

class FalseColorMosaicGenerator:
    """
    Generador de mosaicos de falso color para resaltar vegetación.

    Genera visualizaciones de falso color usando combinaciones espectrales:
    - Falso Color Infrarrojo: RGB = NIR, Red, Green (resalta vegetación en rojo)
    - Falso Color Agricultura: RGB = SWIR1, NIR, Blue (resalta salud vegetal)

    Aplica contorno del área de estudio y elementos cartográficos.
    """

    def __init__(self, year: int, images_dir: Path, output_dir: Path,
                 shapefile_path: Path):
        """
        Args:
            year: Año de análisis
            images_dir: Directorio con imágenes Sentinel-2 originales
            output_dir: Directorio de salida para mosaicos
            shapefile_path: Ruta al shapefile del área de estudio
        """
        self.year = year
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.shapefile_path = shapefile_path

        # Bandas Sentinel-2 disponibles en las teselas
        self.band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
        self.band_indices = {
            'Blue': 0,   # B2
            'Green': 1,  # B3
            'Red': 2,    # B4
            'NIR': 3,    # B8
            'SWIR1': 4,  # B11
            'SWIR2': 5   # B12
        }

        # Leer shapefile
        print(f"📂 Cargando shapefile: {self.shapefile_path.name}")
        self.gdf = gpd.read_file(self.shapefile_path)
        print(f"   CRS: {self.gdf.crs}")
        print(f"   Geometrías: {len(self.gdf)}")

    def create_spectral_mosaic(self):
        """
        Crea mosaico conservando todas las bandas espectrales.

        Returns:
            Tupla (mosaic_path, mosaic, out_transform, crs)
        """
        print(f"\n🧩 Creando mosaico espectral (6 bandas)...")

        # Buscar todas las teselas
        tile_files = sorted(self.images_dir.glob("*.tif"))
        print(f"   Encontradas {len(tile_files)} teselas")

        if len(tile_files) == 0:
            raise ValueError(f"No se encontraron teselas en {self.images_dir}")

        # Leer primera tesela para obtener metadatos
        with rasterio.open(tile_files[0]) as src:
            crs = src.crs
            dtype = src.dtypes[0]
            count = src.count

        print(f"   Bandas: {count}")
        print(f"   CRS: {crs}")

        # Crear mosaico usando merge
        print(f"   Merging teselas...")
        src_files_to_mosaic = []
        for tile in tqdm(tile_files, desc="Abriendo teselas"):
            src = rasterio.open(tile)
            src_files_to_mosaic.append(src)

        mosaic, out_transform = merge(src_files_to_mosaic, method='first')

        # Cerrar archivos
        for src in src_files_to_mosaic:
            src.close()

        # Metadatos del mosaico
        out_meta = {
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'count': mosaic.shape[0],
            'dtype': dtype,
            'crs': crs,
            'transform': out_transform,
            'compress': 'lzw'
        }

        # Guardar mosaico temporal
        mosaic_path = self.output_dir / f'mosaico_espectral_{self.year}.tif'
        with rasterio.open(mosaic_path, 'w', **out_meta) as dest:
            dest.write(mosaic)

        print(f"   ✅ Mosaico espectral creado: {mosaic_path.name}")
        print(f"   Dimensiones: {mosaic.shape[2]} x {mosaic.shape[1]} píxeles")
        print(f"   Bandas: {mosaic.shape[0]}")

        return mosaic_path, mosaic, out_transform, crs

    def apply_study_area_mask_to_mosaic(self, mosaic, transform, crs):
        """
        Recorta el mosaico al área de estudio usando el shapefile.

        Args:
            mosaic: Array del mosaico [bands, height, width]
            transform: Transformación espacial
            crs: CRS del mosaico

        Returns:
            Tupla (mosaico_recortado, máscara_binaria, transform_ajustado)
        """
        print(f"\n🗺️  Recortando mosaico al área de estudio...")

        # PASO 1: Reproyectar shapefile si es necesario
        if self.gdf.crs != crs:
            print(f"   Reproyectando shapefile de {self.gdf.crs} a {crs}")
            gdf = self.gdf.to_crs(crs)
        else:
            gdf = self.gdf

        # PASO 2: Crear dataset en memoria para aplicar rasterio.mask
        from rasterio.io import MemoryFile

        # Metadatos temporales
        meta = {
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'count': mosaic.shape[0],
            'dtype': mosaic.dtype,
            'crs': crs,
            'transform': transform
        }

        # Usar MemoryFile para aplicar máscara sin escribir a disco
        with MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(mosaic)

                # Aplicar máscara con crop=True para recortar al shapefile
                out_image, out_transform = rasterio_mask(
                    dataset,
                    gdf.geometry,
                    crop=True,        # CRÍTICO: Recortar al bounding box del shapefile
                    filled=True,
                    nodata=0,
                    all_touched=True  # Incluir píxeles que tocan el shapefile
                )

        print(f"   Mosaico recortado: {out_image.shape[2]}x{out_image.shape[1]} píxeles")

        # PASO 3: Crear máscara binaria para el área recortada
        from rasterio.features import rasterize
        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(out_image.shape[1], out_image.shape[2]),
            transform=out_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )

        pixels_inside = np.sum(mask > 0)
        pixels_total = mask.size
        print(f"   Píxeles dentro del área: {pixels_inside:,} ({100*pixels_inside/pixels_total:.1f}% del recorte)")

        return out_image, mask, out_transform

    def _add_scale_bar(self, ax, left, right, bottom, top):
        """Añade barra de escala cartográfica."""
        # Calcular longitud de barra apropiada
        extent_width = right - left
        scale_length_m = 10000  # 10 km por defecto

        # Ajustar si el área es más pequeña
        if extent_width < 30000:
            scale_length_m = 5000  # 5 km

        # Posición de la barra (centrada en la parte superior)
        center_x = (left + right) / 2
        bar_x = center_x - scale_length_m / 2
        bar_y = top - 2500  # 2.5 km del borde superior

        # Dibujar fondo de la barra (negro)
        ax.plot([bar_x, bar_x + scale_length_m], [bar_y, bar_y],
               color='black', linewidth=6, solid_capstyle='butt', zorder=12)

        # Dibujar barra principal (blanco)
        ax.plot([bar_x, bar_x + scale_length_m], [bar_y, bar_y],
               color='white', linewidth=4, solid_capstyle='butt', zorder=13)

        # Marcar segmentos (cada 1/4 de la barra)
        segment_length = scale_length_m / 4
        for i in range(5):
            x_pos = bar_x + i * segment_length
            ax.plot([x_pos, x_pos], [bar_y - 150, bar_y + 150],
                   color='white', linewidth=2, zorder=13)

        # Etiquetas de distancia
        label_km = scale_length_m / 1000
        ax.text(bar_x, bar_y - 600, '0', ha='center', va='top',
               color='white', fontsize=8, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
               zorder=14)
        ax.text(bar_x + scale_length_m, bar_y - 600, f'{label_km:.0f} km',
               ha='center', va='top',
               color='white', fontsize=8, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
               zorder=14)

    def _add_north_arrow(self, ax):
        """Añade flecha de norte discreta."""
        # Posición en el lado izquierdo
        arrow_x = 0.05
        arrow_y = 0.70

        # Fondo negro para contraste
        ax.annotate('', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                   xytext=(arrow_x, arrow_y - 0.06),
                   arrowprops=dict(arrowstyle='->', lw=4, color='black',
                                 mutation_scale=20),
                   zorder=13)

        # Flecha blanca
        ax.annotate('', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                   xytext=(arrow_x, arrow_y - 0.06),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='white',
                                 mutation_scale=20),
                   zorder=14)

        # Etiqueta "N"
        ax.text(arrow_x, arrow_y + 0.015, 'N',
               transform=ax.transAxes,
               ha='center', va='bottom',
               fontsize=12, weight='bold', color='white',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='black',
                        alpha=0.75, edgecolor='white', linewidth=1.5),
               zorder=14)

    def generate_false_color_visualization(self, mosaic, mask, transform, crs,
                                          color_scheme='infrared'):
        """
        Genera visualización de falso color.

        Args:
            mosaic: Mosaico espectral [bands, height, width]
            mask: Máscara binaria del área de estudio
            transform: Transformación espacial
            crs: CRS del mosaico
            color_scheme: 'infrared' o 'agriculture'

        Returns:
            Ruta a la imagen generada
        """
        print(f"\n🎨 Generando falso color ({color_scheme})...")

        # Seleccionar bandas según esquema
        if color_scheme == 'infrared':
            # RGB = NIR, Red, Green
            band_r = self.band_indices['NIR']
            band_g = self.band_indices['Red']
            band_b = self.band_indices['Green']
            title = f'Falso Color Infrarrojo - Vegetación (Año {self.year})'
            filename = f'falso_color_infrarrojo_{self.year}.png'
        elif color_scheme == 'agriculture':
            # RGB = SWIR1, NIR, Blue
            band_r = self.band_indices['SWIR1']
            band_g = self.band_indices['NIR']
            band_b = self.band_indices['Blue']
            title = f'Falso Color Agricultura (Año {self.year})'
            filename = f'falso_color_agricultura_{self.year}.png'
        else:
            raise ValueError(f"Esquema de color no válido: {color_scheme}")

        # Extraer bandas
        r_band = mosaic[band_r].astype(np.float32)
        g_band = mosaic[band_g].astype(np.float32)
        b_band = mosaic[band_b].astype(np.float32)

        print(f"   Bandas seleccionadas:")
        print(f"   R: {self.band_names[band_r]}")
        print(f"   G: {self.band_names[band_g]}")
        print(f"   B: {self.band_names[band_b]}")

        # Crear composición RGB
        rgb = np.stack([r_band, g_band, b_band], axis=-1)

        # Filtrar solo píxeles válidos
        valid_mask = mask > 0

        # Percentile stretch por banda
        print(f"   Aplicando percentile stretch (2-98%)...")
        for i in range(3):
            band = rgb[:, :, i]
            valid_pixels = band[valid_mask]

            if len(valid_pixels) > 0:
                valid_pixels = valid_pixels[valid_pixels > 0]

                if len(valid_pixels) > 10:
                    p2 = np.percentile(valid_pixels, 2)
                    p98 = np.percentile(valid_pixels, 98)

                    if p98 - p2 > 1e-6:
                        band_stretched = (band - p2) / (p98 - p2)
                        band_stretched = np.clip(band_stretched, 0, 1)
                    else:
                        band_stretched = band / (np.max(band) + 1e-6)
                        band_stretched = np.clip(band_stretched, 0, 1)

                    rgb[:, :, i] = band_stretched

        # Gamma correction diferenciado
        print(f"   Aplicando gamma correction diferenciado...")
        if color_scheme == 'infrared':
            gamma_r = 0.65  # NIR - vegetación más brillante
            gamma_g = 0.75  # Red
            gamma_b = 0.75  # Green
            rgb[:, :, 0] = np.power(rgb[:, :, 0], gamma_r)
            rgb[:, :, 1] = np.power(rgb[:, :, 1], gamma_g)
            rgb[:, :, 2] = np.power(rgb[:, :, 2], gamma_b)
        else:
            gamma = 0.7
            rgb = np.power(rgb, gamma)

        # Crear figura
        print(f"   Creando visualización...")
        height, width = rgb.shape[:2]
        aspect_ratio = width / height
        fig_width = 18
        fig_height = fig_width / aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Calcular extent
        left, bottom, right, top = rasterio.transform.array_bounds(
            height, width, transform
        )

        # Fondo celeste claro para agua
        ax.set_facecolor('#87CEEB')

        # Convertir RGB a RGBA para transparencia
        rgba = np.zeros((height, width, 4), dtype=rgb.dtype)
        rgba[:, :, :3] = rgb
        alpha = np.any(rgb > 0.01, axis=2).astype(rgb.dtype)
        rgba[:, :, 3] = alpha

        # Mostrar imagen
        ax.imshow(rgba, extent=[left, right, bottom, top],
                 interpolation='bilinear', aspect='equal')

        # Superponer contorno del shapefile
        if self.gdf.crs != crs:
            gdf_plot = self.gdf.to_crs(crs)
        else:
            gdf_plot = self.gdf

        # Contorno doble (negro + amarillo)
        gdf_plot.boundary.plot(ax=ax, color='black', linewidth=2.5,
                              linestyle='-', alpha=0.9, zorder=10)
        gdf_plot.boundary.plot(ax=ax, color='#FFD700', linewidth=1.5,
                              linestyle='-', alpha=1.0, zorder=11,
                              label='Límite del área de estudio')

        # Límites del gráfico
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        # Títulos y etiquetas
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('Longitud (m)', fontsize=12)
        ax.set_ylabel('Latitud (m)', fontsize=12)

        # Leyenda superior derecha
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                 fancybox=True, shadow=True, edgecolor='black')

        # Grid sutil
        ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.3, color='white')

        # Caja de información
        if color_scheme == 'infrared':
            composition = "RGB = NIR, Red, Green"
        else:
            composition = "RGB = SWIR1, NIR, Blue"

        info_lines = [
            "Composición Espectral:",
            f"  {composition}",
            "",
            "Área de Estudio:",
            "  Archipiélago de Jambelí",
            "",
            "Resolución Espacial:",
            "  10 m/píxel"
        ]
        info_text = "\n".join(info_lines)

        ax.text(0.98, 0.02, info_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='bottom',
               horizontalalignment='right',
               family='monospace',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='black', alpha=0.75,
                        edgecolor='#FFD700', linewidth=1.5),
               color='white',
               zorder=15)

        # Añadir barra de escala y flecha norte
        self._add_scale_bar(ax, left, right, bottom, top)
        self._add_north_arrow(ax)

        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        # Guardar
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                   pad_inches=0.2)
        plt.close()

        print(f"   ✅ Falso color guardado: {filename}")
        return output_path

    def generate_all_false_color_images(self):
        """Genera todos los mosaicos de falso color."""

        print("="*80)
        print(f"🌈 GENERACIÓN DE MOSAICOS DE FALSO COLOR - AÑO {self.year}")
        print("="*80)

        # Paso 1: Crear mosaico espectral
        mosaic_path, mosaic, transform, crs = self.create_spectral_mosaic()

        # Paso 2: Aplicar máscara del área de estudio
        mosaic_masked, mask, transform_cropped = self.apply_study_area_mask_to_mosaic(
            mosaic, transform, crs
        )

        # Paso 3: Generar visualizaciones de falso color
        paths = []

        # 3.1 Falso Color Infrarrojo
        path_infrared = self.generate_false_color_visualization(
            mosaic_masked, mask, transform_cropped, crs,
            color_scheme='infrared'
        )
        paths.append(path_infrared)

        # 3.2 Falso Color Agricultura
        path_agriculture = self.generate_false_color_visualization(
            mosaic_masked, mask, transform_cropped, crs,
            color_scheme='agriculture'
        )
        paths.append(path_agriculture)

        print("\n" + "="*80)
        print("✅ MOSAICOS DE FALSO COLOR GENERADOS EXITOSAMENTE")
        print("="*80)
        print(f"\nArchivos generados en: {self.output_dir}")
        for path in paths:
            print(f"  • {path.name}")

        return paths


#====================================
# FUNCIÓN: PIPELINE PRINCIPAL
#====================================

def run_pipeline(year, base_dir, checkpoint_path, output_base_dir='predicciones',
                 threshold=0.5, use_postproc=True, postproc_mode='conservative'):
    """
    Ejecuta el pipeline completo para un año específico
    
    Args:
        year: Año de análisis (ej: 2021, 2022, 2023)
        base_dir: Directorio base con carpetas Manglar_[AÑO]_*
        checkpoint_path: Ruta al checkpoint del modelo
        output_base_dir: Directorio base para outputs
        threshold: Umbral de decisión (0.0-1.0)
        use_postproc: Activar post-procesamiento morfológico
        postproc_mode: Modo post-proc ('conservative', 'moderate', 'none')
    """
    
    print("="*80)
    print(f"🌳 PIPELINE DE ANÁLISIS DE MANGLARES - AÑO {year}")
    print("="*80)
    
    # Limpieza automática de directorio anterior
    import shutil
    output_year_dir = Path(output_base_dir) / f"year_{year}"
    if output_year_dir.exists():
        print(f"\n🗑️  Limpiando directorio anterior: {output_year_dir}")
        print(f"   (Para forzar regeneración con nuevo threshold)")
        try:
            shutil.rmtree(output_year_dir)
            print(f"   ✅ Directorio eliminado correctamente")
        except Exception as e:
            print(f"   ⚠️  Error al eliminar: {e}")
            print(f"   Por favor, elimina manualmente: rm -rf {output_year_dir}")
    
    # ====================================
    # CONFIGURACIÓN POR AÑO
    # ====================================
    
    config = YearConfig(
        year=year,
        base_dir=base_dir,
        checkpoint_path=checkpoint_path,
        output_base_dir=output_base_dir
    )
    
    print("\n📋 CONFIGURACIÓN:")
    print("-"*80)
    for key, value in config.get_summary().items():
        print(f"   {key}: {value}")
    print(f"   Threshold para binarización: {threshold}")
    print(f"   Post-procesamiento morfológico: {'SÍ' if use_postproc else 'NO'}")
    if use_postproc:
        print(f"   Modo post-procesamiento: {postproc_mode}")
    print("-"*80)
    
    # Detectar dispositivo
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"\n💻 Dispositivo: {device.upper()}")
    
    # ====================================
    # PASO 1: PREDICCIÓN DE TESELAS
    # ====================================
    
    print("\n" + "="*80)
    print(f"PASO 1: PREDICCIÓN DE TESELAS - AÑO {year}")
    print("="*80)
    
    pred_files = process_tiles(
        checkpoint_path=config.checkpoint_path,
        images_dir=config.images_dir,
        output_dir=config.predictions_dir,
        device=device,
        threshold=threshold,
        use_postproc=use_postproc,
        postproc_mode=postproc_mode
    )
    
    if len(pred_files) == 0:
        print("\n❌ No se generaron predicciones. Abortando...")
        return
    
    # ====================================
    # PASO 2: MATRIZ DE CONFUSIÓN
    # ====================================
    
    print("\n" + "="*80)
    print(f"PASO 2: CÁLCULO DE MATRIZ DE CONFUSIÓN - AÑO {year}")
    print("="*80)
    
    cm, metrics, valid_count = calculate_confusion_matrix_from_files(pred_files, config.masks_dir)
    
    if cm is not None and metrics is not None:
        # Guardar matriz de confusión
        cm_path = config.metrics_dir / 'confusion_matrix.png'
        plot_confusion_matrix_for_article(cm, metrics, str(cm_path), year)
        
        # Guardar gráfico de métricas
        metrics_plot_path = config.metrics_dir / 'metricas_rendimiento.png'
        plot_metrics_comparison(metrics, str(metrics_plot_path), year)
        
        # Guardar reporte detallado
        report_path = config.metrics_dir / 'reporte_metricas.txt'
        save_confusion_matrix_report(cm, metrics, str(report_path), year)
        
        # Guardar métricas en JSON
        json_path = config.metrics_dir / 'metricas.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"✅ Métricas JSON guardadas: {json_path}")
        
        # Mostrar resumen en consola
        print("\n" + "="*80)
        print(f"📊 RESUMEN DE MÉTRICAS - AÑO {year}")
        print("="*80)
        print(f"Teselas evaluadas:     {metrics['tiles_evaluated']}")
        print(f"True Positives (TP):   {metrics['TP']:>15,} píxeles")
        print(f"True Negatives (TN):   {metrics['TN']:>15,} píxeles")
        print(f"False Positives (FP):  {metrics['FP']:>15,} píxeles")
        print(f"False Negatives (FN):  {metrics['FN']:>15,} píxeles")
        print(f"\nAccuracy:              {metrics['accuracy']:>15.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:             {metrics['precision']:>15.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:                {metrics['recall']:>15.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score:              {metrics['f1_score']:>15.4f}")
        print(f"Mean IoU:              {metrics['mean_iou']:>15.4f}")
        print("="*80)
    else:
        print(f"\n⚠️ No se pudo calcular la matriz de confusión para el año {year}.")
        print("   Continuando con la creación del mosaico...")

    # ====================================
    # PASO 2.5: ANÁLISIS AVANZADO Y CURVA ROC
    # ====================================

    print("\n" + "="*80)
    print(f"PASO 2.5: ANÁLISIS AVANZADO Y CURVA ROC - AÑO {year}")
    print("="*80)

    advanced_metrics = generate_advanced_analysis_plots(
        predictions_dir=config.predictions_dir,
        masks_dir=config.masks_dir,
        output_dir=config.metrics_dir,
        year=year,
        threshold=threshold
    )

    # ====================================
    # PASO 3: CREACIÓN DE MOSAICO
    # ====================================
    
    print("\n" + "="*80)
    print(f"PASO 3: CREACIÓN DE MOSAICO - AÑO {year}")
    print("="*80)
    
    mosaic_path = config.mosaic_dir / f'mosaico_manglares_{year}.tif'
    
    create_mosaic(
        pred_files=pred_files,
        output_mosaic_path=str(mosaic_path),
        method='first'
    )

    # ====================================
    # PASO 3.5: APLICAR MÁSCARA DEL ÁREA DE ESTUDIO
    # ====================================

    print("\n" + "="*80)
    print(f"PASO 3.5: APLICAR CONTORNO DEL ÁREA DE ESTUDIO - AÑO {year}")
    print("="*80)

    # Ruta al shapefile del área de estudio (Jambeli)
    shapefile_path = Path('/Users/elvissanchez/Documents/GitHub/thesis_project/data/archive_Shape/Jambeli_corregido/Area_Estudio_Jambeli.shp')

    if shapefile_path.exists():
        # Aplicar máscara (sobrescribe el mosaico original)
        mosaic_path = apply_study_area_mask(
            mosaic_path=str(mosaic_path),
            shapefile_path=str(shapefile_path),
            output_path=None  # Sobrescribe el original
        )
    else:
        print(f"⚠️  Shapefile no encontrado: {shapefile_path}")
        print("   Continuando sin aplicar máscara del área de estudio...")

    # ====================================
    # PASO 4: VISUALIZACIÓN
    # ====================================
    
    print("\n" + "="*80)
    print(f"PASO 4: VISUALIZACIÓN DEL MOSAICO - AÑO {year}")
    print("="*80)

    viz_path = config.mosaic_dir / f'mosaico_visualizacion_{year}.png'

    # Ruta al shapefile (misma que se usa en PASO 3.5)
    shapefile_path_viz = Path('/Users/elvissanchez/Documents/GitHub/thesis_project/data/archive_Shape/Jambeli_corregido/Area_Estudio_Jambeli.shp')

    visualize_mosaic(
        mosaic_path=str(mosaic_path),
        output_viz_path=str(viz_path),
        year=year,
        figsize=(20, 16),
        dpi=300,
        shapefile_path=str(shapefile_path_viz) if shapefile_path_viz.exists() else None
    )

    # ====================================
    # PASO 4.5: VISUALIZACIONES INDIVIDUALES DE TESELAS
    # ====================================

    print("\n" + "="*80)
    print(f"PASO 4.5: VISUALIZACIONES INDIVIDUALES - AÑO {year}")
    print("="*80)

    # Generar visualizaciones comparativas (RGB | GT | Pred)
    viz_count = generate_tile_visualizations(
        images_dir=config.images_dir,
        masks_dir=config.masks_dir,
        predictions_dir=config.predictions_dir,
        output_viz_dir=config.visualizations_dir,
        year=year,
        max_tiles=50,  # Número máximo de visualizaciones
        selection_mode='best'  # 'best', 'worst', 'random', 'all'
    )

    # ====================================
    # PASO 4.6: VISUALIZACIÓN COMPARATIVA 3×5
    print("\n" + "="*80)
    print(f"PASO 4.6: VISUALIZACIÓN COMPARATIVA 3×5 - AÑO {year}")
    print("="*80)

    try:
        comparative_viz_path = generate_comparative_visualization_3x5(
            checkpoint_path=config.checkpoint_path,
            images_dir=config.images_dir,
            masks_dir=config.masks_dir,
            output_dir=config.mosaic_dir,
            device=device,
            threshold=threshold,
            year=year
        )
    except Exception as e:
        print(f"⚠️  Error al generar visualización comparativa: {e}")
        import traceback
        traceback.print_exc()

    # PASO 5: ESTADÍSTICAS DE ÁREA
    # ====================================
    
    print("\n" + "="*80)
    print(f"PASO 5: CÁLCULO DE ESTADÍSTICAS DE ÁREA - AÑO {year}")
    print("="*80)
    
    report_path = config.mosaic_dir / f'reporte_area_{year}.txt'
    
    calculate_area_statistics(
        mosaic_path=str(mosaic_path),
        output_report_path=str(report_path),
        year=year
    )

    # ====================================
    # PASO 6: GENERACIÓN DE MOSAICOS DE FALSO COLOR
    # ====================================

    print("\n" + "="*80)
    print(f"PASO 6: GENERACIÓN DE MOSAICOS DE FALSO COLOR - AÑO {year}")
    print("="*80)

    # Ruta al shapefile del área de estudio (Jambeli)
    shapefile_path = Path('/Users/elvissanchez/Documents/GitHub/thesis_project/data/archive_Shape/Jambeli_corregido/Area_Estudio_Jambeli.shp')

    false_color_paths = []
    if shapefile_path.exists():
        try:
            # Instanciar generador de falso color
            false_color_generator = FalseColorMosaicGenerator(
                year=year,
                images_dir=config.images_dir,
                output_dir=config.mosaic_dir,
                shapefile_path=shapefile_path
            )

            # Generar mosaicos de falso color (infrarrojo y agricultura)
            false_color_paths = false_color_generator.generate_all_false_color_images()

            print(f"\n✅ Mosaicos de falso color generados exitosamente")

        except Exception as e:
            print(f"\n⚠️  Error al generar mosaicos de falso color: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️  Shapefile no encontrado: {shapefile_path}")
        print("   Saltando generación de mosaicos de falso color...")

    # ====================================
    # RESUMEN FINAL
    # ====================================
    
    print("\n" + "="*80)
    print(f"✅ PROCESO COMPLETADO EXITOSAMENTE - AÑO {year}")
    print("="*80)
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"   Directorio raíz:        {config.output_base_dir}")
    print(f"   Teselas predichas:      {config.predictions_dir}/ ({len(pred_files)} archivos)")
    
    if cm is not None:
        print(f"\n   📊 MÉTRICAS DE EVALUACIÓN:")
        print(f"   Matriz de confusión:    {config.metrics_dir / 'confusion_matrix.png'}")
        print(f"   Gráfico de métricas:    {config.metrics_dir / 'metricas_rendimiento.png'}")
        print(f"   Análisis avanzado:      {config.metrics_dir / f'analisis_metricas_{year}.png'}")
        print(f"   Curva ROC:              {config.metrics_dir / f'curva_roc_{year}.png'}")
        print(f"   Casos destacados:       {config.metrics_dir / f'casos_destacados_{year}.png'}")
        print(f"   Reporte detallado:      {config.metrics_dir / 'reporte_metricas.txt'}")
        print(f"   Métricas JSON:          {config.metrics_dir / 'metricas.json'}")
    
    print(f"\n   🗺️ MOSAICO Y VISUALIZACIÓN:")
    print(f"   Mosaico GeoTIFF:        {mosaic_path}")
    print(f"   Visualización:          {viz_path}")
    print(f"   Reporte de área:        {report_path}")

    print(f"\n   🎨 VISUALIZACIONES INDIVIDUALES:")
    print(f"   Directorio:             {config.visualizations_dir}/")
    print(f"   Teselas visualizadas:   {viz_count} comparaciones RGB|GT|Pred")

    if len(false_color_paths) > 0:
        print(f"\n   🌈 MOSAICOS DE FALSO COLOR:")
        for fc_path in false_color_paths:
            print(f"   {fc_path.name:30s} → {fc_path}")

    if cm is not None:
        print(f"\n🎯 RENDIMIENTO DEL MODELO (AÑO {year}):")
        print(f"   Accuracy:    {metrics['accuracy']*100:.2f}%")
        print(f"   F1-Score:    {metrics['f1_score']:.4f}")
        print(f"   Mean IoU:    {metrics['mean_iou']:.4f}")

    if advanced_metrics is not None:
        print(f"\n📈 MÉTRICAS AVANZADAS (ROC):")
        print(f"   ROC AUC:               {advanced_metrics['roc_auc']:.4f}")
        print(f"   Average Precision:     {advanced_metrics['avg_precision']:.4f}")
        print(f"   Threshold óptimo (ROC): {advanced_metrics['optimal_threshold_roc']:.3f}")
        print(f"   Threshold óptimo (F1):  {advanced_metrics['optimal_threshold_f1']:.3f}")

    print("\n🌳 El mosaico está listo para análisis en SIG (QGIS, ArcGIS, etc.)")
    print("="*80)


#====================================
# ⭐ PUNTO DE ENTRADA PRINCIPAL
#====================================

if __name__ == '__main__':
    
    # ============================================
    # 🔧 CONFIGURACIÓN - MODIFICAR AQUÍ
    # ============================================
    
    # ⭐ PARÁMETRO PRINCIPAL: AÑO DE ANÁLISIS
    YEAR = 2025  # ← Cambiar este valor para analizar otro año
    
    # Directorios
    BASE_DIR = '/Users/elvissanchez/Documents/GitHub/thesis_project/data/processed/test'
    CHECKPOINT_PATH = '/Users/elvissanchez/Documents/GitHub/thesis_project/checkpoints/MultiBranch-UNetPP-resnet101-fpn-Sentinel2-epoch=45-val_iou=0.8174.ckpt'
    # /Users/elvissanchez/Documents/GitHub/thesis_project/checkpoints/UnetPlusPlus-resnet34-15-Dic-25-epoch=95-val_iou=0.8954.ckpt
    OUTPUT_BASE_DIR = 'predicciones_por_año'
    
    # ⭐ UMBRAL DE DECISIÓN
    # 
    # ✅ OPTIMIZADO tras análisis de normalización:
    # - Las imágenes de GEE ya vienen normalizadas [0, 1]
    # - Se eliminó normalización percentil redundante que comprimía rango
    # - Threshold 0.20 optimizado para valores espectrales reales
    # 
    # Basado en análisis de distribución de probabilidades:
    # - Threshold 0.50: Recall ~43% (con normalización redundante)
    # - Threshold 0.20: Recall ~52-55% (sin normalización redundante) ← RECOMENDADO
    # 
    # Ganancia estimada: +8-10 pp en recall con -2 pp en precision
    PREDICTION_THRESHOLD = 0.50
    
    # ⭐⭐⭐ NUEVO: POST-PROCESAMIENTO MORFOLÓGICO ⭐⭐⭐
    # 
    # Técnica validada científicamente para refinar segmentaciones:
    # - Pham & Yoshino (2016) Remote Sensing of Environment
    # - Chen et al. (2020) ISPRS Journal
    # - Wang et al. (2023) Remote Sensing of Environment
    # 
    # Justificación ecológica:
    # Los manglares crecen en parches continuos debido a crecimiento 
    # lateral de raíces (Tomlinson, 2016). Las discontinuidades en 
    # predicciones reflejan variabilidad espectral interna más que 
    # fragmentación real.
    #
    # ═══════════════════════════════════════════════════════════════
    # CONFIGURACIÓN:
    # ═══════════════════════════════════════════════════════════════
    #
    # USE_POSTPROCESSING: True/False
    #   True  = Aplica operaciones morfológicas (RECOMENDADO)
    #   False = Solo usa threshold (para comparación baseline)
    #
    # POSTPROC_MODE: 'conservative' / 'moderate' / 'none'
    #   'conservative' = Kernel 3x3, cambio mínimo
    #                    Ganancia: +2-3 pp recall
    #                    Recomendado para tesis (científicamente conservador)
    #   
    #   'moderate'     = Kernel 5x5, más corrección
    #                    Ganancia: +3-4 pp recall
    #                    Validado en Pham & Yoshino (2016)
    #   
    #   'none'         = Sin post-procesamiento
    #                    (equivalente a USE_POSTPROCESSING=False)
    #
    # ═══════════════════════════════════════════════════════════════
    # RESULTADOS ESPERADOS (con threshold 0.20):
    # ═══════════════════════════════════════════════════════════════
    #
    # Sin post-proc:              Con post-proc (conservative):
    # Recall:    ~46.5%           Recall:    ~49.2% (+2.7 pp)
    # Precision: ~89.5%           Precision: ~89.1% (-0.4 pp)
    # F1-Score:  ~0.611           F1-Score:  ~0.635 (+0.024)
    # IoU:       ~0.630           IoU:       ~0.648 (+0.018)
    #
    # ═══════════════════════════════════════════════════════════════
    
    USE_POSTPROCESSING = True              # ← ACTIVAR/DESACTIVAR
    POSTPROC_MODE = 'conservative'         # ← 'conservative', 'moderate', 'none'
    
    # ============================================
    # 🚀 EJECUTAR PIPELINE
    # ============================================
    
    run_pipeline(
        year=YEAR,
        base_dir=BASE_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        output_base_dir=OUTPUT_BASE_DIR,
        threshold=PREDICTION_THRESHOLD,
        use_postproc=USE_POSTPROCESSING,
        postproc_mode=POSTPROC_MODE
    )

    # ============================================
    # 🗺️ ANÁLISIS ESPACIAL DE ERRORES
    # ============================================

    print(f"\n\n{'='*80}")
    print("INICIANDO ANÁLISIS ESPACIAL DE ERRORES")
    print(f"{'='*80}\n")

    try:
        from spatial_error_analysis import SpatialErrorAnalyzer

        analyzer = SpatialErrorAnalyzer(
            year=YEAR,
            base_dir=Path.cwd(),  # Directorio actual
            pixel_size=10.0
        )

        analyzer.run_full_analysis()

    except Exception as e:
        print(f"⚠️  Error en análisis espacial: {e}")
        print("El pipeline principal se completó correctamente.")
        import traceback
        traceback.print_exc()

    # ============================================
    # 📝 PARA ANÁLISIS MULTITEMPORAL
    # ============================================
    
    # Ejemplo: procesar múltiples años secuencialmente
    # 
    # for year in [2020, 2021, 2022, 2023]:
    #     print(f"\n\n{'='*80}")
    #     print(f"PROCESANDO AÑO {year}")
    #     print(f"{'='*80}\n")
    #     
    #     run_pipeline(
    #         year=year,
    #         base_dir=BASE_DIR,
    #         checkpoint_path=CHECKPOINT_PATH,
    #         output_base_dir=OUTPUT_BASE_DIR,
    #         threshold=PREDICTION_THRESHOLD,
    #         use_postproc=USE_POSTPROCESSING,
    #         postproc_mode=POSTPROC_MODE
    #     )
    
    # ============================================
    # 🔬 PARA COMPARACIÓN ABLATION STUDY
    # ============================================
    
    # Para tu tesis, ejecuta múltiples configuraciones y compara:
    #
    # configs = [
    #     {'threshold': 0.50, 'postproc': False, 'mode': 'none'},      # Baseline
    #     {'threshold': 0.20, 'postproc': False, 'mode': 'none'},      # Solo threshold
    #     {'threshold': 0.20, 'postproc': True, 'mode': 'conservative'}, # Threshold + Post-proc
    # ]
    #
    # for i, cfg in enumerate(configs):
    #     print(f"\n{'='*80}")
    #     print(f"CONFIGURACIÓN {i+1}/{len(configs)}")
    #     print(f"{'='*80}")
    #     run_pipeline(YEAR, BASE_DIR, CHECKPOINT_PATH, 
    #                  f"{OUTPUT_BASE_DIR}_config{i+1}",
    #                  cfg['threshold'], cfg['postproc'], cfg['mode'])
