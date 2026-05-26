"""
router.py
-------
The DICOM Router.

Reads a folder of DICOM files and routes each one into organized subfolders based on Modality tag (0008,0060).

Clinical context: Dicom files arrive from CT scanners, MRI machines, ultrasound systems - all mixed together. Correct routing is the first step in any clinical AI pipeline. Get this wrong and your AI models train on the wrong data.
"""


import pydicom
import shutil
import logging 
from pathlib import Path

logging.basicConfig(level = logging.INFO,format = "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
MODALITY_NAMES = {
    "CT": "CT_computed_tomography",
    "MR": "MR_magnetic_resonance",
    "US": "US_ultrasound",
    "PT": "PT_pet_scan",
    "DX": "DX_digital_xray",
    "CR": "CR_computed_radiography",
    "NM": "NM_nuclear_medicine",
}

def get_modality(filepath):
    """
    Extract the Modality tag from a DICOM file.
    stop_before_pixels = True means we read only the tags, not the image data. This makes the router fast- we never load pixel data we dont need.
    
    Returns None if the file is unreadable. Real clinical data is messy. one bad file must never crash the whole router.
    """
    try:
        ds = pydicom.dcmread(str(filepath),stop_before_pixels = True)
        return ds.Modality
    except exception as e:
        logger.warning(f'Could not read {filepath.name}: {e}')
        return None
    
def route_file(filepath, output_dir):
        """
        Copy a single dicom file to the correct output subfolder.
        """
        modality = get_modality(filepath)
        if modality is None:
            dest_folder = output_dir/ "UNKNOWN"
        else:
            folder_name = MODALITY_NAMES.get(modality, f"{modality}_other")
            dest_folder = output_dir/ folder_name
        dest_folder.mkdir(parents=True, exist_ok= True)
        shutil.copy2(str(filepath), dest_folder / filepath.name)
        logger.info(f"Routed {filepath.name} -> {dest_folder.name}/")

        return modality 

def run_router(input_dir, output_dir):
    """
    Route all DICOm files in input_dir into organized subfolders.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"input folder not found: {input_dir}")
        return
    dicom_files = list(input_path.glob("*.dcm"))
    if not dicom_files:
        logger.warning(f"No .dcm files found in {input_dir}")
        return
    logger.info(f"Found {len(dicom_files)} DICOM files to route")

    routed, failed = 0, 0
    for filepath in dicom_files:
        modality = route_file(filepath, output_path)
        if modality:
            routed += 1
        else:
            failed +=  1
    logger.info(f"Complete.Routed: {routed} | Failed: {failed}")
if __name__ == "__main__":
    run_router(
        input_dir="data/input",
        output_dir="data/output"
    )
                

