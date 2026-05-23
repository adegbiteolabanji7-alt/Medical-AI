"""
explore_dicom.py

DICOM is the file format every medical image is stored in.
Every scan yoy performed as a sonographer was DICOM.

Each file has two parts:
1. Pixel data - the actual image
2. metadata tags - patient info, modality, scan settings.

The tags are what our router uses to make decisions.
"""

import  pydicom
def load_dicom(filepath):
    """ Load a DICOM file and return the dataset."""
    dataset = pydicom.dcmread(filepath)
    return dataset


def print_key_tags(dataset):
    """
    print the tags that matter for routing.
    
    Modality tag (0008, 0060) is the one we care about the most,
    It tells us: CT, MR, US(Ultrasound), PT(PET), etc.
    """
    print("=" * 40)
    print("KEY DICOM TAGS")
    print("=" * 40)
    print(f"Modality: {dataset.Modality}")
    print(f'patient: {dataset.PatientName}')
    print(f"Study Date: {dataset.StudyDate}")
    print(f"Image Size: {dataset.Rows} x {dataset.Columns}")
    print("=" * 40)

if __name__ == "__main__":
    #pydicom 3.x has built-in example datasets ready to use
    print("Loading built-in CT example....\n")
    
    ds = pydicom.examples.ct
    print_key_tags(ds)