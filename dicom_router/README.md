# DICOM Router

A small Python script that sorts DICOM files into folders by modality (CT, MR, US and so on), using the Modality tag (0008,0060). It reads only the header, never the pixel data.

## Why I built it

In a hospital, files from CT scanners, MRI machines and ultrasound all land in the same place, unsorted. Before any AI model can use them, they need to be organised by modality, because a model fed the wrong data is a patient safety problem. After five years scanning in Nigeria I know real archives are messy, so I made sure one bad file can't stop the whole batch.

## How it works

1. It looks for .dcm files in the input folder.
2. It reads the Modality tag from each file's header only (stop_before_pixels=True). Routing needs one tag, so loading the image data would waste time and memory.
3. It copies each file into a named folder such as CT_computed_tomography or MR_magnetic_resonance.
4. A modality I haven't listed goes into a folder ending in _other, so nothing is lost.
5. A file that can't be read is logged as a warning and copied into UNKNOWN for a person to check.

## Test run

I ran it on four files: a CT, an MR and a radiotherapy plan (all from pydicom's sample data) plus a fake file I made to break it. The three real files went to their own folders and the fake one went to UNKNOWN. The log ended with Routed: 3 | Failed: 1.

## Running it

From inside the dicom_router folder:

    pip install pydicom
    python router.py

It reads from data/input and writes to data/output. The paths are set at the bottom of router.py.

## Tests

    pip install pytest
    python -m pytest -v

Four tests cover a valid CT file, a corrupt file, end-to-end routing and an empty input folder. They use temporary folders so leftover output can't make a test pass by accident.

One test exists because of a real bug: I had typed the exception name in lowercase, which only crashed when a corrupt file arrived, so normal runs never showed it. I fixed it, wrote the test, then put the typo back to check the test failed. It did.

## Files

- router.py: the routing code
- test_router.py: the tests
- explore_dicom.py: my script for looking at DICOM tags

## Design decisions

Why header only? Pixel data can be several MB per file, and the router doesn't need it. I haven't benchmarked it on a large archive yet, so I can't give numbers.

Why copy, not move? The original files are never touched. In clinical systems you don't destroy source data.

Why logging, not print? Logs have timestamps and severity levels and can be saved to a file, which gives an audit trail. That matters in regulated settings.

## What it doesn't do yet

- It only looks in the top folder, not subfolders.
- It only picks up files ending in .dcm, so DICOM files with no extension are missed.
- Two files with the same name overwrite each other in the output.
- A valid DICOM file with no Modality tag is also treated as unreadable.
- The paths are fixed in the code, with no command line options.
- No anonymisation, no PACS connection and no log file yet.

## Regulatory context

This is only file sorting. It makes no clinical decisions and is not a medical device. It does not remove patient details, so it shouldn't be used on real patient data without anonymisation and proper data governance (UK GDPR and NHS rules). In a real pipeline it would sit before a quality check and an anonymisation step.

## Next

Subfolder search, checking for DICOM by header instead of file extension, safe filenames, command line options, then anonymisation.
