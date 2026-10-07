# DICOM Router

A small Python script I wrote that sorts DICOM files into folders by modality (CT, MR, US and so on). It only reads the header, not the pixel data.

I built it because the first question with any pile of imaging files is where each one should go, and the answer is in the Modality tag. After five years scanning in Nigeria I know archives are messy, so I made sure one bad file can't stop the whole batch.

## What it does

It looks for .dcm files in data/input and reads each header with pydicom. I set stop_before_pixels=True so it skips the image data, since the router only needs one tag and loading pixels would waste time and memory. The file is then copied (not moved, so the original is never touched) into a folder named after its modality. Modalities I haven't listed go into a folder ending in _other. If a file can't be read, it is logged and copied into UNKNOWN so someone can look at it.

## Test run

I ran it on four files: a CT, an MR, a radiotherapy plan (all from pydicom's sample data) and a fake file I made to break it. The CT, MR and plan went to their own folders and the fake file went to UNKNOWN. The log ended with Routed: 3 | Failed: 1.

## Running it

    pip install pydicom
    python router.py

It reads from data/input and writes to data/output.

## Tests

    pip install pytest
    python -m pytest -v

There are four tests. They use temporary folders so old output can't make a test pass by accident. One of them exists because of a real bug I had: I typed the exception name in lowercase, which only crashed when a corrupt file turned up. Normal files never hit that line, so I missed it for months. I wrote the test first, watched it fail, then fixed it.

## What it doesn't do yet

- It only looks in the top folder, not subfolders.
- It only picks up files ending in .dcm, so DICOM files with no extension are missed.
- Two files with the same name overwrite each other in the output.
- A valid DICOM file with no Modality tag is treated as unreadable.
- The folder paths are written into the code, not passed in.
- No anonymisation, no PACS connection, and no log file yet.

It is not a medical device and makes no clinical decisions. It does not remove patient details, so I wouldn't use it on real patient data without anonymising first.

## Next

Subfolder search, checking for DICOM by header instead of extension, safe filenames, command line arguments, then anonymisation.
