# DICOM Router

Automatically routes DICOM medical imaging files into organised 
subfolders based on the Modality tag (0008,0060).

## Clinical motivation

In a hospital environment, DICOM files arrive from multiple imaging 
systems — CT scanners, MRI machines, ultrasound — all landing in the 
same directory, unsorted. Before any AI model can process this data, 
it must be correctly organised by modality.

Misrouted files mean a model trained on the wrong data. In clinical 
AI, that is a patient safety issue.

## How it works

1. Scans an input folder for `.dcm` files
2. Reads the Modality tag from each file's metadata only
   (`stop_before_pixels=True` — faster, never loads image data)
3. Copies each file into a named subfolder: `CT_computed_tomography/`,
   `MR_magnetic_resonance/`, `US_ultrasound/`, etc.
4. Unknown modalities get their own folder rather than being lost
5. Unreadable files are logged as warnings — one bad file never 
   crashes the pipeline

## Usage

```bash
python dicom_router/router.py
```

Edit the `input_dir` and `output_dir` paths in `router.py` to point 
to your folders.

## Files

- `router.py` — the main routing engine
- `explore_dicom.py` — interactive exploration of DICOM tag structure

## Design decisions

**Why metadata only?** DICOM pixel data can be several MB per file. 
Reading only the tags makes the router viable at scale — thousands 
of overnight scans without memory issues.

**Why copy rather than move?** Non-destructive by default. The 
original files are always preserved. In clinical systems, you never 
destroy source data.

**Why logging over print?** Logs are timestamped, severity-levelled, 
and can be written to file for audit purposes — a regulatory 
requirement in clinical AI deployments.

## Regulatory context

In a production clinical environment this pipeline would sit upstream 
of a quality check and anonymisation step before any data reaches an 
AI model. Patient data governance (GDPR, NHS DSP Toolkit) requires 
that PII is handled and minimised at every stage.