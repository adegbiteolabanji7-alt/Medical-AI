# Import Path so we can build file paths that work on any operating system
from pathlib import Path
# Import shutil so we can copy a real DICOM file into our temporary folder
import shutil
# Import a helper that locates the small sample DICOM files bundled with pydicom
from pydicom.data import get_testdata_file
# Import the two functions we are testing from our own router.py
from router import get_modality, run_router


# Helper: copy a real sample CT file into a given folder and return its new path
def make_ct_file(folder):
    # Find the bundled CT sample on disk
    source = get_testdata_file("CT_small.dcm")
    # Decide where the copy will live inside the temporary folder
    destination = folder / "CT_small.dcm"
    # Copy the file there, keeping its metadata
    shutil.copy2(source, destination)
    # Hand the new path back to the test that asked for it
    return destination


# Test A: a real CT file should be recognised as modality "CT"
# tmp_path is a pytest feature: a fresh empty folder, deleted afterwards
def test_get_modality_reads_ct(tmp_path):
    # Put a real CT file into the fresh folder
    ct_file = make_ct_file(tmp_path)
    # Ask the router which modality the header says it is
    result = get_modality(ct_file)
    # Check that the answer is exactly "CT", otherwise the test fails
    assert result == "CT"


# Test B: a file that is not DICOM must return None, not crash
def test_get_modality_returns_none_for_bad_file(tmp_path):
    # Build the path for a fake DICOM file inside the fresh folder
    bad_file = tmp_path / "bad.dcm"
    # Write plain text into it, so it has no DICM header
    bad_file.write_text("this is not dicom")
    # Ask the router to read it; this must not raise an error
    result = get_modality(bad_file)
    # Check that the failure path returned None
    assert result is None


# Test C: running the router on a CT file should create the CT folder
def test_run_router_creates_ct_folder(tmp_path):
    # Make a temporary input folder inside the fresh folder
    input_dir = tmp_path / "input"
    # Create it on disk
    input_dir.mkdir()
    # Choose where the output folder will be (it does not exist yet)
    output_dir = tmp_path / "output"
    # Put a real CT file into the input folder
    make_ct_file(input_dir)
    # Run the whole router on these temporary folders
    run_router(input_dir, output_dir)
    # Check that the CT output folder now exists
    assert (output_dir / "CT_computed_tomography").exists()
    # Check that the CT file was actually copied into it
    assert (output_dir / "CT_computed_tomography" / "CT_small.dcm").exists()


# Test D: an empty input folder should return cleanly and create no output
def test_run_router_handles_empty_folder(tmp_path):
    # Make an empty input folder
    input_dir = tmp_path / "input"
    # Create it on disk
    input_dir.mkdir()
    # Choose an output path that must stay uncreated
    output_dir = tmp_path / "output"
    # Run the router; it should log a warning and return without crashing
    run_router(input_dir, output_dir)
    # Check that nothing was created in the output location
    assert not output_dir.exists()
