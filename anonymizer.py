"""DICOM anonymization module for research purposes.

Implements the ResearchCTAnonymizer class to remove protected health
information while preserving technical imaging parameters required for
analysis.

Requirements: pydicom, tqdm
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pydicom
from tqdm import tqdm


class ResearchCTAnonymizer:
    """Anonymizer for knee CT DICOM data in research settings.

    This class removes all PHI (Protected Health Information) tags while
    preserving imaging technical parameters (e.g., slice thickness, pixel
    spacing).  It also replaces UIDs with consistent pseudonyms.

    Parameters
    ----------
    output_dir : str or Path
        Base directory where anonymized DICOM files will be written.
    log_dir : str or Path, optional
        Directory for log files, by default "./logs".
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        log_dir: Union[str, Path] = "./logs"
    ) -> None:
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # ---------- Tags to be removed (Protected Health Information) ----------
        # Grouped by DICOM information category for readability

        # Patient identifiers
        self.tags_to_remove: List[Tuple[int, int]] = [
            (0x0010, 0x0010),  # PatientName
            (0x0010, 0x0020),  # PatientID
            (0x0010, 0x0030),  # PatientBirthDate
            (0x0010, 0x0040),  # PatientSex
            (0x0010, 0x1010),  # PatientAge
            (0x0010, 0x1040),  # PatientAddress
            (0x0010, 0x2154),  # PatientTelephoneNumbers
            (0x0010, 0x21C0),  # PregnancyStatus
            (0x0010, 0x4000),  # PatientComments

            # Institution and physician information
            (0x0008, 0x0080),  # InstitutionName
            (0x0008, 0x0081),  # InstitutionAddress
            (0x0008, 0x0090),  # ReferringPhysicianName
            (0x0008, 0x0092),  # ReferringPhysicianAddress
            (0x0008, 0x0094),  # ReferringPhysicianTelephoneNumbers
            (0x0008, 0x1010),  # StationName
            (0x0008, 0x1030),  # StudyDescription
            (0x0008, 0x103E),  # SeriesDescription
            (0x0008, 0x1040),  # InstitutionalDepartmentName
            (0x0008, 0x1048),  # Physician(s) of Record
            (0x0008, 0x1049),  # Physician(s) Reading Study
            (0x0008, 0x1050),  # PerformingPhysicianName
            (0x0008, 0x1060),  # NameOfPhysiciansReadingStudy

            # Request and scheduling information
            (0x0032, 0x1032),  # RequestingPhysician
            (0x0032, 0x1060),  # RequestedProcedureDescription

            # Device and software identifiers
            (0x0018, 0x1000),  # DeviceSerialNumber
            (0x0018, 0x1020),  # SoftwareVersions
            (0x0018, 0x1030),  # ProtocolName

            # Dates and times
            (0x0008, 0x0020),  # StudyDate
            (0x0008, 0x0021),  # SeriesDate
            (0x0008, 0x0022),  # AcquisitionDate
            (0x0008, 0x0023),  # ContentDate
            (0x0008, 0x0030),  # StudyTime
            (0x0008, 0x0031),  # SeriesTime
            (0x0008, 0x0032),  # AcquisitionTime
            (0x0008, 0x0033),  # ContentTime
            (0x0008, 0x0082),  # InstitutionCodeSequence
            (0x0008, 0x0083),  # PhysicianIDSequence
            (0x0008, 0x009C),  # ConsultingPhysicianName

            # Additional personal identifiers
            (0x0010, 0x0021),  # IssuerOfPatientID
            (0x0010, 0x0032),  # PatientBirthTime
            (0x0010, 0x1000),  # OtherPatientIDs
            (0x0010, 0x1001),  # OtherPatientNames
            (0x0010, 0x1002),  # OtherPatientIDsSequence
            (0x0010, 0x1005),  # PatientBirthName
            (0x0010, 0x1060),  # PatientMotherBirthName
            (0x0010, 0x1990),  # MedicalRecordLocator
            (0x0010, 0x2000),  # MedicalAlerts
            (0x0010, 0x2110),  # Allergies
            (0x0010, 0x2150),  # CountryOfResidence
            (0x0010, 0x2152),  # RegionOfResidence
            (0x0010, 0x2155),  # PatientInsurancePlanCodeSequence
            (0x0010, 0x2160),  # EthnicGroup
            (0x0010, 0x2180),  # Occupation
            (0x0010, 0x21A0),  # SmokingStatus
            (0x0010, 0x21B0),  # AdditionalPatientHistory
            (0x0010, 0x21D0),  # LastMenstrualDate
            (0x0010, 0x21F0),  # PatientReligiousPreference
            (0x0010, 0x2201),  # PatientSpeciesDescription
            (0x0010, 0x2202),  # PatientSpeciesCodeSequence
            (0x0010, 0x2203),  # PatientBreedDescription
            (0x0010, 0x2299),  # ResponsiblePerson
            (0x0010, 0x2297),  # ResponsibleOrganization
        ]

        # Tags to preserve (imaging technical parameters)
        self.tags_to_keep: List[Tuple[int, int]] = [
            (0x0028, 0x0010),  # Rows
            (0x0028, 0x0011),  # Columns
            (0x0028, 0x0030),  # PixelSpacing
            (0x0018, 0x0050),  # SliceThickness
            (0x0018, 0x0088),  # SpacingBetweenSlices
            (0x0020, 0x0032),  # ImagePositionPatient
            (0x0020, 0x0037),  # ImageOrientationPatient
            (0x0018, 0x0060),  # KVP
            (0x0018, 0x0090),  # DataCollectionDiameter
            (0x0018, 0x1100),  # ReconstructionDiameter
            (0x0018, 0x1110),  # DistanceSourceToDetector
            (0x0018, 0x1111),  # DistanceSourceToPatient
            (0x0018, 0x1120),  # GantryDetectorTilt
            (0x0018, 0x1130),  # TableHeight
            (0x0018, 0x1140),  # RotationDirection
            (0x0018, 0x1150),  # ExposureTime
            (0x0018, 0x1151),  # XRayTubeCurrent
            (0x0018, 0x1160),  # FilterType
            (0x0018, 0x1210),  # ConvolutionKernel
            (0x0028, 0x0100),  # BitsAllocated
            (0x0028, 0x0101),  # BitsStored
            (0x0028, 0x0102),  # HighBit
            (0x0028, 0x0103),  # PixelRepresentation
            (0x0028, 0x1050),  # WindowCenter
            (0x0028, 0x1051),  # WindowWidth
            (0x0028, 0x1052),  # RescaleIntercept
            (0x0028, 0x1053),  # RescaleSlope
            (0x0028, 0x1054),  # RescaleType
            (0x0020, 0x000D),  # StudyInstanceUID
            (0x0020, 0x000E),  # SeriesInstanceUID
            (0x0020, 0x0011),  # SeriesNumber
            (0x0020, 0x0013),  # InstanceNumber
        ]

        # Processing statistics
        self.stats: Dict[str, Any] = {
            'processed_files': 0,
            'successful_anonymization': 0,
            'failed_files': 0,
            'errors': []
        }

    def _setup_logging(self) -> None:
        """Configure logging to both file and console.

        Note:
            basicConfig is only effective on the first call; subsequent
            instances will reuse the same configuration.
        """
        log_file = (
            self.log_dir
            / f"anonymization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _generate_anonymous_id(
        self,
        original_value: str,
        salt: str = "knee_ct_research"
    ) -> str:
        """Generate a consistent, anonymized identifier using MD5.

        MD5 is used for fast, non‑cryptographic pseudonym generation.
        The first 16 hex digits of the hash are returned.

        Parameters
        ----------
        original_value : str
            The original string to be hashed.
        salt : str, optional
            A salt to avoid dictionary attacks, by default "knee_ct_research".

        Returns
        -------
        str
            Anonymous 16‑character hexadecimal ID.
        """
        hash_input = f"{original_value}_{salt}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def anonymize_single_file(
        self,
        input_path: Path,
        output_path: Path
    ) -> bool:
        """Anonymize a single DICOM file.

        Parameters
        ----------
        input_path : Path
            Path to the original DICOM file.
        output_path : Path
            Destination path for the anonymized file.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            ds = pydicom.dcmread(input_path)

            # Generate a consistent patient ID based on the parent folder name
            patient_id_from_path = input_path.parent.name
            anonymous_patient_id = self._generate_anonymous_id(
                patient_id_from_path
            )
            if hasattr(ds, 'PatientID'):
                ds.PatientID = anonymous_patient_id

            self._remove_sensitive_tags(ds)
            self._rewrite_identifiers(ds)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            ds.save_as(output_path, write_like_original=False)

            self.stats['successful_anonymization'] += 1
            self.logger.info("Anonymized: %s", input_path)
            return True

        except Exception as exc:
            error_msg = f"Error processing {input_path}: {exc}"
            self.logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            self.stats['failed_files'] += 1
            return False

    def _remove_sensitive_tags(self, dataset: pydicom.Dataset) -> None:
        """Remove all sensitive tags from the dataset.

        Only catches KeyError when a tag is missing; other exceptions are
        logged but not raised.

        Parameters
        ----------
        dataset : pydicom.Dataset
            The DICOM dataset to be cleaned.
        """
        for tag in self.tags_to_remove:
            if tag in dataset:
                try:
                    del dataset[tag]
                except KeyError:
                    pass
                except Exception as exc:
                    self.logger.debug(
                        "Failed to delete tag %s: %s", tag, exc
                    )

    def _rewrite_identifiers(self, dataset: pydicom.Dataset) -> None:
        """Replace Study and Series Instance UIDs with consistent pseudonyms.

        Parameters
        ----------
        dataset : pydicom.Dataset
            DICOM dataset whose UIDs will be overwritten.
        """
        if hasattr(dataset, 'StudyInstanceUID'):
            dataset.StudyInstanceUID = self._generate_anonymous_id(
                dataset.StudyInstanceUID, "study"
            )
        if hasattr(dataset, 'SeriesInstanceUID'):
            dataset.SeriesInstanceUID = self._generate_anonymous_id(
                dataset.SeriesInstanceUID, "series"
            )

    def batch_anonymize(
        self,
        input_dir: Union[str, Path],
        output_subdir: str = "anonymized",
        recursive: bool = True,
        file_pattern: str = "*.dcm"
    ) -> int:
        """Batch process all DICOM files in a directory.

        Parameters
        ----------
        input_dir : str or Path
            Root directory containing original DICOM files.
        output_subdir : str, optional
            Subdirectory within the output base directory, by default
            "anonymized".  Use "" to place files directly in the output root.
        recursive : bool, optional
            If True, search subdirectories recursively, by default True.
        file_pattern : str, optional
            Glob pattern for DICOM files, by default "*.dcm".

        Returns
        -------
        int
            Number of successfully anonymized files.
        """
        input_path = Path(input_dir)
        output_path = (
            self.output_dir / output_subdir
            if output_subdir else self.output_dir
        )
        output_path.mkdir(parents=True, exist_ok=True)

        if recursive:
            dicom_files = list(input_path.rglob(file_pattern))
        else:
            dicom_files = list(input_path.glob(file_pattern))

        self.logger.info("Found %d DICOM files in %s", len(dicom_files), input_path)

        success_count = 0
        for dicom_file in tqdm(dicom_files, desc="Anonymizing"):
            relative_path = dicom_file.relative_to(input_path)
            output_file = output_path / relative_path
            if self.anonymize_single_file(dicom_file, output_file):
                success_count += 1
            self.stats['processed_files'] += 1

        self.logger.info(
            "Batch finished: %d/%d successful", success_count, len(dicom_files)
        )
        self._generate_report(input_path)
        return success_count

    def _generate_report(self, input_directory: Path) -> None:
        """Save a JSON processing report with a timestamp.

        Parameters
        ----------
        input_directory : Path
            The original input directory (for the record).
        """
        report = {
            'processing_date': datetime.now().isoformat(),
            'input_directory': str(input_directory),
            'output_directory': str(self.output_dir),
            'statistics': self.stats,
            'total_files_processed': self.stats['processed_files'],
            'success_rate': (
                self.stats['successful_anonymization']
                / max(1, self.stats['processed_files'])
            )
        }
        report_filename = (
            f"anonymization_report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_path = self.output_dir / report_filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info("Report saved to %s", report_path)

    def verify_complete_anonymization(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Verify that a DICOM file has been fully anonymized.

        This method checks only a subset of common PHI tags; it does not
        guarantee full compliance with all regulatory standards.

        Parameters
        ----------
        file_path : str or Path
            Path to the DICOM file to verify.

        Returns
        -------
        dict
            A dictionary with keys:
            - 'file' (str): The verified file path.
            - 'completely_anonymized' (bool): Whether the checked tags are
              absent.
            - 'remaining_demographics' (list[str]): Names of demographic tags
              still present.
            - 'remaining_identifiers' (list[str]): Names of identifier tags
              still present.
            If an error occurs, 'completely_anonymized' is False and an
            'error' key is included.
        """
        try:
            ds = pydicom.dcmread(file_path)
            verification: Dict[str, Any] = {
                'file': str(file_path),
                'completely_anonymized': True,
                'remaining_demographics': [],
                'remaining_identifiers': []
            }

            demographic_tags = [
                (0x0010, 0x0010),  # PatientName
                (0x0010, 0x0030),  # PatientBirthDate
                (0x0010, 0x0040),  # PatientSex
                (0x0010, 0x1010),  # PatientAge
            ]
            for tag in demographic_tags:
                if tag in ds:
                    name = (
                        pydicom.datadict.keyword_for_tag(tag) or str(tag)
                    )
                    verification['remaining_demographics'].append(name)
                    verification['completely_anonymized'] = False

            identifier_tags = [
                (0x0008, 0x0080),  # InstitutionName
                (0x0008, 0x0090),  # ReferringPhysicianName
                (0x0008, 0x1010),  # StationName
                (0x0010, 0x1040),  # PatientAddress
            ]
            for tag in identifier_tags:
                if tag in ds:
                    name = (
                        pydicom.datadict.keyword_for_tag(tag) or str(tag)
                    )
                    verification['remaining_identifiers'].append(name)
                    verification['completely_anonymized'] = False

            return verification

        except Exception as exc:
            return {
                'file': str(file_path),
                'completely_anonymized': False,
                'error': str(exc)
            }