"""
AWS Cloud Persistence and Storage Service for Sulcus Clinical AI.
Manages Amazon DynamoDB persistence for patient diagnostic telemetry,
S3 medical imaging archiving, and FDA/HIPAA audit event logging.
"""
import os
import json
import logging
import datetime
from decimal import Decimal
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger("sulcus.aws_service")

# AWS Resource Configuration
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))
DYNAMO_PATIENT_TABLE = os.environ.get("DYNAMO_PATIENT_TABLE", "sulcus_patient_records")
DYNAMO_AUDIT_TABLE = os.environ.get("DYNAMO_AUDIT_TABLE", "sulcus_audit_events")
S3_IMAGING_VAULT = os.environ.get("S3_IMAGING_VAULT", "sulcus-imaging-vault-145023097151")


def _convert_floats_to_decimals(obj: Any) -> Any:
    """Recursively converts Python float objects to Decimal for DynamoDB serialization."""
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    elif isinstance(obj, dict):
        return {k: _convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_decimals(v) for v in obj]
    return obj


class AWSService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AWSService, cls).__new__(cls)
            cls._instance._init_clients()
        return cls._instance

    def _init_clients(self):
        try:
            self.session = boto3.Session(region_name=AWS_REGION)
            self.dynamodb = self.session.resource("dynamodb", region_name=AWS_REGION)
            self.s3_client = self.session.client("s3", region_name=AWS_REGION)
            self.patient_table = self.dynamodb.Table(DYNAMO_PATIENT_TABLE)
            self.audit_table = self.dynamodb.Table(DYNAMO_AUDIT_TABLE)
            self.enabled = True
            logger.info("AWSService initialized successfully for region %s", AWS_REGION)
        except Exception as e:
            logger.warning("AWSService initialization disabled: %s", str(e))
            self.enabled = False

    def save_patient_record(self, patient_id: str, record_data: Dict[str, Any]) -> bool:
        """Persists clinical inference telemetry to DynamoDB table."""
        if not self.enabled:
            return False
        try:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            item = {
                "patient_id": str(patient_id),
                "timestamp": now_iso,
                "recorded_at": now_iso,
                **_convert_floats_to_decimals(record_data),
            }
            self.patient_table.put_item(Item=item)
            logger.info("Persisted diagnostic record for patient %s to DynamoDB", patient_id)
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error("Failed to write to DynamoDB %s: %s", DYNAMO_PATIENT_TABLE, str(e))
            return False
        except Exception as e:
            logger.error("Unexpected error saving patient record: %s", str(e))
            return False

    def archive_imaging_artifact(
        self,
        patient_id: str,
        filename: str,
        file_bytes: bytes,
        content_type: str = "application/octet-stream",
        artifact_type: str = "raw_scan",
    ) -> Optional[str]:
        """Archives raw scan or visual plate to S3 imaging vault."""
        if not self.enabled:
            return None
        try:
            key = f"vault/{patient_id}/{artifact_type}_{filename}"
            self.s3_client.put_object(
                Bucket=S3_IMAGING_VAULT,
                Key=key,
                Body=file_bytes,
                ContentType=content_type,
                Metadata={
                    "patient-id": str(patient_id),
                    "artifact-type": artifact_type,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
            )
            s3_uri = f"s3://{S3_IMAGING_VAULT}/{key}"
            logger.info("Archived %s artifact to %s", artifact_type, s3_uri)
            return s3_uri
        except Exception as e:
            logger.error("Failed to archive artifact to S3: %s", str(e))
            return None

    def log_audit_event(
        self,
        event_type: str,
        actor_id: str,
        details: Dict[str, Any],
    ) -> bool:
        """Logs HIPAA/FDA compliant audit event to DynamoDB."""
        if not self.enabled:
            return False
        try:
            event_id = f"EVT-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            item = {
                "event_id": event_id,
                "timestamp": now_iso,
                "event_type": event_type,
                "actor_id": actor_id,
                "details": _convert_floats_to_decimals(details),
            }
            self.audit_table.put_item(Item=item)
            return True
        except Exception as e:
            logger.error("Failed to log audit event: %s", str(e))
            return False

    def get_recent_patient_records(self, patient_id: str, limit: int = 10) -> list:
        """Retrieves historical diagnostic records for a patient from DynamoDB."""
        if not self.enabled:
            return []
        try:
            from boto3.dynamodb.conditions import Key
            response = self.patient_table.query(
                KeyConditionExpression=Key("patient_id").eq(str(patient_id)),
                ScanIndexForward=False,
                Limit=limit,
            )
            items = response.get("Items", [])
            # Convert Decimal back to float/int for JSON serialization
            def _clean(val):
                if isinstance(val, Decimal):
                    return float(val) if val % 1 else int(val)
                elif isinstance(val, dict):
                    return {k: _clean(v) for k, v in val.items()}
                elif isinstance(val, list):
                    return [_clean(v) for v in val]
                return val
            return [_clean(item) for item in items]
        except Exception as e:
            logger.error("Failed to query patient records: %s", str(e))
            return []


# Module singleton instance
aws_service = AWSService()
