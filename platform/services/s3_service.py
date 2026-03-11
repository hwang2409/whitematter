"""S3 proxy service — operates on user's S3 using their credentials."""
import boto3
from typing import BinaryIO
from services.credential_service import CredentialService


class S3Service:
    def __init__(self, cred_service: CredentialService | None = None):
        self.cred_service = cred_service or CredentialService()

    def _get_client(self, encrypted_access_key: bytes, encrypted_secret_key: bytes, region: str):
        return boto3.client(
            "s3",
            aws_access_key_id=self.cred_service.decrypt(encrypted_access_key),
            aws_secret_access_key=self.cred_service.decrypt(encrypted_secret_key),
            region_name=region,
        )

    def list_buckets(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str) -> list[dict]:
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        resp = client.list_buckets()
        return [{"name": b["Name"], "created": b["CreationDate"].isoformat()} for b in resp["Buckets"]]

    def create_bucket(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str, bucket_name: str):
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        config = {"LocationConstraint": region} if region != "us-east-1" else {}
        create_args = {"Bucket": bucket_name}
        if config:
            create_args["CreateBucketConfiguration"] = config
        client.create_bucket(**create_args)

    def list_objects(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str,
                     bucket: str, prefix: str = "") -> list[dict]:
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        items = []
        for p in resp.get("CommonPrefixes", []):
            items.append({"key": p["Prefix"], "type": "folder"})
        for obj in resp.get("Contents", []):
            items.append({
                "key": obj["Key"],
                "type": "file",
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })
        return items

    def upload_file(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str,
                    bucket: str, key: str, file_obj: BinaryIO):
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        client.upload_fileobj(file_obj, bucket, key)

    def delete_object(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str,
                     bucket: str, key: str):
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        client.delete_object(Bucket=bucket, Key=key)

    def generate_presigned_url(self, encrypted_ak: bytes, encrypted_sk: bytes, region: str,
                               bucket: str, key: str, expires_in: int = 3600) -> str:
        client = self._get_client(encrypted_ak, encrypted_sk, region)
        return client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
        )
