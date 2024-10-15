#!/usr/bin/env python
import os, json
from constructs import Construct
from cdktf import App, TerraformStack
from cdktf import TerraformAsset, AssetType

from imports.aws.provider import AwsProvider

from imports.aws.iam_role import IamRole
from imports.aws.iam_role_policy import IamRolePolicy
from imports.aws.iam_role_policy_attachment import IamRolePolicyAttachment

from imports.aws.s3_bucket import S3Bucket
from imports.aws.s3_bucket_object import S3BucketObject

from imports.aws.glue_catalog_database import GlueCatalogDatabase
from imports.aws.glue_catalog_table import GlueCatalogTable
from imports.aws.glue_crawler import GlueCrawler, GlueCrawlerS3Target
from imports.aws.glue_job import GlueJob, GlueJobCommand
from cdktf import Token

from imports.aws.kinesis_stream import KinesisStream
from imports.aws.kinesis_stream import KinesisStreamConfig





class MyStack(TerraformStack):
    def __init__(self, scope: Construct, id: str):
        super().__init__(scope, id)
        self.app_name = "brad-ml-training"
        self.aws_region = os.environ["AWS_REGION"]
        self.aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        self.aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        self.aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)

        AwsProvider(self, 
                    "aws", 
                    region=self.aws_region, 
                    access_key=self.aws_access_key_id, 
                    secret_key=self.aws_secret_access_key, 
                    token=self.aws_session_token)

        # create media store bucket
        self.input_bucket = S3Bucket(
            self,
            f"{self.app_name}-input",
            bucket=f"{self.app_name}-input",
            force_destroy=True,
        )

        self.output_bucket = S3Bucket(
            self,
            f"{self.app_name}-output",
            bucket=f"{self.app_name}-output",
            force_destroy=True,
        )

        user_data_asset = TerraformAsset(self, "Asset1", 
                               path="assets/user-data.jsonl", 
                               type=AssetType.FILE)

        S3BucketObject(
            self,
            f"user-data",
            bucket=self.input_bucket.bucket,
            key=user_data_asset.file_name,
            source=user_data_asset.path
        )


        export_data_asset = TerraformAsset(self, "Asset2", 
                               path="assets/myExport.csv", 
                               type=AssetType.FILE)
        S3BucketObject(
            self,
            f"export-data",
            bucket=self.input_bucket.bucket,
            key=export_data_asset.file_name,
            source=export_data_asset.path
        )


        spark_script_asset = TerraformAsset(self, "Asset3", 
                               path="assets/gender-mapping.py", 
                               type=AssetType.FILE)     

        S3BucketObject(
            self,
            f"spark-script",
            bucket=self.input_bucket.bucket,
            key=spark_script_asset.file_name,
            source=spark_script_asset.path
        )


        ufo_asset = TerraformAsset(self, "Asset5", 
                               path="assets/ufo_fullset.csv", 
                               type=AssetType.FILE)  

        S3BucketObject(
            self,
            f"asset5",
            bucket=self.input_bucket.bucket,
            key=ufo_asset.file_name,
            source=ufo_asset.path
        )







        KinesisStream(self, "kinesis_stream",
            name=f"{self.app_name}-stream",
            retention_period=24,
            shard_count=1,
            shard_level_metrics=["IncomingBytes", "OutgoingBytes"],
            tags={
                "Environment": "test"
            }
        )

        # create glue crawler
        # ============================================================

        self.crawler_role = IamRole(
            self,
            f"crawler-role",
            name=f"{self.app_name}-crawler-role",
            assume_role_policy=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": {
                        "Action": "sts:AssumeRole",
                        "Principal": {
                            "Service": "glue.amazonaws.com",
                        },
                        "Effect": "Allow",
                        "Sid": "",
                    },
                }
            ),          
        )
        IamRolePolicy(self, "glue-crawler-policy",
            name=f"{self.app_name}_crawler-role_policy",
            policy=json.dumps({
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "s3:GetObject",
                                        "s3:PutObject"
                                    ],
                                    "Resource": [
                                        f"{self.input_bucket.arn}/*",
                                        f"{self.output_bucket.arn}/*"
                                    ]
                                }
                            ],
                    "Version": "2012-10-17"
                }),
            role=self.crawler_role.id
        )

        IamRolePolicyAttachment(
            self,
            "glue-crawler-policy-attachment",
            policy_arn="arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole",
            role=self.crawler_role.name,
        )
                
        self.glue_database = GlueCatalogDatabase(
            self,
            "catalog-database",
            name=f"{self.app_name}-catalog-database"
        )

        GlueCrawler(
            self,
            "glue-crawler",
            database_name=self.glue_database.name,
            name=f"{self.app_name}-crawler",
            role=self.crawler_role.arn,
            schedule="cron(0 1 * * ? *)", # Run every day at 1am
            s3_target=[GlueCrawlerS3Target(
                path="s3://${" + self.input_bucket.bucket + "}"
            )]
        )


        # create glue job role

        # create glue job
        # ============================================================

        GlueJob(self, "example",
            command=GlueJobCommand(
                script_location="s3://${" + self.input_bucket.bucket + "}/" + spark_script_asset.file_name,
            ),
            name=f"{self.app_name}-ml-glue-job",
            default_arguments = {
                "--customer-executor-env-vars": f"AWS_GLUE_DATABASE_NAME={self.glue_database.name},AWS_GLUE_TABLE_NAME=user_data_jsonl,S3_BUCKET_NAME={self.output_bucket.bucket}"
            },
            role_arn=Token.as_string(self.crawler_role.arn)
        )




app = App()
MyStack(app, "infra")

app.synth()
