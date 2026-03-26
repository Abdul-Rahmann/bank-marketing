#!/bin/bash
touch dvc.yaml                    # DVC pipeline definition
mkdir -p configs
touch configs/train.yaml
touch configs/data.yaml

# Adding Azure/DVC to requirements
echo "dvc[s3]" >> requirements.txt
echo "dvc-azure" >> requirements.txt
echo "mlflow" >> requirements.txt
echo "python-dotenv" >> requirements.txt