# bank_marketing_analysis documentation!

## Description

This is an end-to-end system for predicting if a client would sbuscribe to a term deposit

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `az storage blob upload-batch -d` to recursively sync files in `data/` up to `bank-marketing-data/data/`.
* `make sync_data_down` will use `az storage blob upload-batch -d` to recursively sync files from `bank-marketing-data/data/` to `data/`.


