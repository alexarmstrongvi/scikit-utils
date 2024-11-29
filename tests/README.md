Tests
=====

# CLI tests
fit_supervised_model

```sh
# Check help message is generated
>> fit_supervised_model.py -h
# Check help runtime errors are raised about all missing args
>> fit_supervised_model.py

>> fit_supervised_model.py
>> fit_supervised_model.py -i ./data/classification_test.parquet -o tmp/ --estimator ExtraTreeClassifier --log-file -c config1.yml config2.yml
