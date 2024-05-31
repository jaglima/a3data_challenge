# Iris Dataset Classifier

Project created with MLOps-Template cookiecutter. For more info: https://mlopsstudygroup.github.io/mlops-guide/


## 📋 Requirements

* DVC
* Python3 and pip
* Access to IBM Cloud Object Storage

## 🏃🏻 Running Project

~/.aws/credentials

```credentials
[default]
aws_access_key_id = {Key ID}
aws_secret_access_key = {Access Key}
```


### ✅ Pre-commit Testings

In order to activate pre-commit testing you need ```pre-commit```

Installing pre-commit with pip
```
pip install pre-commit
```

Installing pre-commit on your local repository. Keep in mind this creates a Github Hook.
```
pre-commit install
```

Now everytime you make a commit, it will run some tests defined on ```.pre-commit-config.yaml``` before allowing your commit.

**Example**
```
$ git commit -m "Example commit"

black....................................................................Passed
pytest-check.............................................................Passed
```