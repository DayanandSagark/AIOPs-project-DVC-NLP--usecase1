# dvc-project-template
DVC project template

## Important reference
* [Bag of Words]
* [TF-IDF]

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 05- initialize the dvc project
```bash
dvc init
```

### STEP 06- ignore dat file to be tracked by git as it taken care by dvc
```bash
echo "*.logs" >> logs/.gitignore
git rm -r --cached 'data\data.xml'
git commit -m "stop tracking data\data.xml"
git add 'data\.gitignore' dvc.lock

git rm -r --cached 'artifacts\features\train.pkl'
git commit -m "stop tracking artifacts\features\train.pkl"
git rm -r --cached 'artifacts\features\test.pkl'
git commit -m "stop tracking artifacts\features\test.pkl"
dvc repro -f
dvc dag
```

### STEP 06- commit and push the changes to the remote repository


