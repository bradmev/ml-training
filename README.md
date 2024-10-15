

# ml-training



## notebooks

#### Usage
1. Create a .env file using env.sample as example content

2. Create venv and install requirements 
    ```
    pyenv virtualenv 3.10 ml_training
    pyenv activate ml_training
    pip install jupyter notebook ipykernel jupyterlab
    python -m ipykernel install --user --name bedrock --display-name "Python (ml_training-3.10)"
    
    ```
3. install requirements
```
    pyenv activate ml_training
    pip install -r requirements.txt
```
4. Open and run your notebooks in vs code




## infra


#### Requirements
Install tfswitch to manage terraform versions
```
brew install warrensbox/tap/tfswitch
tfswitch --latest
```

Install terraform, pipenv and cdktf 
```
brew install pipenv
npm install --global cdktf-cli@latest
```

#### Usage

Run get to populate imports
```
cdktf get
```

Run synth to create terraform code and test operation.
```
source .env && cdktf synth
```

Run deploy to deploy
```
source .env && cdktf deploy
```

Run destory to destroy
```
source .env && cdktf destroy
```