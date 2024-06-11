# CDI-Coherence APP

## Description

This repository contains the files to launch an [Streamlit](https://streamlit.io/) App locally and interact with a self-supervised model trained with synthetic data.

## Set up and first time usage

**1. Create and activate environment**

In order to use this app locally you will need to create a Conda environment with Python 3.9. Let's name it, for example, **"app_coherence"**.

```
conda create -n app_coherence python=3.9
```

Once created and installed all default libraries, let's activate the environment.

```
conda activate app_coherence
```

**2. Install requirements**

Once activated the enviroment, install the requirements just as usual:

```
pip install -r requirements.txt
```

**3. Configure secrets file**

In the root directory, with your favorite text editor or from your IDE, create an ```/.streamlit/secrets.toml``` file and define a password on it.

For example:

```
password = "mypassword"
```

More info from the official Streamlit documentation [here](https://docs.streamlit.io/develop/api-reference/connections/secrets.toml) and [here](https://docs.streamlit.io/develop/concepts/connections/secrets-management).


**4. Launch app**

Once installed the requirements, let's launch the app with the following command:

```
streamlit run app.py
```

## Future usage

In the future, you just need to activate the environment and then launch the app.

```
conda activate app_coherence
streamlit run app.py
```


## Folder structure

Here's the folder structure of the project.

```
streamlit_coherence/   # Root directory.
|- .streamlit/         # Contains secrets.toml file with password defintion
|- artifacts/          # Contains artifacts for model, projection, and index.
|- data/               # Contain csv file with test data
|- visualization/      # Contains htmls files
|- app.py              # Main Streamlit app file
|- requirements.txt    # Contains needed libraries and versions
|- utils.py            # Utility functions used in app
```
