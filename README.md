# Welcome to the ISAAC 🍎 Demo! 👋

## Description

This repository contains the files to launch an [Streamlit](https://streamlit.io/) app locally to show a prototype of the Interpretable System for Automatic Analysis of Coherence (ISAAC).

## Set up and first time usage

**0. Clone this repo.**

**1. Create and activate environment**

In order to use this app locally you will need to create a Conda environment with Python 3.10. Let's name it, for example, **"app_coherence"**.

```
conda create -n app_coherence python=3.10
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

<!-- **X. Configure secrets file**

In the root directory, with your favorite text editor or from your IDE, create an ```/.streamlit/secrets.toml``` file and define a password on it.

For example:

```
password = "mypassword"
```

More info from the official Streamlit documentation [here](https://docs.streamlit.io/develop/api-reference/connections/secrets.toml) and [here](https://docs.streamlit.io/develop/concepts/connections/secrets-management). -->

**3. Download the required models artifacts**

- Option 1 (manually):
While logged in with your corporate email credentials, please download the compressed file from [here](https://drive.google.com/drive/folders/1wteSsc1jlOqwLsMmaugSsmvL7U5WtB67). Then, uncompress the file and place it in the root directory of this project.
It should be a single folder named **artifacts** with two subfolders: **biencoder** and **crossencoder**. After this, you can delete the compressed file **artifacts.zip**.

- Option 2 (with terminal commands):
Once installed the requirements, execute the following commands.

First, download the compressed file:
```
gdown 1UwI02AW08g0EAh6DYIkHtgzQtwWM_Pz6
```

Second, decompress the file and delete the compressed file.
```
unzip artifacts.zip && rm artifacts.zip
```

**4. Launch app**

Once installed the requirements, let's launch the app with the following command:

```
streamlit run Welcome.py
```

To shut down the app, just press <kbd>Ctrl</kbd> + <kbd>C</kbd>.

## Future usage

In the future, you just need to activate the environment and then launch the app.

```
conda activate app_coherence
streamlit run Welcome.py
```


## Folder structure

Here's the folder structure of the project.

```
streamlit_coherence/        # Root directory
|- artifacts/               # Contains artifacts for both models
|- data/                    # Contains csv file with test data
|- pages/                   # Contains the .py files for each page
|- visualization/           # Contains htmls files
|- requirements.txt         # Contains needed libraries and versions
|- utils_biencoder.py       # Utility functions for bi-encoder model
|- utils_crossencoder.py    # Utility functions for cross-encoder model
|- Welcome.py               # Main Streamlit app file
```