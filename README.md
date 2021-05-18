# Molecules
This sample shows how to create, train, evaluate, and make predictions on a machine learning model, using [Apache Beam](https://beam.apache.org/), [Google Cloud Dataflow](https://cloud.google.com/dataflow/), [TensorFlow](https://www.tensorflow.org/), and [AI Platform](https://cloud.google.com/ai-platform/).

The dataset for this sample is extracted from the [National Center for Biotechnology Information](https://www.ncbi.nlm.nih.gov/) ([FTP source](ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF)).
The file format is [`SDF`](https://en.wikipedia.org/wiki/Chemical_table_file#SDF).
Here's a more detailed description of the [MDL/SDF file format](http://c4.cabrillo.edu/404/ctfile.pdf).

These are the general steps:
 1. Data extraction
 2. Preprocessing the data
 3. Training the model
 4. Doing predictions

## Initial setup

### Python virtual environment

Install a [Python virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments).

Run the following to set up and activate a new virtual environment:
```bash
virtualenv -p python3 venv
source env/bin/activate
```

Once you are done with the tutorial, you can deactivate the virtual environment by running `deactivate`.

### Installing requirements
You can use the `requirements.txt` to install the dependencies.
```bash
pip install -U -r requirements.txt
```


## Quickstart
We'll start by creating a bucket:

The script requires a working directory, which is where all the temporary files and other intermediate data will be stored throughout the full run. 
```bash
gsutil mb -l <region> gs://<your-bucket>
```

```bash
# Set the working directory
export WORK_DIR=gs://<your-bucket>
```

Each SDF file contains data for 25,000 molecules.
The script will download only 5 SDF files to the working directory by default.

## Manual run

### Data Extraction
> Source code: [`data-extractor.py`](data-extractor.py)

This is a data extraction tool to download SDF files from the specified FTP source.
The data files will be stored within a `data` subdirectory inside the working directory.

To store data files locally:
```bash
python data-extractor.py --work-dir $WORK_DIR --max-data-files 5
```

### Preprocessing
> Source code: [`preprocess.py`](preprocess.py)

This is an [Apache Beam](https://beam.apache.org/) pipeline that will do all the preprocessing necessary to train a Machine Learning model.
It uses [tf.Transform](https://github.com/tensorflow/transform), which is part of [TensorFlow Extended](https://www.tensorflow.org/tfx/), to do any processing that requires a full pass over the dataset.

For this sample, we're doing a very simple feature extraction.
It uses Apache Beam to parse the SDF files and count how many Carbon, Hydrogen, Oxygen, and Nitrogen atoms a molecule has.
To create more complex models we would need to extract more sophisticated features, such as [Coulomb Matrices](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301).

We will eventually train a [Neural Network](https://skymind.ai/wiki/neural-network) to do the predictions.
Neural Networks are more stable when dealing with small values, so it's always a good idea to [normalize](https://en.wikipedia.org/wiki/Feature_scaling) the inputs to a small range (typically from 0 to 1).
Since there's no maximum number of atoms a molecule can have, we have to go through the entire dataset to find the minimum and maximum counts.
Fortunately, tf.Transform integrates with our Apache Beam pipeline and does that for us.

After preprocessing our dataset, we also want to split it into a training and evaluation dataset.
The training dataset will be used to train the model.
The evaluation dataset contains elements that the training has never seen, and since we also know the "answers" (the molecular energy), we'll use these to validate that the training accuracy roughly matches the accuracy on unseen elements.

These are the general steps:
1) Parse the SDF files
2) Feature extraction (count atoms)
3) *Normalization (normalize counts to 0 to 1)
4) Split into 80% training data and 20% evaluation data

> (*) During the normalization step, the Beam pipeline doesn't actually apply the tf.Transform function to our data.
It analyzes the whole dataset to find the values it needs (in this case the minimums and maximums), and with that it creates a TensorFlow graph of operations with those values as constants.
This graph of operations will be applied by TensorFlow itself, allowing us to pass the unnormalized data as inputs rather than having to normalize them ourselves during prediction.

The `preprocess.py` script will preprocess all the data files it finds under `$WORK_DIR/data/`, which is the path where `data-extractor.py` stores the files.

As you want to preprocess a larger amount of data files, it will scale better using [Cloud Dataflow](https://cloud.google.com/dataflow/).
> NOTE: this will incur charges on your Google Cloud Platform project. See [Dataflow pricing](https://cloud.google.com/dataflow/pricing).
```bash
PROJECT=$(gcloud config get-value project)
python preprocess.py \
  --project $PROJECT \
  --runner DataflowRunner \
  --setup_file ./setup.py \
  --work-dir $WORK_DIR
```

### Training the Model
> Source code: [`trainer/task.py`](trainer/task.py)

We'll train a [Deep Neural Network Regressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) in [TensorFlow](https://www.tensorflow.org/).
This will use the preprocessed data stored within the working directory.
During the preprocessing stage, the Apache Beam pipeline transformed extracted all the features (counts of elements) and tf.Transform generated a graph of operations to normalize those features.

The TensorFlow model actually takes the unnormalized inputs (counts of elements), applies the tf.Transform's graph of operations to normalize the data, and then feeds that into our DNN regressor.

If the training dataset is too large, it will scale better to train on [AI Platform](https://cloud.google.com/ai-platform/).
> NOTE: this will incur charges on your Google Cloud Platform project. See [AI Platform pricing](https://cloud.google.com/ml-engine/docs/pricing).

```bash
JOB="cloudml_samples_molecules_$(date +%Y%m%d_%H%M%S)"
WORK_DIR=gs://<your-gcs-bucket>/cloudml-samples/molecules
gcloud ai-platform jobs submit training $JOB \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket $WORK_DIR \
  --runtime-version 2.4 \
  --python-version 3.7 \
  --region us-central1 \
  --stream-logs \
  -- \
  --work-dir $WORK_DIR

# To get the path of the trained model
EXPORT_DIR=$WORK_DIR/model/export/final
MODEL_DIR=$(gsutil ls -d "$EXPORT_DIR/*" | sort -r | head -n 1)
```

To visualize the training job, we can use [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
```bash
tensorboard --logdir $WORK_DIR/model
```

You can access the results at [`localhost:6006`](localhost:6006).

### Predictions

#### Option 1: Batch Predictions


#### Option 2: Streaming Predictions

## Cleanup

Finally, let's clean up all the Google Cloud resources used, if any.

```sh
# To delete the model and version
gcloud ai-platform versions delete $VERSION --model $MODEL
gcloud ai-platform models delete $MODEL

# To delete the inputs topic
gcloud pubsub topics delete molecules-inputs

# To delete the outputs topic
gcloud pubsub topics delete molecules-predictions

# To delete the working directory
gsutil -m rm -rf $WORK_DIR
```
