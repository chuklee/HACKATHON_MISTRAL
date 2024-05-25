# Smol🦎

**Smol** is our project for the Hackathons of Mistral AI. The project is a platform that allows users to automatically create a small fine-tuned model from a pre-trained model on a specific task. We used a **new approach never seen before in the literature** to create a small fine-tuned model from a pre-trained model.

## Key Features
The core idea involves using a Large Language Model (e.g., Mistral Large) to generate a dataset tailored for fine-tuning a smaller model (e.g., Mistral Small). The small model is then trained using a Differential Preference Optimization (DPO) algorithm. Here's a breakdown of the process:
1. **Initial Dataset Generation**: The large and small model generates a dataset.

2. **First Fine-tuning**: The small model is initially trained on this dataset using the DPO algorithm.

3. **Dataset Refinement**: The large and refined small model generates a new dataset.

4. **Sample Selection**: Only samples that differ significantly from the large model’s decisions are retained.

5. **Iterative Fine-tuning**: The small model undergoes further fine-tuning with this new dataset.

6. **Repetition**: Steps 3-5 are repeated until user-defined limits are met.

To summarize, the platform allows users to automatically create a small fine-tuned model from a pre-trained model on a specific task. The user can then use the fine-tuned model by a endpoint give by the platform.

It opens the door to a new and fastest way to create agent specialized in a specific task.

## Participants
- Eithan Nakache
- Martin Natale
- Ilyas Oulkadda
- Camil Ziane

## Installation
This project use python 3.10.12, you can create a virtual environment with the following command:

```bash
conda create -n smol python=3.10.12
```

Then, you can activate the environment with:

```bash
conda activate smol
```

You can then install the dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Run the project
There are two servers to run. The first one is the server that will receive the requests from the client side and the second one is the server that will run the automatic fine-tuning pipeline.

To run the server that will receive the requests from the client side, you can run the following command:

```bash
streamlit run 1_🏠_Home.py
```

To run the server that will run the automatic fine-tuning pipeline, you can run the following command:

```bash
python main.py
```