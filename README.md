# Smol🦎

**Smol** is our project for the Hackathons of Mistral AI. The project is a platform that allows users to automatically create a small fine-tuned model from a pre-trained model on a specific task. We used a **new approach never seen before in the literature** to create a small fine-tuned model from a pre-trained model.
The approach is based on the idea of using a LLM (Mistral Large for example) to generate a dataset that will be used to fine-tune a small model (Mistral Small for example). We will then train our small model with a DPO (Differential Preference Optimization) algorithm. We then use the new pre-trained model to generate a new dataset. We keep only the samples that are far from the decision of the large model. Finally we fine-tune the small model with the new dataset. We repeat this process until we reach the limits set by the user. The user can then use the fine-tuned model by a endpoint give by the platform.

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