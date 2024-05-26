#!/usr/bin/env python
from fastapi import FastAPI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langserve import add_routes
from langchain_mistralai import ChatMistralAI
import dotenv

dotenv.load_dotenv()

app = FastAPI(
    title="Smol Server",
    version="1.0",
    description="Smol inference server",
)


add_routes(
        app,
        HuggingFacePipeline.from_model_id(
            model_id="./dpo_mistral",
            task="text-generation",
            device=0,
            model_kwargs={"do_sample": True},
            batch_size=4,
            pipeline_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.2,
                "repetition_penalty": 1.1,
            },
        ),
        path="/smol",
)


add_routes(
    app,
    ChatMistralAI(model= "open-mistral-7b"),
    path="/mistral_small"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)