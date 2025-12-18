import getpass
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=getpass.getpass("Enter IBM Cloud API Key: ")
)

SPACE_ID = "ddc9382d-4ff5-455e-bc52-6890c2a892db"


MODEL_ID = "ibm/granite-3-8b-instruct"

PARAMS = {
    "decoding_method": "greedy",
    "max_new_tokens": 300,
    "temperature": 0.5
}


model = ModelInference(
    model_id=MODEL_ID,
    credentials=credentials,
    space_id=SPACE_ID,
    params=PARAMS
)


def fitness_buddy(question):
    prompt = f"""
You are a fitness assistant.
Provide a clear workout and diet plan.

User question:
{question}
"""
    response = model.generate_text(prompt)
    return response

if __name__ == "__main__":
    print("\n🏋️ FITNESS BUDDY AI")
    q = input("Ask your fitness question: ")

    result = fitness_buddy(q)

    print("\nAI Response:\n")
    print(result)
