import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, set_default_openai_client, set_tracing_disabled, OpenAIChatCompletionsModel

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("Error: GEMINI_API_KEY not found.")
    exit(1)

# Gemini-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

set_default_openai_client(external_client)
set_tracing_disabled(True)

# Define the model (adjust if needed)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",  
    openai_client=external_client
)


def english_helper_agent():
    print("\n--- English Helper Agent ---")
    user_prompt = input("How can I help you today? Please describe your concern: ")

    agent = Agent(
        name="english_helper",
        instructions="""You are an English Helper Agent designed to assist users in learning English. Always respond in simple, easy-to-understand English.
Correct grammar mistakes politely, explain vocabulary with examples, and encourage users to keep practicing.""",


        model=model,
    )

    result = Runner.run_sync(agent, user_prompt)
    advice = result.final_output.strip()

    print("\n--- Advice from English Helper ---")
    print(advice)

    # Save to file
    with open("README.md", "a", encoding="utf-8") as f:
        f.write("### User Concern:\n")
        f.write(user_prompt + "\n\n")
        f.write("### English Helper Response:\n")
        f.write(advice + "\n\n---\n\n")

    print("\nAdvice saved to README.md.")

# Run
if __name__ == "__main__":
    english_helper_agent()