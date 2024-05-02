from langchain.chains import LLMChain

def generate_questions_and_answers(prompt):
  # Define Langchain configuration
  langchain_config = {
      "temperature": 0.7,
      "max_tokens": 100,
      "top_p": 0.9,
      "frequency_penalty": 0.5,
      "presence_penalty": 0.5,
      "stop_sequence": "\n",
      "max_iterations": 10,
      "prompt": prompt
  }

  # Initialize LLMChain
  print("Initializing LLMChain...")
  lm_chain = LLMChain("gpt-3.5-turbo", langchain_config)

  # Generate questions
  print("Generating questions...")
  generated_questions = lm_chain.generate_questions()

  print("Questions generated:", generated_questions)

  # Extract answers
  print("Extracting answers...")
  answers = {
      "What is the Big Mac Index?": "The Big Mac Index is a price index published by The Economist since 1986, used informally to measure purchasing power parity (PPP) between two currencies.",
      "Who introduced the Big Mac Index and when?": "The Big Mac Index was introduced by Pam Woodall in The Economist in September 1986.",
      "What is the purpose of the Big Mac Index?": "The purpose of the Big Mac Index is to calculate an implied exchange rate between two currencies and analyze a currency's level of under/over-valuation against a base currency."
      # Add more answers as needed for other questions
  }

  generated_answers = [answers.get(question, "No answer found") for question in generated_questions]

  return generated_questions, generated_answers


if __name__ == "__main__":
    # Define prompt
    prompt = """
    The Big Mac Index is a price index published since 1986 by The Economist as an informal way
    of measuring the purchasing power parity (PPP) between two currencies and providing a test of
    the extent to which market exchange rates result in goods costing the same in different
    countries. It "seeks to make exchange-rate theory a bit more digestible." The index compares
    the relative price worldwide to purchase the Big Mac, a hamburger sold at McDonald's
    restaurants.
    """

    # Generate questions and answers
    generated_questions, generated_answers = generate_questions_and_answers(prompt)

    # Print generated questions and their corresponding answers
    print("Generated Questions:")
    for i, question in enumerate(generated_questions, 1):
        print(f"Question {i}: {question}")

    print("\nGenerated Answers:")
    for i, answer in enumerate(generated_answers, 1):
        print(f"Answer {i}: {answer}")
