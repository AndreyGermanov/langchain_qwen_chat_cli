# Qwen Chat Langchain RAG CLI

## About

This is a command line Chat Bot, that uses locally installed [Qwen Chat 7B model](https://huggingface.co/Qwen/Qwen-7B-Chat) to respond to requests. To process queries, it uses text files located in the "texts" folder as a knowledge base. During run, the application creates vector embeddings from these files and stores these embeddings to Chroma database. Then, when receive the question from user, it selects top 10 text chunks from this database, which are close to this question by cosine similarity and injects these texts as a context to the prompt, that LLM uses to generate an answer for the user.

By default, the "texts" folder contains content of all articles of [ReadyTensor Agentic AI Developer Certification Program](https://app.readytensor.ai/hubs/ready_tensor_certifications) until 11 Jun 2025. So, you can ask questions about the first module of this program.

Example chat:

```
Enter query: How to start learning?

To start learning, you should sign up for a free account on Ready Tensor if you haven't already done so. Then, enroll in the program and navigate to the Certifications hub to request access. After your request is approved, you will have immediate access to program materials, including weekly lectures, reading materials, and project guidelines. You can also use the lectures, tools, or other resources you prefer to learn.

Enter query: What is an objective of the module 1 project?

To build the first project - a question-answering assistant using core concepts of agent architectures, retrieval-augmented generation (RAG) and tool use.

Enter query: What should I deliver to complete the project?

The deliverable for the project is a simple RAG-based question-answering or document-assistant app. This means you should create an application that uses the RAG (Retrieve And Generate) system to answer questions or assist with documents.

Enter query: When should I submit the first project?

Your first project submission is due by June 13, 2025, at 11:59 PM UTC.
```

The chat bot is limited to the context. If the query is not related to the data from the `texts` folder, it will respond something like this: 

```
The given context does not contain an answer to your question. It's recommended to provide more context if you need further assistance.
```

You can add more text files to the `texts` folder to extend knowledge base of the Chat Bot and restart the program to start chatting about the content of these text files.

# Install

1. Clone this repository.
2. Move to the root folder of cloned repository.
3. Create a Python virtual environment.

```bash
python -m venv myenv
```

4. Activate the environment

```bash
source myenv/bin/activate
```

5. Install required packages

```bash
pip install -r requirements.txt

```

# Use

Run the program using the following command:

```bash
python app.py
```

During first run it should download the Qwen Chat 7B LLM model from HuggingFace Hub, which can take some time.

Then, you should see the prompt `Enter query: `.

Type your questions and press `Enter` key to send them to LLM and receive answers. You can enter any number of questions.

To exit the program type `exit` as a question or press `Ctrl+C`.

# Uninstall

1. Deactivate the virtual environment.

```
deactivate
```

2. Delete the folder with the application.