# Multi-Tool AI Agent

This project implements a sophisticated AI agent powered by language models and a suite of specialized tools. It leverages a graph-based architecture using LangGraph to intelligently route user requests to the most appropriate tool, ensuring accurate and well-sourced answers. The agent can perform a variety of tasks, including academic research, internet searches, weather lookups, mathematical calculations, and coding assistance.

## Features

* **Multi-Tool Capability**: Integrates several tools to handle a wide range of queries:
    * **arXiv Research**: Fetches and synthesizes information from academic papers on arXiv.
    * **Internet Search**: Conducts web searches for up-to-date information.
    * **Weather Forecast**: Provides real-time weather data for any location.
    * **Calculator**: Solves mathematical expressions.
    * **Code Assistant**: Helps with programming questions and generates code snippets.
    * **Date Checker**: Retrieves the current date.
* **Intelligent Routing**: Uses a language model to analyze the user's prompt and select the most suitable tool for the job[cite: 1].
* **Graph-Based Logic**: Built on LangGraph, it manages the conversation flow, tool execution, and model calls in a structured and stateful manner[cite: 1].
* **Stateful Conversations**: Remembers the context of the conversation using a `MemorySaver`, allowing for follow-up questions and more natural interactions[cite: 1].
* **Sourced-Based Answers**: Ensures that the information provided is traceable to its source, whether it's an arXiv paper, a web page, or a model-generated response.

***

## Project Structure


| File | Description |
| --- | --- |
| **`graph.py`** | This is the main entry point of the application. [cite_start]It defines the agent's state, the graph workflow, and manages the interaction between the user and the AI model[cite: 1]. |
| **`llm_tools.py`** | Contains the definitions for several core tools: `internet_search_tool`, `math_tool`, `code_assistant_tool`, `get_current_date_tool`, and `get_weather_tool`. |
| **`arxiv_tool.py`** | Defines the `arxiv_research_tool`, which performs a full RAG (Retrieval-Augmented Generation) pipeline on data from arXiv to answer research-related questions. |
| **`requirements.txt`** | Lists all the Python dependencies required to run the project. |

***

## Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

* Python 3.8 or higher
* An active Google AI Studio account with a `GOOGLE_API_KEY`
* A Tavily AI account with a `TAVILY_API_KEY`
* A WeatherAPI.com account with a `WEATHER_API_KEY`

### 2. Clone the Repository

First, clone the project repository to your local machine or download and extract the source files (`graph.py`, `llm_tools.py`, `arxiv_tool.py`, `requirements.txt`) into a dedicated project folder.

### 3. Install Dependencies

Navigate to the project directory in your terminal and install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
````

### 4\. Set Up Environment Variables

This project requires API keys for several services. You'll need to create a `.env` file in the root of your project directory to store these keys.

1.  Create a new file named `.env` in the project's root folder.

2.  Add the following lines to the `.env` file, replacing the placeholder text with your actual API keys:

    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
    WEATHER_API_KEY="YOUR_WEATHER_API_KEY"
    ```

### 5\. Run the Application

Once you've installed the dependencies and configured your API keys, you can start the agent by running the `graph.py` script:

```bash
python graph.py
```

[cite\_start]The application will initialize, and you will be prompted to enter your query in the console.

### 6\. Interact with the Agent

After the agent is ready, you can start asking questions. Here are a few examples of queries you can try:

  * **Research**: "What are the latest advancements in quantum machine learning?"
  * **Weather**: "What's the weather like in London today?"
  * **Coding**: "Can you write a Python function to reverse a string?"
  * **Math**: "What is 100 / (5 \* 2)?"
  * **Date**: "What is the current date?"

To end the conversation, simply type `exit` or `stop`.
