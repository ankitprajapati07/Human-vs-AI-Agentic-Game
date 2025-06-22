# 🧠 Human vs AI: Agentic Game

Welcome to **Human vs AI**, a fun and interactive game powered by Agentic AI workflows using **LangChain** and **LangGraph**. This project demonstrates how agents can interact with humans in a guessing game — using intelligent strategies to either guess a number or a word.


---
## 🧱 System Architecture Overview (Just for your information)

### 1. Frontend (Web UI)
- Presents a simple interface to:
  - Show the main menu.
  - Accept and display user interactions during gameplay.
- UI supports:
  - Game selection input
  - Question-answer interactions
  - Game result messages

### 2. LangGraph Application (Core Logic)
- Implements a multi-agent architecture using LangGraph:
  - **Game Selector Agent** – routes the game flow.
  - **Number Game Agent** – uses binary search logic to guess numbers.
  - **Word Game Agent** – uses descriptive clue-based narrowing and guessing.
- Manages state transitions and agent-specific memory.
- Tracks how many games were played during a session.

### 3. Database Layer
- Persists data such as:
  - Game history
  - Total number of games played
  - Win/loss statistics
- Implemented using JSON file

## 🕹 Game Overview

This game allows a human to compete against an AI in two modes:

### 1. 🔢 Number Game
- The user secretly picks a number between **1 and 50**.
- The AI uses **binary search logic** to guess the number.
- It asks questions like: "Is it greater than 25?" , "Is it less than 10?" and so on.
- Once it guesses correctly:
  - It congratulates the user.
  - Increments the Number Game counter.
  - Returns to the **main menu**.

### 2. 🧩 Word Clue Guesser
- The user selects a word from a **predefined list**:  
  `["apple", "chair", "elephant", "guitar", "rocket", "pencil", "pizza", "tiger"]`
- The AI is allowed to ask **up to 5 yes/no/maybe questions** to gather clues.
- Then it attempts to guess the word:
  - If **correct**: Celebrates and returns to the menu.
  - If **wrong**: Offers the option to retry or return to the menu.
- Each attempt increases the Word Game counter.

---

## 🧠 Tech Stack

- **LangChain** – for building intelligent language agent workflows.
- **LangGraph** – for stateful agent orchestration.
- **Python** – for core logic and interaction.
- Optional: Streamlit/CLI if UI is added (not included in this version).

---

## ✅ How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://gitlab.com/git_python_lr/gen-ai-practical-task.git
   cd gen-ai-practical-task
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Create a `.env` file by copying from `.env.sample`:
   ```bash
   cp .env.sample .env
   ```
   Then, add the respective values.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the game (adjust based on your interface):
   ```bash
   python main.py
   ```

