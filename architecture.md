# Interview Prep Suggester -- Architecture

## How It Works

```
User types their interview context
        |
        v
  [understand_interview_context] -- extracts role, urgency, preparedness
        |
        +---> [suggest_technical]    \
        |                            |
        +---> [suggest_behavioural]  +--> run in PARALLEL
        |                            |
        +---> [confidence_suggestion] /
        |
        v
  [pick_best_prep] -- reads all 3 suggestions, decides quick vs deep
        |
        +-- false --> [quick_prep] --> short practice plan
        |
        +-- true  --> [deep_prep]  --> deeper study session
        |
        v
  Final `final_plan` printed to user
```

## Interactive Mode

```
$ python interview_prep_graph.py

  =======================================================
    INTERVIEW PREP SUGGESTER
  =======================================================

    Tell me about the interview you're preparing for and I'll suggest a
    personalized prep plan just for you.
    Type 'quit' to exit.

    What interview are you preparing for? > I have a backend engineer interview next week
    ...graph runs...
    YOUR PERSONALIZED PRACTICE
    ...

    What interview are you preparing for? > quit
    Take care of yourself. Goodbye!
```

## Graph Structure (Detailed)

```
                    +-------+
                    | START |
                    +---+---+
                        |
                        v
     +------------------+------------------+
     | understand_interview_context        |
     |                                    |
     | Extracts role, urgency, and         |
     | preparedness from user text         |
     +------------------+------------------+
                        |
           PARALLEL FAN-OUT (3 edges)
          /             |              \
         v              v               v
+--------+---+ +--------+---------+ +-------------------+
| suggest    | | suggest          | | confidence_       |
| technical  | | behavioural      | | suggestion        |
|            | |                  | |                   |
| Top topics | | STAR stories,    | | Confidence habits |
| practice   | | behavioral prep  | | and mindset        |
| tasks      | | guidance         | | suggestions        |
+--------+---+ +--------+---------+ +-------------------+
         \              |               /
          FAN-IN (all 3 must finish)
                        |
                        v
          +-------------+-------------+
          |       pick_best_prep       |
          |                           |
          | Reads all 3 suggestions    |
          | Returns:                   |
          | {needs_deep_prep, reason}  |
          +-------------+-------------+
                        |
               CONDITIONAL EDGE
              route_after_decision()
                   /         \
                 /             \
               NO               YES
            (quick_prep)      (deep_prep)
                 |               |
  +--------------+--+   +--------+-------------+
  | quick_prep      |   | deep_prep             |
  |                 |   |                       |
  | Short practical |   | Structured longer     |
  | plan focused on |   | study plan with       |
  | top gaps        |   | technical, behavioral, |
  |                 |   | and confidence phases |
  +-----------------+   +-----------------------+
                 \               /
                  \             /
                   v           v
                   +----+----+
                   |   END   |
                   +---------+
```

## State Fields

```
InterviewPrepState
|
|-- user_input               <-- provided by the user
|-- technical_suggestion      <-- written by suggest_technical
|-- behavioural_suggestion    <-- written by suggest_behavioural
|-- confidence_suggestion     <-- written by confidence_suggestion
|-- job_role                  <-- extracted by understand_interview_context
|-- urgency_level             <-- extracted by understand_interview_context
|-- preparedness_level        <-- extracted by understand_interview_context
|-- needs_deep_prep           <-- written by pick_best_prep
|-- prep_reason               <-- written by pick_best_prep
|-- final_plan                <-- written by quick_prep OR deep_prep
|-- messages                  <-- appended by all nodes via operator.add
```

## LangGraph Concepts Used

| Concept | Where in Code | What It Does |
|---------|---------------|--------------|
| State (Pydantic) | `InterviewPrepState` class | Typed data that travels between nodes |
| Nodes | `understand_interview_context`, `suggest_technical`, `suggest_behavioural`, `confidence_suggestion`, `pick_best_prep`, `quick_prep`, `deep_prep` | Single-purpose functions that return state updates |
| Parallel Execution | Edges from `understand_interview_context` to three suggestion nodes | Runs multiple suggestion nodes concurrently |
| Fan-In | Edges from suggestion nodes into `pick_best_prep` | Waits for all suggestions before deciding |
| Conditional Edge | `route_after_decision()` | Routes to `quick_prep` or `deep_prep` based on state |
| Graph Compilation | `graph.compile()` | Turns the graph definition into a runnable app |
| Invocation | `app.invoke({...})` | Executes the graph with the initial state |
| Message Accumulation | `messages: Annotated[list, operator.add]` | Merges log messages from multiple nodes |

## Tech Stack

| Component | Purpose |
|-----------|---------|
| LangGraph | Graph orchestration, state, nodes, edges, compilation |
| LangChain | OpenAI wrapper via `ChatOpenAI` |
| OpenAI | LLM backend used by the node prompts |
| Pydantic | State validation and typed data model |
| python-dotenv | Loads the OpenAI API key from `.env` |

## File Structure

```
interview_prep_graph.py    Main code with graph and interactive loop
architecture.md             This architecture explanation
architecture.drawio         Diagram source file
requirements.txt            Python dependencies
.env.example                Example environment file
.gitignore                 Ignored local files
```
