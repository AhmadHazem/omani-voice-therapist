import os
import asyncio


from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_tavily import TavilySearch
from langchain.callbacks.base import AsyncCallbackHandler

from pydantic import BaseModel, Field

## ====================================================================================================================|
##                                              API KEYS and Models Intailization                                      |
##=====================================================================================================================|
## Intializng OpenAI
openai_model = "gpt-4o"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
GPT4o = ChatOpenAI(temperature = 0, model= openai_model, streaming = True).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

## Intailzing Claude
claude_model = "claude-3-7-sonnet-latest"
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
ClaudeSonnet3_7 = ChatAnthropic(temperature= 0, model_name= claude_model, streaming= True).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

## Azure Speech Services API Key
SPEECH_KEY =  os.getenv("SPEECH_KEY")
SPEECH_ENDPOINT = os.getenv("ENDPOINT")


## ====================================================================================================================|
##                                              Tool Creation and Intialization                                        |
##=====================================================================================================================|
tavily_search = TavilySearch(
    max_results=5,
    topic="general",
)

@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

# Define the multiply tool
@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y

# Define the exponentiate tool
@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y

@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x

def Notify(email):
    """Notify Authorities if patient emotional Health is in critical conditions or he/she are mentioning about harming themselves or others"""
    return "Notified Police and Called an Ambulance, and " + email

@tool
def health_searcher(query : str):
    """Use this function strictly when the user asks for you to search about mental health therapy techniques, nearest mental hospitals, 
    mental health medications in Oman, hotlines and nothing else"""
    return tavily_search.invoke(query)

@tool
def final_answer(answer: str, tools_used: list[str]) -> str:
    """Use this tool to provide a final answer to the user.
    The answer should be in natural language as this will be provided
    to the user directly. The tools_used must include a list of tool
    names that were used within the `scratchpad`.
    """
    return {"answer": answer, "tools_used": tools_used}

tools = [add, multiply, exponentiate, subtract, final_answer, health_searcher]
name2tool = {tool.name : tool.func for tool in tools}

## ====================================================================================================================|
##                                     Prompts Templates creation for system and human                                 |
##=====================================================================================================================|
SYSTEM_PROMPT_RISK_ANALYZER = \
(
    "You are an Omani AI Therapist, and you're task is to provide solutions for the users emotional issues. \n"
    "Also, You're task is to analyze the user's prompt emotion/feeling.\n"
    "You will try to include, if posssible, omani and islamic values for the cultural context for how to solve this problem.\n"
    "You will fill 8 variables: requires_analysis, user_emotion, severity, diagnosis, user_prompt, cbt_technique, reason_for_technique, cultural_context \n"
    "If you did not manage to fill any of those variables just put None, and do not add anything else"
    "You will reply in arabic and be brief as possible"
)

SYSTEM_PROMPT_THERAPIST = \
(
    "You are an Omani AI Therapist, and you will recieve a structured enhanced prompt of patients.\n"
    "Your task is reply back to the user in authentic omani arabic"
    "You must adhere to omani gulf culture, and Islamic context, and not mention anything that contradicts with those values"
    "You can use Islamic versus if it is needed but do not over use it"
    "If variable required_analysis is False, reply back normally to the user without usage of the variables you will recieve about him/her"
    "You must use the following summary dowm below about the user and adapot your response to it"
    "Patient Summary :\n" \
    "requires_analysis: {requires_analysis}\n"
    "user_emotion: {user_emotion}\n"
    "severity:{severity}\n"
    "diagnosis:{diagnosis}\n"
    "cbt_technique:{cbt_technique}\n"
    "reason_for_technique:{reason_for_technique}\n"
    "cultural_context:{cultural_context}\n"
    "Mention the CBT Technique needed, and provide life examples for how to apply it, why it will work."
    "(user_prompt:{user_prompt}) only to the tool as args"
    "Use Notify tool if the user exhibits any critically concerning prompt like harming himself or others"
    "You MUST then use the final_answer tool to provide a final answer to the user. "
    "DO NOT use the same tool more than once."
)
#---------------------------------------------------- Human Prompts --------------------------------------------------------

HUMAN_PROMPT_RISK_ANALYZER = \
(
    "{patient_prompt}"
)

HUMAN_PROMPT_ENHANCED = \
(
    "This is the user's prompt : {user_prompt}"
)
# ----------------------------------------------- Template used by the LLM ----------------------------------------------------

Prompt_Template_Step_One = ChatPromptTemplate.from_messages([
    ("system",  SYSTEM_PROMPT_RISK_ANALYZER),
    ("user", HUMAN_PROMPT_RISK_ANALYZER)
])

Prompt_Template_Step_Two = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_THERAPIST),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", HUMAN_PROMPT_ENHANCED)
])
# ------------------------------------------------------- Structured Output for Step 1 --------------------------------------------------
# Output For Step 1
class StructuredRiskAssessorTemplate(BaseModel):
    requires_analysis: bool = Field(description= "This varaible will contain whether the user's prompt requires analysis if he/she is in" \
                                                " emotional pain. It will be true if user is in emotional pain, and false if it is a" \
                                                "normal prompt or not emotional pain is detected")
    user_emotion: str = Field(description=
                                    "This is a variable, where will you fill it with your predication about the user's " \
                                    "emotion based on his/her prompt.")
    severity: str = Field(description=
                                    "This is a variable, where will you fill it with you predication about the severity" \
                                    "of the emotion - it should be eith LOW/MED/HIGH/CRIT.")
    diagnosis: str = Field(description= 
                                    "This variable will contain the diagnosis behind this user's emotion. " \
                                    "It has to be a Hierarchical Diagnosis something similar to those examples but not explicitly from فاثة: " \
                                    "الحالات النفسية السلوكية - مشاكل النوم - أحلام اليقظة" \
                                    "الأمراض الذهانية - مشاكل الفصام والذهان - الفصام" \
                                    "الأمراض المزاجية - الاكتئاب - الاكتئاب الحاد" \
                                    "الأمراض القلقية - القلق العام - اضطراب الهلع" \
    
                                    "If not mentioned, then state that it is non mentioned."
                                    "")
    user_prompt: str = Field(description= 
                                    "This variable will contain the user's original prompt, " \
                                    "so just copy it as it is and put in that variable.")
    cbt_technique: str = Field(description="This variable will contain the suggested (CBT) to help the user pick one of the CBT techniques here. Do not keep using CBT اعادة الهيكلة المعرفية" )
    reason_for_technique: str = Field(description= "This varaible will contain the reason behind picking that CBT, and how can it help. Explain more how it can help in no more than 100 words and provide examples on how to apply it")
    cultural_context: str = Field(description="This variable will contain how can Omani and Islamic values can help the user in his emotional problem")

## ====================================================================================================================|
##                                      Agents Intialization - Memory - Streaming                                      |
##=====================================================================================================================|
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def format_messages(self) -> None:
        for i in range(len(self.messages)):
            if (isinstance(self.messages[i], ToolMessage)):
                start = max(0, i - 2)
                end = i + 1
                self.messages = self.messages[:start] + self.messages[end:]
                return

    def clear(self) -> None:
        self.messages = []

chat_map = {}
def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = InMemoryHistory()
    return chat_map[session_id]

#--------------------------------------------------- Streaming Class -----------------------------------------------------
class QueueCallbackHandler(AsyncCallbackHandler):
    """Callback handler that puts tokens into a queue."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()

            if token_or_done == "<<DONE>>":
                # this means we're done
                return
            if token_or_done:
                yield token_or_done

    async def on_llm_new_token(self, *args, **kwargs) -> None:
        """Put new token in the queue."""
        #print(f"on_llm_new_token: {args}, {kwargs}")
        chunk = kwargs.get("chunk")
        if chunk:
            # check for final_answer tool call
            if tool_calls := chunk.message.additional_kwargs.get("tool_calls"):
                if tool_calls[0]["function"]["name"] == "final_answer":
                    # this will allow the stream to end on the next `on_llm_end` call
                    self.final_answer_seen = True
        self.queue.put_nowait(kwargs.get("chunk"))
        return

    async def on_llm_end(self, *args, **kwargs) -> None:
        """Put None in the queue to signal completion."""
        #print(f"on_llm_end: {args}, {kwargs}")
        # this should only be used at the end of our agent execution, however LangChain
        # will call this at the end of every tool call, not just the final tool call
        # so we must only send the "done" signal if we have already seen the final_answer
        # tool call
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")
        return
    

queue = asyncio.Queue()
streamer = QueueCallbackHandler(queue)

#--------------------------------------------- Therapist Agent Creation --------------------------------------------------
class Therpaist:
    chat_history: list[BaseMessage]

    def __init__(self, model, max_iterations = 3):
        self.chat_history = []
        self.model = model
        self.max_iterations = max_iterations
        self.model_riskAnalyzer = model.with_structured_output(StructuredRiskAssessorTemplate)
        self.chain_one: RunnableSerializable = \
        (
            Prompt_Template_Step_One |
            self.model_riskAnalyzer |
            {
                "requires_analysis": lambda x: x.requires_analysis,
                "user_emotion": lambda x: x.user_emotion,
                "severity": lambda x: x.severity,
                "diagnosis": lambda x:x.diagnosis,
                "user_prompt": lambda x:x.user_prompt,
                "cbt_technique": lambda x:x.cbt_technique,
                "reason_for_technique" : lambda x:x.reason_for_technique,
                "cultural_context": lambda x:x.cultural_context
            } 
        )


        self.chain_two : RunnableSerializable = \
        (
            {
                "requires_analysis": lambda x: x['requires_analysis'],
                "user_emotion": lambda x: x['user_emotion'],
                "severity": lambda x: x['severity'],
                "diagnosis": lambda x:x['diagnosis'],
                "user_prompt": lambda x:x['user_prompt'],
                "cbt_technique": lambda x:x['cbt_technique'],
                "reason_for_technique" : lambda x:x['reason_for_technique'],
                "cultural_context": lambda x:x['cultural_context'],
                "chat_history": lambda x:x.get('chat_history', []),
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            } |
            Prompt_Template_Step_Two |
            self.model.bind_tools(tools, tool_choice="any")
        )
        

        self.agent = RunnableWithMessageHistory (
            runnable= self.chain_two,
            get_session_history= get_chat_history,
            input_messages_key="user_prompt",
            history_messages_key="chat_history"
        ) 


    async def invoke_step1(self, input: str) -> dict:
        return await self.chain_one.ainvoke({"patient_prompt": input})
    

    async def invoke_step2(self, input:dict , session_id:str, verbose:bool = False):
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        agent_scratchpad = []
        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            async def stream(query: dict):
                response = self.agent.with_config(
                    callbacks=[streamer]
                )
                # we initialize the output dictionary that we will be populating with
                # our streamed output
                output = None
                # now we begin streaming
                input = query 
                async for token in response.astream(input = input , config= {"session_id" : session_id}):
                    if output is None:
                        output = token
                    else:
                        # we can just add the tokens together as they are streamed and
                        # we'll have the full response object at the end
                        output += token
                    if token.content != "":
                        # we can capture various parts of the response object
                        if verbose: print(f"content: {token.content}", flush=True)
                    tool_calls = token.additional_kwargs.get("tool_calls")
                    if tool_calls:
                        if verbose: print(f"tool_calls: {tool_calls}", flush=True)
                        tool_name = tool_calls[0]["function"]["name"]
                        if tool_name:
                            if verbose: print(f"tool_name: {tool_name}", flush=True)
                        arg = tool_calls[0]["function"]["arguments"]
                        if arg != "":
                            if verbose: print(f"arg: {arg}", flush=True)
                if output.tool_calls and len(output.tool_calls) > 0:
                    return AIMessage(
                        content=output.content,
                        tool_calls=output.tool_calls,
                        tool_call_id=output.tool_calls[0]["id"]
                    )
                else:
                    return AIMessage(content=output.content)


            tool_call = await stream(query=input)
            # add initial tool call to scratchpad
            #agent_scratchpad.append(tool_call)
            # otherwise we execute the tool and add it's output to the agent scratchpad
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_call_id
            tool_out = name2tool[tool_name](**tool_args)
            # add the tool output to the agent scratchpad
            tool_exec = ToolMessage(
                content=f"{tool_out}",
                tool_call_id=tool_call_id
            )
            #agent_scratchpad.append(tool_exec)
            chat_map[session_id].add_messages([tool_exec])
            count += 1
            # if the tool call is the final answer tool, we stop
            if tool_name == "final_answer":
                break
        # add the final output to the chat history, we only add the "answer" field
        final_answer = tool_out["answer"]
        chat_map[session_id].add_messages([
            HumanMessage(content=input['user_prompt']),
            AIMessage(content=final_answer)
        ])


        chat_map[session_id].format_messages()
        return final_answer


    def invoke(self, input: str, session_id:str):
        analysis = self.chain_one.invoke({
            "patient_prompt": input
        })   

        
        out = self.agent.invoke(analysis, config={"session_id" : session_id})
        return out
          
          
