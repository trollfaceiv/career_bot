from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import gradio as gr
from pypdf import PdfReader

load_dotenv(override=True)


pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(message):
    print (f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message":message}
    requests.post(pushover_url, data=payload)

def record_user_details(email, name="Name not provided", notes="not provided"):
    push (f"Recording interest from {name}, with {email}, and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

record_user_details_json = {
    "name" : "record_user_details",
    "description" : "Use this tool to record that a user is interested in bein in touch and provided an email address",
    "parameters" : {
        "type" : "object",
        "properties" : {
            "email": {
                "type" : "string",
                "description" : "The email address of this user"
            },
            "name" : {
                "type" : "string",
                "description" : "The user's name, if they provided it"
            },
            "notes" : {
                "type" : "string",
                "description" : "Any additional notes about the contact"
            }
        },
    "required" : ["email"],
    "additionalProperties" : False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}


tools = [{"type": "function", "function": record_unknown_question_json},
         {"type": "function", "function": record_user_details_json}]

class Me:
    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.model_name = "gemini-2.0-flash"
        self.name = "Gennaro Rascato"
        base = os.path.dirname(__file__)             
        self.profile_path = os.path.join(base, "me", "Profile.pdf")
        summary_path = os.path.join(base, "me", "summary.txt")
        self.cv_path = os.path.join(base, "me", "CV.pdf")
        self.linkedin_url = "https://www.linkedin.com/in/gennaro-rascato"
        self.github_url   = "https://github.com/trollfaceiv"
        reader = PdfReader(self.profile_path)
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open(summary_path, "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_calls(self, tool_calls):
        results=[]
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print (f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
        particularly questions related to {self.name}'s career, background, skills and experience. \
        Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
        You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
        Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
        If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt


    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model=self.model_name, messages=messages, tools=tools)
            finish_reason = response.choices[0].finish_reason
            if finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_calls(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content



if __name__ == "__main__":
    me = Me()

    with gr.Blocks(title="Career Assistant – Gennaro") as demo:
        gr.Markdown("## 💼 Assistente di **Gennaro Rascato**")

        with gr.Row():
            # Colonna sinistra: Chat (INVARIATA nella logica)
            with gr.Column(scale=3):
                gr.ChatInterface(
                me.chat,
                type="messages",
                chatbot=gr.Chatbot(label="Chat", height=520, type="messages"),
                textbox=gr.Textbox(placeholder="Scrivi un messaggio…", label="Messaggio"),
                submit_btn="Invia 🚀"
            )


            # Colonna destra: About me
            with gr.Column(scale=2, min_width=320):
                gr.Markdown("### 👤 About me")
                gr.Markdown(
                    f"- 🌐 **LinkedIn:** [{me.linkedin_url}]({me.linkedin_url})  \n"
                    f"- 💻 **GitHub:** [{me.github_url}]({me.github_url})"
                )
                gr.DownloadButton(
                    label="⬇️ Scarica CV (PDF)",
                    value=me.cv_path
                )

                # opzionale: mostra anche un estratto del sommario
                with gr.Accordion("Mostra sommario profilo", open=False):
                    gr.Markdown(me.summary[:1200] + ("..." if len(me.summary) > 1200 else ""))

    demo.launch()

