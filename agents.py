from autogen import UserProxyAgent, AssistantAgent
from autogen import config_list_from_json
import autogen

config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
)

user = UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)


therapist = AssistantAgent(
    name="therapist",
    system_message="You are an AI therapist, you understand the users message and help him or her to feel good̀. Ask them questions to analyse their mental health;",
    llm_config={"config_list": config_list},
)

groupchat = autogen.GroupChat(agents=[user, therapist],
                              messages=[],
                              speaker_selection_method="round_robin",
                              max_round=15)
chat_manager = autogen.GroupChatManager(groupchat=groupchat,
                                        llm_config={"config_list": config_list})

user.initiate_chat(chat_manager,
                    message="Hi, I have been feeling really down for the past few months. I'm struggling to keep up with my studies, and it feels like everything is just too overwhelming. I don't have much motivation to do anything, and I can't seem to shake off this constant sadness.")

