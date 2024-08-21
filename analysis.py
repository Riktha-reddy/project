from autogen import UserProxyAgent, AssistantAgent, config_list_from_json
import autogen
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

class SentimentAnalyzer:
    def __init__(self, age):
        self.sia = SentimentIntensityAnalyzer()
        self.conversation = []
        self.sentiment_scores = []
        self.age = age
        self.exam_seasons = []
        self.stress_reasons = []
        self.timestamps = []

    def analyze_sentiment(self, text, is_exam_season=False, stress_reason=None):
        sentiment = self.sia.polarity_scores(text)
        self.conversation.append(text)
        self.sentiment_scores.append(sentiment['compound'])
        self.exam_seasons.append(is_exam_season)
        self.stress_reasons.append(stress_reason)
        self.timestamps.append(datetime.now())
        return sentiment

    def generate_report(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Sentiment over time
        ax1.plot(self.timestamps, self.sentiment_scores, marker='o')
        ax1.set_title('Sentiment Analysis Report')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Sentiment Score')
        ax1.axhline(y=0, color='r', linestyle='--')
        
        # Highlight exam seasons
        for i, is_exam in enumerate(self.exam_seasons):
            if is_exam:
                ax1.axvline(x=self.timestamps[i], color='yellow', alpha=0.3)
        
        # Stress reasons
        unique_reasons = list(set(filter(None, self.stress_reasons)))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_reasons)))
        for reason, color in zip(unique_reasons, colors):
            mask = [r == reason for r in self.stress_reasons]
            ax2.scatter(np.array(self.timestamps)[mask], np.array(self.sentiment_scores)[mask], 
                        label=reason, color=color)
        
        ax2.set_title('Sentiment Scores by Stress Reason')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Sentiment Score')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('sentiment_report.png')
        plt.close()

        average_sentiment = sum(self.sentiment_scores) / len(self.sentiment_scores)
        
        report = f"Therapy Session Report\n"
        report += f"Age: {self.age}\n"
        report += f"Number of messages: {len(self.conversation)}\n"
        report += f"Average sentiment score: {average_sentiment:.2f}\n"
        report += f"Exam seasons: {sum(self.exam_seasons)}\n"
        report += f"Most common stress reason: {max(set(self.stress_reasons), key=self.stress_reasons.count)}\n\n"
        report += "Conversation summary:\n"
        for i, (message, score, timestamp, is_exam, reason) in enumerate(zip(
            self.conversation, self.sentiment_scores, self.timestamps, self.exam_seasons, self.stress_reasons)):
            report += f"{i+1}. User: {message}\n"
            report += f"   Timestamp: {timestamp}\n"
            report += f"   Sentiment: {score:.2f}\n"
            report += f"   Exam Season: {'Yes' if is_exam else 'No'}\n"
            report += f"   Stress Reason: {reason if reason else 'Not specified'}\n\n"

        def generate_suggestions(self):
            suggestions = {
                "Academic Stress": [
                    "Create a study schedule to manage your time effectively.",
                    "Take regular breaks to avoid burnout.",
                    "Seek help from teachers or classmates if you're struggling with a subject.",
                    "Practice relaxation techniques such as deep breathing or meditation."
                ],
                "Work-Related Stress": [
                    "Prioritize your tasks and focus on one thing at a time.",
                    "Communicate with your manager about your workload and seek support if needed.",
                    "Take short breaks throughout the day to rest and recharge.",
                    "Engage in physical activity or hobbies outside of work to relieve stress."
                ],
                "Relationship Stress": [
                    "Communicate openly and honestly with the people involved.",
                    "Spend time with supportive friends and family members.",
                    "Set healthy boundaries to protect your well-being.",
                    "Consider seeking advice or counseling to work through relationship issues."
                ],
                "Health Stress": [
                    "Follow a healthy diet and get regular exercise.",
                    "Practice mindfulness and relaxation techniques.",
                    "Seek medical advice and follow treatment plans as prescribed.",
                    "Connect with support groups or mental health professionals."
                ],
                "General stress": [
                    "Practice mindfulness or meditation.",
                    "Engage in regular physical activity.",
                    "Maintain a healthy work-life balance.",
                    "Seek support from friends, family, or a therapist."
                ]
            }

            report = "Suggested Ways to Overcome Stress:\n"
            for reason in set(self.stress_reasons):
                if reason:
                    report += f"\n{reason}:\n"
                    for suggestion in suggestions.get(reason, ["Consider talking to a therapist for personalized advice."]):
                        report += f"  - {suggestion}\n"
        return report

def round_robin_selection(agents, messages):
    index = len(messages) % len(agents)
    return agents[index]


config_list = config_list_from_json("OAI_CONFIG_LIST")

user = UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)

therapist = AssistantAgent(
    name="therapist",
    system_message="You are an AI therapist, you understand the users message and help him or her to feel good. Ask them questions to analyse their mental health;",
    llm_config={"config_list": config_list},
)

groupchat = autogen.GroupChat(agents=[user, therapist],
                              messages=[],
                              speaker_selection_method=round_robin_selection,
                              max_round=5)
chat_manager = autogen.GroupChatManager(groupchat=groupchat,
                                        llm_config={"config_list": config_list})

# Initialize the sentiment analyzer
# Initialize the sentiment analyzer with an age (e.g., 20)
sentiment_analyzer = SentimentAnalyzer(age=20)

# Initial message to start the conversation
initial_message = "Hi, I have been feeling really down for the past few months. I'm struggling to keep up with my studies, and it feels like everything is just too overwhelming. I don't have much motivation to do anything, and I can't seem to shake off this constant sadness."

# Function to determine if it's exam season (you can implement your own logic)
def is_exam_season():
    # For example, you could check if the current month is typically an exam month
    return datetime.now().month in [5, 6, 11, 12]

# Function to extract stress reason (you would need to implement this based on message content)
def extract_stress_reason(message):
    # This is a placeholder. You'd need to implement logic to extract the stress reason from the message.
    if "Exams"or "grades" or "project" or "deadlines" or "Balancing" or "studies" or "responsibilities" or "examination" in message:
        return "Academic Stress"
    elif "high workload" or "job"or "insecurity"or "Lack" or "satisfaction" or "working hours" or "conflict" or "colleagues" in message:
        return "Work-Related Stress"
    elif "family members" or "family" or "loneliness" or "isolation" or "friends" or "partners" or "social" in message:
        return "Relationship Stress"
    elif "illness" or "injury" or "anxiety" or "depression" or "personal health" in message:
        return "Health Stress"
    return "General stress"

# Analyze the initial message
# sentiment_analyzer.analyze_sentiment(
#     initial_message,
#     is_exam_season=is_exam_season(),
#     stress_reason=extract_stress_reason(initial_message)
# )

# Run the chat
user.initiate_chat(therapist, message=initial_message)
content_main = ""

#print("user chat: ", user.chat_messages)
#print('------')
data = user.chat_messages
# print(data["assistant_agent"])
#print('---------')
# After the conversation, analyze sentiment for each user message
assistant_content = []

# Iterating through the defaultdict to access the messages
for key, messages in data.items():
    for message in messages:
        if message['role'] == 'assistant':
            assistant_content.append(message['content'])


# Print the extracted content
for content in assistant_content:
    sentiment_analyzer.analyze_sentiment(
            content,
            is_exam_season=is_exam_season(),
            stress_reason=extract_stress_reason(content)
        )

# Generate the report
    try:
        report = sentiment_analyzer.generate_report()
        print("\nGenerating report...")
        print(report)

        with open('therapy_report.txt', 'w') as f:
            f.write(report)

        print("Report saved as 'therapy_report.txt'")
        print("Sentiment graph saved as 'sentiment_report.png'")
    except Exception as e:
        print(f"An error occurred: {e}")
