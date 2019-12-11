import os
import slackbot.bot as bot
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

WATSON_API_KEY = str(os.getenv('WATSONAPIKEY_WATSON-BOT'))
WATSON_ASSISTANT_ID = str(os.getenv('WATSONASSISTANTID_WATSON-BOT'))

@bot.listen_to(r'.+')
def reply_from_watson(message):
    text = message.body['text']
    message.reply(text+'って言った？')

    response = assistant.message(
        assistant_id=WATSON_ASSISTANT_ID,
        session_id=session_id,
        input={
            'message_type': 'text',
            'text': text
        }
    ).get_result()

    message.reply(response['output']['generic'][0]['text'])


authenticator = IAMAuthenticator(WATSON_API_KEY)
assistant = AssistantV2(
    version='2019-02-28',
    authenticator=authenticator
)
assistant.set_service_url('https://gateway-tok.watsonplatform.net/assistant/api')

session_id = assistant.create_session(
    assistant_id=WATSON_ASSISTANT_ID
).get_result()['session_id']
print(session_id)