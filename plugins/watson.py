import os
import slackbot.bot as bot
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from plugins import classify
from plugins import functions
from plugins import consts

WATSON_API_KEY = str(os.getenv('WATSONAPIKEY_WATSON-BOT'))
WATSON_ASSISTANT_ID = str(os.getenv('WATSONASSISTANTID_WATSON-BOT'))

# Reply
def reply_from_watson(message):
    text = message.body['text']

    response = assistant.message(
        assistant_id=WATSON_ASSISTANT_ID,
        session_id=session_id,
        input={
            'message_type': 'text',
            'text': text
        }
    ).get_result()

    for res in response['output']['generic']:
        while(True):
            if res['response_type'] == 'text':
                res = res['text']
                break
            elif res['response_type'] == 'suggestion':
                res = res['suggestions'][0]['output']['generic'][0]    
        message.reply(str(res))    


    ans = model.classify([text],w2v)[3]
    print('ans : '+ans)
    print('response : ')
    print(response)

# Response
@bot.listen_to(r'.+')
def listen_watson(message):
    reply_from_watson(message)

@bot.respond_to(r'.+')
def respond_watson(message):
    reply_from_watson(message)

# Start session
authenticator = IAMAuthenticator(WATSON_API_KEY)
assistant = AssistantV2(
    version='2019-02-28',
    authenticator=authenticator
)
assistant.set_service_url('https://gateway-tok.watsonplatform.net/assistant/api')

session_id = assistant.create_session(
    assistant_id=WATSON_ASSISTANT_ID
).get_result()['session_id']
print('Session ID: '+str(session_id))

# Building Classify class
w2v = functions.load_w2v(consts.W2V_PATH)
model = classify.Classify(3)

print('Build Complete.')