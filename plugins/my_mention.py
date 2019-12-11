import os
import slackbot.bot as bot

# @respond_to('string')     bot宛のメッセージ
#                           stringは正規表現が可能 「r'string'」
# @listen_to('string')      チャンネル内のbot宛以外の投稿
#                           @botname: では反応しないことに注意
#                           他の人へのメンションでは反応する
#                           正規表現可能
# @default_reply()          DEFAULT_REPLY と同じ働き
#                           正規表現を指定すると、他のデコーダにヒットせず、
#                           正規表現にマッチするときに反応
#                           ・・・なのだが、正規表現を指定するとエラーになる？

# message.reply('string')   @発言者名: string でメッセージを送信
# message.send('string')    string を送信
# message.react('icon_emoji')  発言者のメッセージにリアクション(スタンプ)する
#                               文字列中に':'はいらない

@bot.respond_to('こんにちは')
def hello(message):
    message.reply('こんにちは！')

@bot.listen_to('またね')
def seeyou(message):
    message.send('誰かがでてくね')
    message.reply('またね')

# @bot.listen_to(r'.+')
# def all_listen(message):
#     text = message.body['text']
#     message.reply(text+'って言った人がいる')