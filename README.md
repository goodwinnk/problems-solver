# problems-solver

# Installing

```
git clone https://github.com/goodwinnk/problems-solver.git
pip install -r requirements.txt
```

First you must get a tokens. Open the [list](https://api.slack.com/apps) of your applications and select one,
click "Install Application" and copy your bot-token from the opened page. The second one is signing token,
you can find it on "Basic information" page. 

The next step is to create secret.env file in the project directory,
and place there following line without \<brackets\>  
```
SLACK_BOT_TOKEN=<paste your bot token>
SLACK_SIGNING_TOKEN=<paste your signing token>
GOOGLE_APPLICATION_CREDENTIALS=<paste your translator token>
```
  
**DO NOT ADD .env FILES TO YOUR GIT REPOSITORY**

Now you should install [ngrok](https://ngrok.com).

Lets run ngrok with the port 3000, if you install it in one of $PATH directories:
```ngrok http 3000```  
Or if you install it directly into your project directory:  
```./ngrok http 3000```  
After this step you will see the domain that ngrok service is providing you, 
something like `https://2be3ac1b2fed.ngrok.io`, remind this 

Paste 'https://2be3ac1b2fed.ngrok.io/slack/events`' in  
YourApp -> Basic Information -> Add features and functionality:

    * Interactive components -> Interactivity -> Request URL
    * Event Subscriptions -> Request URL
    * Also turn on 'Enable Events' check.

Give permissions to the app,   
YourApp -> Basic Information -> Permissions:

    * app_mentions:read
    * channels:history
    * channels:read
    * chat:write
    * im:history
    * groups:history
    * links:read
    * mpim:history
    
Install your app to your workspace:  
YourApp -> Basic Information -> Install your app to your workspace -> Install

[Install MongoDB](https://docs.mongodb.com/manual/installation/)

Run MongoDB:

```
sudo systemctl start mongo
sudo systemctl status mongo # Check is it running

```

Run app:
```
cd problems-solver
python3 app.py
```

Dump data from db:
```
python3 extracter.py
```
The result will be in extracted foled.

make analysis/translation/research:
  * move from google drive data folder to problems-solver/nlp/data
  * run analysys.py
  * write your own scripts using nlp module auxiliary scripts
  
