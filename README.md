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
```
  
**DO NOT ADD .env FILES TO YOUR GIT REPOSITORY**

Now you should install [ngrok](https://ngrok.com).

Lets run ngrok with the port 3000, if you install it in one of $PATH directories:
```ngrok http 3000```  
Or if you install it directly into your project directory:  
```./ngrok http 3000```  
After this step you will see the domain that ngrok service is providing you, 
something like `https://2be3ac1b2fed.ngrok.io`, remind this 

To be continued:
* Connect your uri with app  
* Install and run mongodb

