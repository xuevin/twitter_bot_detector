from flask import Flask, render_template, request, redirect, session, jsonify, url_for
from celery import Celery
import random 
import os
import time
from TwitterClassifier import *


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'



celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@app.route('/', methods=['GET','POST'])
def index():
    if (request.method == 'GET'):
        return render_template('index.html')
    else:
        #twitterUserName = request.form['twitterUserName']
        #task = scoreUser.apply_async(args=[twitterUserName])
        return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/scoreUserPage',methods=['GET','POST'])
def scoreUserPage():
    if request.method == 'GET':
        return "oops"
    else:
        pass
        #request.form['twitterUserName']
        #executeScript = "<script> chart.load({columns: [['data', 22]]}); </script>"
        #return render_template('index.html',scoreUser=executeScript)

@celery.task(bind=True)
def scoreUser(self,userName):

    self.update_state(state='PROGRESS',
                          meta={'current': 0,
                                'total': 1,
                                'status': "Received User"})

    SN,prediction,probs, encodedUser= scoreUsers(getUserDF(userName))

    predictionProb = round(100*probs[0][0],2)

    return {'current': predictionProb,
            'total': 100,
            'status': 'Task completed!',
            'result': userName }
    

@app.route('/longtask', methods=['POST'])
def longtask():

    twitterUserName = "DEFAULT"
    
    twitterUserName = request.form['twitterUser']
    task = scoreUser.apply_async(args=[twitterUserName])

    return jsonify({}), 202, {'Location': url_for('taskstatus', task_id=task.id)}


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = scoreUser.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=33507,debug=1)
