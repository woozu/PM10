from flask import Flask, request, render_template
import lr, dt

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/home')
def hellohtml():
    return render_template("home.html")

@app.route('/lresult')
def lresult():
    result = lr.result()
    return render_template("lresult.html", data=result)

@app.route('/lresult_run')
def lresult_run():
    sample = lr.sample()
    predict=lr.predict()
    real=lr.real()
    errorv=lr.errorv()
    return render_template("lresult_run.html", sample=sample, predict=predict, real=real, errorv=errorv)

@app.route('/tree')
def tree():
    accuracy = dt.result_accuracy()
    best_acc = dt.result_best_accuracy()
    best_score = dt.result_best_scroe()
    best_params = dt.result_best_params()
    return render_template("tree.html", accuracy=accuracy, best_acc=best_acc, best_score=best_score, best_params=best_params)

@app.route('/tree_run')
def tree_run():
    sample = dt.sample()
    predict=dt.predict()
    real=dt.real()
    return render_template("tree_run.html", sample=sample, predict=predict, real=real)

@app.route('/index')
def index():
    return render_template("index.html")
    
if __name__=='__main__':
    app.run(debug=True)
